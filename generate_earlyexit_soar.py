import torch
import numpy as np
import torch.nn.functional as F
import time
from transformers import AutoTokenizer, AutoModel
import json

def should_early_exit(current_step, max_steps, answer_gap, thresholds=None):
    """Phase-aware early exit strategy."""
    if answer_gap is None:
        return False
    
    # Use default or provided thresholds
    if thresholds is None:
        thresholds = {'early': 7.5, 'mid': 5.0, 'late': 2.5}
    
    progress = current_step / max_steps
    
    # Phase-based thresholds
    if progress < 0.33:  # Early phase
        return answer_gap >= thresholds.get('early', 7.5)
    elif progress < 0.67:  # Mid phase  
        return answer_gap >= thresholds.get('mid', 5.0)
    else:  # Late phase
        return answer_gap >= thresholds.get('late', 2.5)


def add_gumbel_noise(logits, temperature):
    '''Gumbel-Max sampling with float64 precision.'''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''Calculate tokens to transfer at each step for uniform denoising.'''
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    
    return num_transfer_tokens


@torch.no_grad()
@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, max_beam_size=2, log=False, 
             logits_eos_inf=False, confidence_eos_eot_inf=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        max_beam_size: Maximum beam size for dynamic beam search.
    '''
    print("======SOAR + Prophet, temperature: {:.1f}====".format(temperature))
    
    # ========== SOAR 配置参数 ==========
    confidence_threshold = 0.95      # 高置信度阈值，超过则并行解码
    min_parallel_tokens = 1          # 并行解码最小token数
    max_parallel_tokens = 5          # 并行解码最大token数
    
    # ========== Prophet 配置参数 ==========
    early_exit_thresholds = {
        'min_step': 0.3,      # 最小步数比例（相对于总步数）
        'min_gap': 2.0,       # 最小平均gap阈值
        'window_size': 5      # 答案区域检查窗口大小
    }
    answer_token_id = None     # 答案起始token，需要时设置
    
    # ========== 初始化 ==========
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    # SOAR: 初始化beam
    beam = [(x.clone(), 0.0, 0, [])]  # (sequence, cumulative_log_prob, current_block, records)
    prompt_index = (x != mask_id)
    
    # Prophet: 早停状态
    early_exit_triggered = False
    exit_decision_step = None
    
    # 生成长度相关
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    total_steps = steps
    
    global_step = 0
    
    if log:
        print(f"=== SOAR + Prophet Generation Start ===")
        print(f"Total blocks: {num_blocks}, Steps per block: {steps_per_block}")
        print(f"Max beam size: {max_beam_size}, Confidence threshold: {confidence_threshold}")
        print(f"Early exit thresholds: {early_exit_thresholds}")
    
    # ========== 主循环 ==========
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        
        for block_step in range(steps_per_block):
            global_step += 1
            
            # 检查所有beam中的序列是否还有mask
            has_remaining_masks = False
            for seq, _, _, _ in beam:
                if (seq == mask_id).any():
                    has_remaining_masks = True
                    break
            
            if not has_remaining_masks:
                if log:
                    print(f"No masks remaining, early stopping at step {global_step}")
                break
            
            # 批量收集所有beam中的序列
            beam_sequences = [seq for seq, _, _, _ in beam]
            batch_sequences = torch.cat(beam_sequences, dim=0)
            
            # 批量计算logits
            with torch.no_grad():
                if cfg_scale > 0.:
                    unconditional_seqs = []
                    for seq in beam_sequences:
                        un_seq = seq.clone()
                        un_seq[prompt_index] = mask_id
                        unconditional_seqs.append(un_seq)
                    unconditional_batch = torch.cat(unconditional_seqs, dim=0)
                    combined_batch = torch.cat([batch_sequences, unconditional_batch], dim=0)
                    batch_logits = model(combined_batch).logits
                    conditional_logits, unconditional_logits = torch.chunk(batch_logits, 2, dim=0)
                    batch_logits = unconditional_logits + (cfg_scale + 1) * (conditional_logits - unconditional_logits)
                else:
                    batch_logits = model(batch_sequences).logits
            
            if logits_eos_inf:
                batch_logits[:, :, 126081] = -torch.inf
            
            logits_with_noise = add_gumbel_noise(batch_logits, temperature=temperature)
            batch_x0 = torch.argmax(logits_with_noise, dim=-1)
            
            if confidence_eos_eot_inf:
                batch_logits[:, :, 126081] = batch_logits[:, :, 126348] = -torch.inf
            
            if remasking == 'low_confidence':
                p = F.softmax(batch_logits, dim=-1)
                batch_x0_p = torch.gather(p, dim=-1, index=batch_x0.unsqueeze(-1)).squeeze(-1)
            else:
                batch_x0_p = torch.rand(batch_x0.shape, device=batch_x0.device)
            
            # ========== Prophet 早停检查 ==========
            if not early_exit_triggered and answer_token_id is not None:
                # 找到answer token的位置
                answer_positions = []
                for seq, _, _, _ in beam:
                    answer_mask = (seq[0] == answer_token_id)
                    if answer_mask.any():
                        answer_idx = torch.where(answer_mask)[0][0].item()
                        answer_positions.append(answer_idx)
                
                if answer_positions:
                    avg_answer_pos = sum(answer_positions) / len(answer_positions)
                    answer_start = int(avg_answer_pos)
                    
                    # 计算答案区域的logits gap
                    answer_length = early_exit_thresholds.get('window_size', 5)
                    answer_end = min(answer_start + answer_length, batch_logits.shape[1])
                    
                    if answer_start < batch_logits.shape[1]:
                        answer_logits = batch_logits[:, answer_start:answer_end, :]
                        answer_gaps = []
                        for t in range(answer_logits.shape[1]):
                            top2_vals, _ = torch.topk(answer_logits[0, t, :], k=2, dim=-1)
                            gap = (top2_vals[0] - top2_vals[1]).item()
                            answer_gaps.append(gap)
                        
                        if answer_gaps:
                            avg_answer_gap = sum(answer_gaps) / len(answer_gaps)
                            min_step_ratio = early_exit_thresholds.get('min_step', 0.3)
                            min_step = int(total_steps * min_step_ratio)
                            
                            if avg_answer_gap >= early_exit_thresholds.get('min_gap', 2.0) and global_step >= min_step:
                                if log:
                                    print(f"Prophet early exit at step {global_step}/{total_steps}, gap={avg_answer_gap:.3f}")
                                early_exit_triggered = True
                                exit_decision_step = global_step
                                
                                # 填充所有剩余的mask
                                for seq_idx, (seq, cum_log_prob, curr_block, records) in enumerate(beam):
                                    mask_idx = (seq == mask_id)
                                    seq[mask_idx] = batch_x0[seq_idx][mask_idx]
                                
                                # 跳出循环
                                break
            
            # ========== SOAR: 候选生成 ==========
            new_beam_candidates = []
            has_multi_unmask_candidate = False
            
            for beam_idx, (seq, cumulative_log_prob, current_block, records) in enumerate(beam):
                if not (seq == mask_id).any():
                    new_beam_candidates.append((seq, cumulative_log_prob, current_block, records))
                    continue
                
                block_start_pos = prompt.shape[1] + current_block * block_length
                block_end_pos = prompt.shape[1] + (current_block + 1) * block_length
                
                mask_index = (seq == mask_id)
                confidence = torch.where(mask_index, batch_x0_p[beam_idx:beam_idx+1], -np.inf)
                confidence[:, block_end_pos:] = -np.inf
                
                # 获取当前block内的mask位置和置信度
                block_mask_positions = torch.where(mask_index[0, block_start_pos:block_end_pos])[0] + block_start_pos
                if len(block_mask_positions) == 0:
                    new_current_block = min(current_block + 1, num_blocks - 1)
                    new_beam_candidates.append((seq, cumulative_log_prob, new_current_block, records))
                    continue
                
                block_mask_confidence = confidence[0, block_mask_positions]
                
                # SOAR: 策略选择 - 检查高置信度token
                high_confidence_mask = block_mask_confidence >= confidence_threshold
                high_confidence_indices = torch.where(high_confidence_mask)[0]
                
                if len(high_confidence_indices) >= min_parallel_tokens:
                    # 策略(1): 并行解码多个高置信度token
                    num_to_unmask = min(len(high_confidence_indices), max_parallel_tokens)
                    top_probs, top_indices = torch.topk(block_mask_confidence[high_confidence_indices], num_to_unmask)
                    selected_indices = high_confidence_indices[top_indices]
                    
                    new_seq = seq.clone()
                    new_log_prob = cumulative_log_prob
                    new_records = records.copy()
                    
                    for idx in range(num_to_unmask):
                        original_idx = selected_indices[idx].item()
                        pos = block_mask_positions[original_idx].item()
                        token = batch_x0[beam_idx, pos].item()
                        prob = top_probs[idx].item()
                        
                        new_seq[0, pos] = token
                        new_log_prob += prob
                        new_records.append({
                            "step": global_step,
                            "position": pos,
                            "confidence": prob,
                            "token_id": token
                        })
                    
                    new_current_block = current_block
                    if new_current_block < num_blocks - 1:
                        current_block_mask = (new_seq[:, block_start_pos:block_end_pos] == mask_id)
                        if not current_block_mask.any():
                            new_current_block += 1
                    
                    new_beam_candidates.append((new_seq, new_log_prob, new_current_block, new_records))
                    has_multi_unmask_candidate = True
                    
                else:
                    # 策略(2): Beam search探索
                    k = min(max_beam_size, len(block_mask_confidence))
                    top_probs, top_indices = torch.topk(block_mask_confidence, k)
                    top_positions = block_mask_positions[top_indices]
                    top_tokens = batch_x0[beam_idx, top_positions]
                    
                    for idx in range(k):
                        new_seq = seq.clone()
                        pos = top_positions[idx].item()
                        token = top_tokens[idx].item()
                        prob = top_probs[idx].item()
                        
                        new_seq[0, pos] = token
                        new_log_prob = cumulative_log_prob + prob
                        
                        new_current_block = current_block
                        if new_current_block < num_blocks - 1:
                            current_block_mask = (new_seq[:, block_start_pos:block_end_pos] == mask_id)
                            if not current_block_mask.any():
                                new_current_block += 1
                        
                        new_records = records.copy()
                        new_records.append({
                            "step": global_step,
                            "position": pos,
                            "confidence": prob,
                            "token_id": token
                        })
                        
                        new_beam_candidates.append((new_seq, new_log_prob, new_current_block, new_records))
            
            # 检查是否早停触发
            if early_exit_triggered:
                break
            
            # SOAR: Beam更新
            if not new_beam_candidates:
                break
            
            new_beam_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 去重
            uniq_candidates = []
            seen = set()
            for tensor, log_prob, block_progress, records in new_beam_candidates:
                tensor_tuple = tuple(tensor.flatten().cpu().numpy().tolist())
                if tensor_tuple not in seen:
                    seen.add(tensor_tuple)
                    uniq_candidates.append((tensor, log_prob, block_progress, records))
            
            # 动态调整beam size
            if has_multi_unmask_candidate and uniq_candidates:
                best_candidate = uniq_candidates[0]
                best_seq, best_log_prob, best_block, best_records = best_candidate
                
                original_mask_count = (beam[0][0] == mask_id).sum().item()
                current_mask_count = (best_seq == mask_id).sum().item()
                masks_unmasked = original_mask_count - current_mask_count
                
                if masks_unmasked >= min_parallel_tokens:
                    beam = [best_candidate]
                    if log:
                        print(f"Parallel decoded {masks_unmasked} tokens, beam reduced to 1")
                else:
                    beam_size = min(max_beam_size, len(uniq_candidates))
                    beam = uniq_candidates[:beam_size]
            else:
                beam_size = min(max_beam_size, len(uniq_candidates))
                beam = uniq_candidates[:beam_size]
        
        if early_exit_triggered:
            break
    
    # ========== 返回结果 ==========
    if beam:
        best_sequence, best_score, _, best_records = beam[0]
    else:
        best_sequence = x
        best_records = []
    
    if log:
        print(json.dumps(best_records))
        print(f"Records count: {len(best_records)}")
        if early_exit_triggered:
            print(f"Prophet early exit at step {exit_decision_step}/{total_steps}")
    
    return best_sequence

def main():
    device = 'cuda'
    
    model = AutoModel.from_pretrained('/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct', trust_remote_code=True)
    
    prompt = "What is 25 + 37?"
    m = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    
    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    # Test with early exit
    out, gap_data = generate(
        model, input_ids, 
        steps=128, 
        gen_length=128, 
        block_length=32, 
        temperature=0., 
        cfg_scale=0., 
        remasking='low_confidence',
        analyze_gap=True,
        answer_start_pos=input_ids.shape[1] + 100,  # Estimated answer position
        early_exit_thresholds={'early': 7.5, 'mid': 5.0, 'late': 2.5}
    )
    
    generated_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(f"Generated: {generated_text}")
    print(f"Exit info: {gap_data['exit_info']}")


if __name__ == '__main__':
    main()