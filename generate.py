import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


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


@ torch.no_grad()
def generate_origin(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
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
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336,
             constraints=None, log=False, logits_eos_inf=False,
             confidence_eos_eot_inf=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
    '''
    import json

    print("======greedy, temperature: {:.1f}====".format(temperature))

    x = torch.full(
        (1, prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long
    ).to(model.device)

    x[:, :prompt.shape[1]] = prompt.clone()

    # Apply constraints
    if constraints is not None:
        for pos, token_id in constraints.items():
            absolute_pos = prompt.shape[1] + pos
            if absolute_pos < x.shape[1]:
                x[:, absolute_pos] = token_id

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # 初始化 records 列表
    records = []

    # ===== statistics =====
    forward_count = 0
    decoding_steps_used = 0

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            mask_index = (x == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)

                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # 统计 model forward 调用次数
            forward_count += 1

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = -torch.inf
                logits_with_noise[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(
                        p,
                        dim=-1,
                        index=torch.unsqueeze(x0, -1)
                    ),
                    -1
                )  # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # 当前 block 之后的位置不能被解码
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # 当前 step 的 mask positions 和 confidence
            mask_positions = torch.where(mask_index[0])[0]
            mask_confidence = confidence[0, mask_positions]

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

            selected_positions = []
            selected_confidences = []

            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(
                    confidence[j],
                    k=num_transfer_tokens[j, i]
                )

                transfer_index[j, select_index] = True

                selected_positions.extend(
                    select_index.cpu().float().detach().numpy()
                )
                selected_confidences.extend(
                    confidence[j, select_index].cpu().float().detach().numpy()
                )

                # 为每个选择的 position 添加记录
                for pos, conf in zip(select_index, confidence[j, select_index]):
                    pos_int = pos.item()
                    token = x0[j, pos].item()
                    conf_float = conf.item()

                    records.append({
                        "step": i + 1,
                        "block": num_block + 1,
                        "position": pos_int,
                        "confidence": conf_float,
                        "token_id": token
                    })

            if log:
                print(f"Selected positions: {selected_positions}")
                print(f"Selected confidences: {selected_confidences}")

            # 统计本 step 实际解码 token 数
            decoded_this_step = int(transfer_index.sum().item())
            if decoded_this_step > 0:
                decoding_steps_used += 1

            # 更新序列
            x[transfer_index] = x0[transfer_index]

            # Maintain constraints
            if constraints is not None:
                for pos, token_id in constraints.items():
                    absolute_pos = prompt.shape[1] + pos
                    if absolute_pos < x.shape[1]:
                        x[:, absolute_pos] = token_id

    # ===== final statistics =====
    decoded_tokens = len(records)
    tpf = decoded_tokens / forward_count if forward_count > 0 else 0.0
    avg_tokens_per_decoding_step = (
        decoded_tokens / decoding_steps_used if decoding_steps_used > 0 else 0.0
    )

    print("====== Decoding Statistics ======")
    print(f"Decoded tokens: {decoded_tokens}")
    print(f"Model forward calls: {forward_count}")
    print(f"Steps: {decoding_steps_used}")
    print(f"TPF (tokens per forward): {tpf:.4f}")
    print(f"TPS (tokens per decoding step): {avg_tokens_per_decoding_step:.4f}")

    # 输出 records 作为 JSON
    print(json.dumps(records))
    print(len(records))

    return x


def generate_full_confidence(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336,
             constraints=None, log=False, logits_eos_inf=False,
             confidence_eos_eot_inf=False,
             print_all_token_records=True):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        print_all_token_records:
            If True, print a JSON object containing both selected records
            and all candidate token confidence traces.
    '''
    import json

    print("======greedy, temperature: {:.1f}====".format(temperature))

    x = torch.full(
        (1, prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long
    ).to(model.device)

    x[:, :prompt.shape[1]] = prompt.clone()

    # Apply constraints
    if constraints is not None:
        for pos, token_id in constraints.items():
            absolute_pos = prompt.shape[1] + pos
            if absolute_pos < x.shape[1]:
                x[:, absolute_pos] = token_id

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # selected-only records, same as before
    records = []

    # NEW: all candidate token confidence records
    all_token_records = []

    # ===== statistics =====
    forward_count = 0
    decoding_steps_used = 0

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            global_step = num_block * steps_per_block + i + 1
            local_step = i + 1

            mask_index = (x == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)

                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            forward_count += 1

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = -torch.inf
                logits_with_noise[:, :, 126348] = -torch.inf

            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(
                        p,
                        dim=-1,
                        index=torch.unsqueeze(x0, -1)
                    ),
                    -1
                )  # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # 当前 block 之后的位置不能被解码
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            # x0 only proposes tokens for mask positions; fixed positions stay unchanged
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

            selected_positions = []
            selected_confidences = []

            # 先选择本 step 要 unmask 的 token
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(
                    confidence[j],
                    k=num_transfer_tokens[j, i]
                )

                transfer_index[j, select_index] = True

                selected_positions.extend(
                    select_index.cpu().float().detach().numpy()
                )
                selected_confidences.extend(
                    confidence[j, select_index].cpu().float().detach().numpy()
                )

            # NEW:
            # 记录当前 block 内所有仍为 mask 的候选 token 的 top-1/confidence
            # 注意：不记录未来 block，因为未来 block 此时不应该参与当前 block 解码。
            current_block_mask_positions = (
                torch.where(mask_index[0, block_start:block_end])[0] + block_start
            )

            selected_position_set = set(
                torch.where(transfer_index[0])[0].detach().cpu().tolist()
            )

            for pos in current_block_mask_positions:
                pos_int = int(pos.item())
                token = int(x0[0, pos_int].item())
                conf_float = float(confidence[0, pos_int].item())
                is_selected = pos_int in selected_position_set

                all_token_records.append({
                    "global_step": global_step,
                    "local_step": local_step,
                    "block": num_block + 1,
                    "position": pos_int,
                    "block_relative_position": pos_int - block_start,
                    "generation_relative_position": pos_int - prompt.shape[1],
                    "confidence": conf_float,
                    "token_id": token,
                    "selected": is_selected,
                })

            # 保留原来的 selected-only records
            for j in range(confidence.shape[0]):
                selected_for_j = torch.where(transfer_index[j])[0]

                for pos in selected_for_j:
                    pos_int = int(pos.item())
                    token = int(x0[j, pos].item())
                    conf_float = float(confidence[j, pos].item())

                    records.append({
                        "global_step": global_step,
                        "local_step": local_step,
                        "step": local_step,  # backward compatible
                        "block": num_block + 1,
                        "position": pos_int,
                        "block_relative_position": pos_int - block_start,
                        "generation_relative_position": pos_int - prompt.shape[1],
                        "confidence": conf_float,
                        "token_id": token
                    })

            if log:
                print(f"Selected positions: {selected_positions}")
                print(f"Selected confidences: {selected_confidences}")

            decoded_this_step = int(transfer_index.sum().item())
            if decoded_this_step > 0:
                decoding_steps_used += 1

            # 更新序列
            x[transfer_index] = x0[transfer_index]

            # Maintain constraints
            if constraints is not None:
                for pos, token_id in constraints.items():
                    absolute_pos = prompt.shape[1] + pos
                    if absolute_pos < x.shape[1]:
                        x[:, absolute_pos] = token_id

    # ===== final statistics =====
    decoded_tokens = len(records)
    tpf = decoded_tokens / forward_count if forward_count > 0 else 0.0
    avg_tokens_per_decoding_step = (
        decoded_tokens / decoding_steps_used if decoding_steps_used > 0 else 0.0
    )

    print("====== Decoding Statistics ======")
    print(f"Decoded tokens: {decoded_tokens}")
    print(f"Model forward calls: {forward_count}")
    print(f"Steps: {decoding_steps_used}")
    print(f"TPF (tokens per forward): {tpf:.4f}")
    print(f"TPS (tokens per decoding step): {avg_tokens_per_decoding_step:.4f}")
    print(f"All candidate token records: {len(all_token_records)}")

    if print_all_token_records:
        # 推荐用 JSON object，避免你的旧 parser 误把 all_token_records 当 selected records。
        print(json.dumps({
            "selected_records": records,
            "all_token_records": all_token_records,
            "stats": {
                "decoded_tokens": decoded_tokens,
                "model_forward_calls": forward_count,
                "steps": decoding_steps_used,
                "tpf": tpf,
                "tokens_per_decoding_step": avg_tokens_per_decoding_step,
                "num_all_token_records": len(all_token_records),
            }
        }))
    else:
        # 兼容旧逻辑：只输出 selected-only records
        print(json.dumps(records))
        print(len(records))

    return x

def generate_adaptive_parallel(model, prompt, steps=128, gen_length=128,
        block_length=128, temperature=0.,
        cfg_scale=0.,
        remasking='low_confidence',
        mask_id=126336,
        log=False,
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,
        confidence_threshold=0.90,
        min_parallel_tokens=1,
        max_parallel_tokens=100,
        **kwargs):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
    '''

    import json

    # 配置参数
    confidence_threshold = confidence_threshold
    min_parallel_tokens = min_parallel_tokens
    max_parallel_tokens = max_parallel_tokens

    # 初始化序列
    x = torch.full(
        (1, prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long
    ).to(model.device)

    x[:, :prompt.shape[1]] = prompt.clone()

    current_seq = x.clone()
    current_block = 0
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # 初始化 records 列表
    records = []

    # ===== statistics =====
    forward_count = 0
    decoding_steps_used = 0

    if log:
        print("=== Confidence-based Parallel Decoding Start ===")
        print(f"Total blocks: {num_blocks}, Steps per block: {steps_per_block}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Initial mask count: {(x == mask_id).sum().item()}")

    for global_step in range(steps):
        # 检查序列是否还有 mask
        if not (current_seq == mask_id).any():
            if log:
                print(f"No masks remaining, early stopping at step {global_step + 1}")
            break

        if log:
            print(f"=== Step {global_step + 1}/{steps} ===")

        # 计算 logits
        with torch.no_grad():
            if cfg_scale > 0.:
                unconditional_seq = current_seq.clone()
                unconditional_seq[prompt_index] = mask_id

                combined_seq = torch.cat([current_seq, unconditional_seq], dim=0)
                combined_logits = model(combined_seq).logits

                conditional_logits, unconditional_logits = torch.chunk(
                    combined_logits,
                    2,
                    dim=0
                )

                logits = unconditional_logits + (
                    cfg_scale + 1
                ) * (conditional_logits - unconditional_logits)
            else:
                logits = model(current_seq).logits

        if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

        if confidence_eos_eot_inf:
            logits_with_noise[:, :, 126081] = -torch.inf
            logits_with_noise[:, :, 126348] = -torch.inf
        # 统计 model forward 调用次数
        forward_count += 1

        # 添加 Gumbel 噪声并获取预测
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        if remasking == 'low_confidence':
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(
                p,
                dim=-1,
                index=x0.unsqueeze(-1)
            ).squeeze(-1)
        elif remasking == 'random':
            x0_p = torch.rand(x0.shape, device=x0.device)
        else:
            raise NotImplementedError(remasking)

        # 当前 block 范围
        block_start = prompt.shape[1] + current_block * block_length
        block_end = prompt.shape[1] + (current_block + 1) * block_length

        # 获取 mask 和 confidence
        mask_index = (current_seq == mask_id)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        # 限制在当前 block 及之前
        confidence[:, prompt.shape[1] + (current_block + 1) * block_length:] = -np.inf

        # 当前 block 内的 mask 位置和置信度
        block_mask_positions = (
            torch.where(mask_index[0, block_start:block_end])[0] + block_start
        )

        block_mask_confidence = confidence[0, block_mask_positions]

        # 高置信 token
        high_confidence_mask = block_mask_confidence > confidence_threshold
        high_confidence_indices = torch.where(high_confidence_mask)[0]

        if len(high_confidence_indices) >= min_parallel_tokens:
            # 策略 1：并行解码多个高置信 token
            if log:
                print(f"Parallel decoding {len(high_confidence_indices)} tokens")

            num_to_unmask = min(
                len(high_confidence_indices),
                max_parallel_tokens
            )

            top_probs, top_indices = torch.topk(
                block_mask_confidence[high_confidence_indices],
                num_to_unmask
            )

            selected_indices = high_confidence_indices[top_indices]

            for idx in range(num_to_unmask):
                original_idx = selected_indices[idx].item()
                pos = block_mask_positions[original_idx].item()
                token = x0[0, pos].item()
                conf = block_mask_confidence[original_idx].item()

                # 更新序列
                current_seq[0, pos] = token

                records.append({
                    "step": global_step + 1,
                    "position": pos,
                    "confidence": conf,
                    "token_id": token,
                    "strategy": "parallel",
                    "block": current_block,
                    "parallel_group_size": num_to_unmask
                })

            # 本 global step 实际发生了一次 unmask
            decoding_steps_used += 1

            # 检查当前 block 是否完成
            if current_block < num_blocks - 1:
                current_block_mask = (
                    current_seq[:, block_start:block_end] == mask_id
                )
                if not current_block_mask.any():
                    current_block += 1

        else:
            # 策略 2：单个最高置信 token 解码
            if len(block_mask_confidence) > 0:
                top_prob, top_idx = torch.max(block_mask_confidence, dim=0)

                pos = block_mask_positions[top_idx].item()
                token = x0[0, pos].item()
                conf = top_prob.item()

                # 更新序列
                current_seq[0, pos] = token

                if log:
                    print(
                        f"Single token decoding: position {pos}, "
                        f"confidence {conf:.4f}"
                    )

                records.append({
                    "step": global_step + 1,
                    "position": pos,
                    "confidence": conf,
                    "token_id": token,
                    "strategy": "single",
                    "block": current_block
                })

                decoding_steps_used += 1

                # 检查当前 block 是否完成
                if current_block < num_blocks - 1:
                    current_block_mask = (
                        current_seq[:, block_start:block_end] == mask_id
                    )
                    if not current_block_mask.any():
                        current_block += 1

            else:
                # 当前 block 没有 mask，移动到下一个 block
                if current_block < num_blocks - 1:
                    current_block += 1
                else:
                    break

        if log:
            remaining = (current_seq == mask_id).sum().item()
            print(f"Remaining masks: {remaining}")

    if log:
        print("=== Generation Complete ===")
        print(f"Final mask count: {(current_seq == mask_id).sum().item()}")
        print(f"Total decoded tokens: {len(records)}")

    # ===== final statistics =====
    decoded_tokens = len(records)
    tpf = decoded_tokens / forward_count if forward_count > 0 else 0.0
    avg_tokens_per_decoding_step = (
        decoded_tokens / decoding_steps_used if decoding_steps_used > 0 else 0.0
    )

    parallel_token_count = sum(
        1 for r in records if r.get("strategy") == "parallel"
    )

    single_token_count = sum(
        1 for r in records if r.get("strategy") == "single"
    )

    # 统计 step 层面的 parallel/single
    step_to_strategies = {}
    step_to_group_size = {}

    for r in records:
        step = r["step"]
        step_to_strategies.setdefault(step, set()).add(r.get("strategy"))
        if r.get("strategy") == "parallel":
            step_to_group_size[step] = r.get("parallel_group_size", 1)
        else:
            step_to_group_size.setdefault(step, 1)

    parallel_step_count = 0
    single_step_count = 0

    for step, strategies in step_to_strategies.items():
        if "parallel" in strategies:
            parallel_step_count += 1
        elif "single" in strategies:
            single_step_count += 1

    print(json.dumps(records))
    print(len(records))

    print("====== Decoding Statistics ======")
    print(f"Decoded tokens: {decoded_tokens}")
    print(f"Model forward calls: {forward_count}")
    print(f"Actual decoding steps with unmask: {decoding_steps_used}")
    print(f"TPF (tokens per forward): {tpf:.4f}")
    print(f"Avg tokens per decoding step: {avg_tokens_per_decoding_step:.4f}")
    print(f"Parallel decoded tokens: {parallel_token_count}")
    print(f"Single decoded tokens: {single_token_count}")
    print(f"Parallel decoding steps: {parallel_step_count}")
    print(f"Single decoding steps: {single_step_count}")

    if len(step_to_group_size) > 0:
        avg_group_size = sum(step_to_group_size.values()) / len(step_to_group_size)
    else:
        avg_group_size = 0.0

    print(f"Avg selected tokens per active decoding step: {avg_group_size:.4f}")

    return current_seq


def generate_adaptive_parallel_full_confidence(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    cfg_scale=0.,
    remasking='low_confidence',
    mask_id=126336,
    log=False,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    confidence_threshold=0.90,
    min_parallel_tokens=1,
    max_parallel_tokens=100,
    constraints=None,
    print_all_token_records=True,
    **kwargs
):
    """
    Confidence-threshold adaptive parallel decoding with full confidence logging.

    Output format is consistent with generate_full_confidence:
      {
        "selected_records": [...],
        "all_token_records": [...],
        "stats": {...}
      }

    selected_records:
        Tokens actually committed/unmasked.

    all_token_records:
        At each decoding step, records every still-masked candidate token
        inside the current block, including its top-1 token, confidence,
        threshold, and whether it was selected.
    """

    import json

    # 初始化序列
    x = torch.full(
        (1, prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long
    ).to(model.device)

    x[:, :prompt.shape[1]] = prompt.clone()

    # Apply constraints
    if constraints is not None:
        for pos, token_id in constraints.items():
            absolute_pos = prompt.shape[1] + pos
            if absolute_pos < x.shape[1]:
                x[:, absolute_pos] = token_id

    current_seq = x.clone()
    current_block = 0
    prompt_index = (current_seq != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # selected-only records
    records = []

    # all candidate token records
    all_token_records = []

    # statistics
    forward_count = 0
    decoding_steps_used = 0

    # adaptive decoding does not have fixed local step from for-loop,
    # so maintain a local step counter for each block.
    block_local_steps = [0 for _ in range(num_blocks)]

    if log:
        print("=== Confidence-based Adaptive Parallel Decoding Start ===")
        print(f"Total blocks: {num_blocks}, Nominal steps per block: {steps_per_block}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Min parallel tokens: {min_parallel_tokens}")
        print(f"Max parallel tokens: {max_parallel_tokens}")
        print(f"Initial mask count: {(current_seq == mask_id).sum().item()}")

    for global_step_idx in range(steps):
        global_step = global_step_idx + 1

        # 如果已经没有 mask，提前停止
        if not (current_seq == mask_id).any():
            if log:
                print(f"No masks remaining, early stopping at step {global_step}")
            break

        # 当前 block 范围
        block_start = prompt.shape[1] + current_block * block_length
        block_end = prompt.shape[1] + (current_block + 1) * block_length

        # 如果当前 block 已经没有 mask，移动到下一个 block
        current_block_mask = current_seq[:, block_start:block_end] == mask_id
        if not current_block_mask.any():
            if current_block < num_blocks - 1:
                current_block += 1
                continue
            else:
                break

        # 当前 block 的 local step，1-indexed，和 generate_full_confidence 对齐
        block_local_steps[current_block] += 1
        local_step = block_local_steps[current_block]

        if log:
            print(f"=== Global Step {global_step}/{steps}, Block {current_block + 1}, Local Step {local_step} ===")

        # 计算 logits
        with torch.no_grad():
            if cfg_scale > 0.:
                unconditional_seq = current_seq.clone()
                unconditional_seq[prompt_index] = mask_id

                combined_seq = torch.cat([current_seq, unconditional_seq], dim=0)
                combined_logits = model(combined_seq).logits

                conditional_logits, unconditional_logits = torch.chunk(
                    combined_logits,
                    2,
                    dim=0
                )

                logits = unconditional_logits + (
                    cfg_scale + 1
                ) * (conditional_logits - unconditional_logits)
            else:
                logits = model(current_seq).logits

        forward_count += 1

        if logits_eos_inf:
            logits[:, :, 126081] = -torch.inf

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

        if confidence_eos_eot_inf:
            logits_with_noise[:, :, 126081] = -torch.inf
            logits_with_noise[:, :, 126348] = -torch.inf

        x0 = torch.argmax(logits_with_noise, dim=-1)

        if remasking == 'low_confidence':
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(
                p,
                dim=-1,
                index=x0.unsqueeze(-1)
            ).squeeze(-1)
        elif remasking == 'random':
            x0_p = torch.rand(x0.shape, device=x0.device)
        else:
            raise NotImplementedError(remasking)

        # 当前 mask 位置
        mask_index = (current_seq == mask_id)

        # x0 only proposes tokens for mask positions; fixed positions stay unchanged
        x0 = torch.where(mask_index, x0, current_seq)

        # confidence 只对 mask 位置有效
        confidence = torch.where(mask_index, x0_p, -np.inf)

        # 限制在当前 block 及之前，未来 block 不参与当前 block 解码
        confidence[:, prompt.shape[1] + (current_block + 1) * block_length:] = -np.inf

        # 当前 block 内仍为 mask 的 positions
        block_mask_positions = (
            torch.where(mask_index[0, block_start:block_end])[0] + block_start
        )

        if len(block_mask_positions) == 0:
            if current_block < num_blocks - 1:
                current_block += 1
                continue
            else:
                break

        block_mask_confidence = confidence[0, block_mask_positions]
        block_mask_token_ids = x0[0, block_mask_positions]

        # 高置信 token
        high_confidence_mask = block_mask_confidence > confidence_threshold
        high_confidence_indices = torch.where(high_confidence_mask)[0]

        # ------------------------------------------------------------
        # 先决定本 step selected indices，但先不更新 current_seq
        # selected_indices 是 block_mask_positions 内部的 index
        # ------------------------------------------------------------
        if len(high_confidence_indices) >= min_parallel_tokens:
            num_to_unmask = min(
                len(high_confidence_indices),
                max_parallel_tokens
            )

            _, top_indices = torch.topk(
                block_mask_confidence[high_confidence_indices],
                num_to_unmask
            )

            selected_indices = high_confidence_indices[top_indices]
            strategy = "parallel"
            parallel_group_size = num_to_unmask

            if log:
                print(
                    f"Parallel decoding {num_to_unmask} tokens "
                    f"from {len(high_confidence_indices)} candidates"
                )

        else:
            # fallback: 单个最高 confidence token
            top_prob, top_idx = torch.max(block_mask_confidence, dim=0)
            selected_indices = top_idx.view(1)
            strategy = "single"
            parallel_group_size = 1

            if log:
                pos = block_mask_positions[top_idx].item()
                conf = top_prob.item()
                print(
                    f"Single token fallback: position {pos}, "
                    f"confidence {conf:.4f}"
                )

        selected_index_set = set(selected_indices.detach().cpu().tolist())

        # ------------------------------------------------------------
        # FULL LOG:
        # 记录当前 block 内所有 remaining masked candidate tokens
        # 格式尽量和 generate_full_confidence 保持一致
        # ------------------------------------------------------------
        for original_idx in range(len(block_mask_positions)):
            pos_int = int(block_mask_positions[original_idx].item())
            token = int(block_mask_token_ids[original_idx].item())
            conf_float = float(block_mask_confidence[original_idx].item())
            is_selected = original_idx in selected_index_set

            all_token_records.append({
                "global_step": global_step,
                "local_step": local_step,
                "block": current_block + 1,
                "position": pos_int,
                "block_relative_position": pos_int - block_start,
                "generation_relative_position": pos_int - prompt.shape[1],
                "confidence": conf_float,
                "token_id": token,
                "selected": is_selected,
                # adaptive-specific extra fields
                "threshold": float(confidence_threshold),
                "strategy": strategy if is_selected else None,
                "remaining_masks_in_block": int(len(block_mask_positions)),
            })

        # ------------------------------------------------------------
        # Commit selected tokens + selected_records
        # ------------------------------------------------------------
        for idx in range(len(selected_indices)):
            original_idx = int(selected_indices[idx].item())

            pos_int = int(block_mask_positions[original_idx].item())
            token = int(block_mask_token_ids[original_idx].item())
            conf_float = float(block_mask_confidence[original_idx].item())

            current_seq[0, pos_int] = token

            records.append({
                "global_step": global_step,
                "local_step": local_step,
                "step": local_step,  # backward compatible
                "block": current_block + 1,
                "position": pos_int,
                "block_relative_position": pos_int - block_start,
                "generation_relative_position": pos_int - prompt.shape[1],
                "confidence": conf_float,
                "token_id": token,
                # adaptive-specific extra fields
                "threshold": float(confidence_threshold),
                "strategy": strategy,
                "parallel_group_size": int(parallel_group_size),
                "remaining_masks_in_block": int(len(block_mask_positions)),
            })

        decoding_steps_used += 1

        # Maintain constraints
        if constraints is not None:
            for pos, token_id in constraints.items():
                absolute_pos = prompt.shape[1] + pos
                if absolute_pos < current_seq.shape[1]:
                    current_seq[:, absolute_pos] = token_id

        # 检查当前 block 是否完成
        if current_block < num_blocks - 1:
            current_block_mask = current_seq[:, block_start:block_end] == mask_id
            if not current_block_mask.any():
                current_block += 1

        if log:
            remaining = (current_seq == mask_id).sum().item()
            print(f"Remaining masks: {remaining}")

    if log:
        print("=== Generation Complete ===")
        print(f"Final mask count: {(current_seq == mask_id).sum().item()}")
        print(f"Total decoded tokens: {len(records)}")

    # ===== final statistics =====
    decoded_tokens = len(records)
    tpf = decoded_tokens / forward_count if forward_count > 0 else 0.0
    avg_tokens_per_decoding_step = (
        decoded_tokens / decoding_steps_used if decoding_steps_used > 0 else 0.0
    )

    parallel_token_count = sum(
        1 for r in records if r.get("strategy") == "parallel"
    )

    single_token_count = sum(
        1 for r in records if r.get("strategy") == "single"
    )

    # 统计 step 层面的 parallel/single
    step_to_strategies = {}
    step_to_group_size = {}

    for r in records:
        step = r["global_step"]
        step_to_strategies.setdefault(step, set()).add(r.get("strategy"))

        if r.get("strategy") == "parallel":
            step_to_group_size[step] = r.get("parallel_group_size", 1)
        else:
            step_to_group_size.setdefault(step, 1)

    parallel_step_count = 0
    single_step_count = 0

    for step, strategies in step_to_strategies.items():
        if "parallel" in strategies:
            parallel_step_count += 1
        elif "single" in strategies:
            single_step_count += 1

    if len(step_to_group_size) > 0:
        avg_group_size = sum(step_to_group_size.values()) / len(step_to_group_size)
    else:
        avg_group_size = 0.0

    print("====== Decoding Statistics ======")
    print(f"Decoded tokens: {decoded_tokens}")
    print(f"Model forward calls: {forward_count}")
    print(f"Actual decoding steps with unmask: {decoding_steps_used}")
    print(f"TPF (tokens per forward): {tpf:.4f}")
    print(f"Avg tokens per decoding step: {avg_tokens_per_decoding_step:.4f}")
    print(f"Parallel decoded tokens: {parallel_token_count}")
    print(f"Single decoded tokens: {single_token_count}")
    print(f"Parallel decoding steps: {parallel_step_count}")
    print(f"Single decoding steps: {single_step_count}")
    print(f"Avg selected tokens per active decoding step: {avg_group_size:.4f}")
    print(f"All candidate token records: {len(all_token_records)}")

    if print_all_token_records:
        print(json.dumps({
            "selected_records": records,
            "all_token_records": all_token_records,
            "stats": {
                "decoded_tokens": decoded_tokens,
                "model_forward_calls": forward_count,
                "steps": decoding_steps_used,
                "tpf": tpf,
                "tokens_per_decoding_step": avg_tokens_per_decoding_step,
                "parallel_decoded_tokens": parallel_token_count,
                "single_decoded_tokens": single_token_count,
                "parallel_decoding_steps": parallel_step_count,
                "single_decoding_steps": single_step_count,
                "avg_selected_tokens_per_active_decoding_step": avg_group_size,
                "num_all_token_records": len(all_token_records),
                "confidence_threshold": float(confidence_threshold),
            }
        }))
    else:
        print(json.dumps(records))
        print(len(records))

    return current_seq

def generate_token_threshold_parallel(
    model,
    prompt,
    threshold_dict,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    cfg_scale=0.,
    remasking='low_confidence',
    mask_id=126336,
    log=False,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    max_threshold=0.9,
    min_threshold=0.05,
    default_threshold=0.9,
    min_parallel_tokens=1,
    max_parallel_tokens=100,
    **kwargs
):
    '''
    Token-wise threshold adaptive parallel decoding.

    Args:
        model: Mask predictor.
        prompt: Tensor of shape (1, L).
        threshold_dict: dict, {token_id: threshold}. Keys can be int or str.
        steps: Sampling steps.
        gen_length: Generated answer length.
        block_length: Block length.
        temperature: Sampling temperature.
        cfg_scale: Classifier-free guidance scale.
        remasking: 'low_confidence' or 'random'.
        mask_id: [MASK] token id.
        logits_eos_inf: Whether to set EOS logit to -inf before argmax.
        confidence_eos_eot_inf: Whether to suppress EOS/EOT for confidence.
        max_threshold: Upper bound for token-wise threshold.
        min_threshold: Lower bound for token-wise threshold.
        default_threshold: Threshold for token ids not in threshold_dict.
        min_parallel_tokens: Minimum high-confidence tokens to trigger parallel decoding.
        max_parallel_tokens: Maximum tokens decoded in one forward step.

    Returns:
        current_seq: Final decoded sequence.
    '''

    import json

    # Normalize threshold_dict keys to int.
    token_thresholds = {}
    for k, v in threshold_dict.items():
        try:
            token_thresholds[int(k)] = float(v)
        except Exception:
            continue

    def get_token_threshold(token_id):
        tau = token_thresholds.get(int(token_id), default_threshold)
        tau = max(min_threshold, min(max_threshold, float(tau)))
        return tau

    # 初始化序列
    x = torch.full(
        (1, prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long
    ).to(model.device)

    x[:, :prompt.shape[1]] = prompt.clone()

    current_seq = x.clone()
    current_block = 0
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    records = []

    # ===== statistics =====
    forward_count = 0
    decoding_steps_used = 0

    if log:
        print("=== Token-threshold Adaptive Parallel Decoding Start ===")
        print(f"Total blocks: {num_blocks}, Steps per block: {steps_per_block}")
        print(f"Default threshold: {default_threshold}")
        print(f"Min threshold: {min_threshold}, Max threshold: {max_threshold}")
        print(f"Loaded token thresholds: {len(token_thresholds)}")
        print(f"Initial mask count: {(x == mask_id).sum().item()}")

    for global_step in range(steps):
        # 如果已经没有 mask，提前停止
        if not (current_seq == mask_id).any():
            if log:
                print(f"No masks remaining, early stopping at step {global_step + 1}")
            break

        if log:
            print(f"=== Step {global_step + 1}/{steps} ===")

        # 计算 logits
        with torch.no_grad():
            if cfg_scale > 0.:
                unconditional_seq = current_seq.clone()
                unconditional_seq[prompt_index] = mask_id

                combined_seq = torch.cat([current_seq, unconditional_seq], dim=0)
                combined_logits = model(combined_seq).logits

                conditional_logits, unconditional_logits = torch.chunk(
                    combined_logits,
                    2,
                    dim=0
                )

                logits = unconditional_logits + (
                    cfg_scale + 1
                ) * (conditional_logits - unconditional_logits)
            else:
                logits = model(current_seq).logits

        forward_count += 1

        if logits_eos_inf:
            logits[:, :, 126081] = -torch.inf

        # 添加 Gumbel noise，得到每个 position 的 top-1 token
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

        if confidence_eos_eot_inf:
            logits_with_noise[:, :, 126081] = -torch.inf
            logits_with_noise[:, :, 126348] = -torch.inf

        x0 = torch.argmax(logits_with_noise, dim=-1)

        # 计算 confidence
        if remasking == 'low_confidence':
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(
                p,
                dim=-1,
                index=x0.unsqueeze(-1)
            ).squeeze(-1)
        elif remasking == 'random':
            x0_p = torch.rand(x0.shape, device=x0.device)
        else:
            raise NotImplementedError(remasking)

        # 当前 block 范围
        block_start = prompt.shape[1] + current_block * block_length
        block_end = prompt.shape[1] + (current_block + 1) * block_length

        # 获取 mask 和 confidence
        mask_index = (current_seq == mask_id)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        # 限制在当前 block 及之前
        confidence[:, prompt.shape[1] + (current_block + 1) * block_length:] = -np.inf

        # 当前 block 内的 mask positions
        block_mask_positions = (
            torch.where(mask_index[0, block_start:block_end])[0] + block_start
        )

        # 当前 block 没有 mask，移动到下一个 block
        if len(block_mask_positions) == 0:
            if current_block < num_blocks - 1:
                current_block += 1
                continue
            else:
                break

        block_mask_confidence = confidence[0, block_mask_positions]

        # 每个 position 的 top-1 token id
        block_mask_token_ids = x0[0, block_mask_positions]

        # 为每个 candidate position 取对应 token threshold
        threshold_values = []
        for token_id in block_mask_token_ids.detach().cpu().tolist():
            threshold_values.append(get_token_threshold(token_id))

        threshold_tensor = torch.tensor(
            threshold_values,
            dtype=block_mask_confidence.dtype,
            device=block_mask_confidence.device
        )

        # token-wise threshold 判断
        high_confidence_mask = block_mask_confidence >= threshold_tensor
        high_confidence_indices = torch.where(high_confidence_mask)[0]

        if len(high_confidence_indices) >= min_parallel_tokens:
            # 策略 1：并行解码多个满足 token-wise threshold 的 token
            num_to_unmask = min(len(high_confidence_indices), max_parallel_tokens)

            # 在满足阈值的 token 里，仍然选择 confidence 最高的前几个
            top_probs, top_indices = torch.topk(
                block_mask_confidence[high_confidence_indices],
                num_to_unmask
            )

            selected_indices = high_confidence_indices[top_indices]

            if log:
                print(
                    f"Parallel decoding {num_to_unmask} tokens "
                    f"from {len(high_confidence_indices)} candidates"
                )

            for idx in range(num_to_unmask):
                original_idx = selected_indices[idx].item()
                pos = block_mask_positions[original_idx].item()
                token = x0[0, pos].item()
                conf = block_mask_confidence[original_idx].item()
                tau = threshold_tensor[original_idx].item()

                current_seq[0, pos] = token

                records.append({
                    "step": global_step + 1,
                    "position": pos,
                    "confidence": conf,
                    "threshold": tau,
                    "token_id": token,
                    "strategy": "parallel",
                    "block": current_block,
                    "parallel_group_size": num_to_unmask
                })

            decoding_steps_used += 1

            # 检查当前 block 是否完成
            if current_block < num_blocks - 1:
                current_block_mask = (
                    current_seq[:, block_start:block_end] == mask_id
                )
                if not current_block_mask.any():
                    current_block += 1

        else:
            # 策略 2：没有任何 token 达到对应 threshold，fallback 到单个最高 confidence token
            top_prob, top_idx = torch.max(block_mask_confidence, dim=0)

            pos = block_mask_positions[top_idx].item()
            token = x0[0, pos].item()
            conf = top_prob.item()
            tau = threshold_tensor[top_idx].item()

            current_seq[0, pos] = token

            if log:
                print(
                    f"Single token fallback: position {pos}, "
                    f"confidence {conf:.4f}, threshold {tau:.4f}, token_id {token}"
                )

            records.append({
                "step": global_step + 1,
                "position": pos,
                "confidence": conf,
                "threshold": tau,
                "token_id": token,
                "strategy": "single",
                "block": current_block
            })

            decoding_steps_used += 1

            # 检查当前 block 是否完成
            if current_block < num_blocks - 1:
                current_block_mask = (
                    current_seq[:, block_start:block_end] == mask_id
                )
                if not current_block_mask.any():
                    current_block += 1

        if log:
            remaining = (current_seq == mask_id).sum().item()
            print(f"Remaining masks: {remaining}")

    if log:
        print("=== Generation Complete ===")
        print(f"Final mask count: {(current_seq == mask_id).sum().item()}")
        print(f"Total decoded tokens: {len(records)}")

    # ===== final statistics =====
    decoded_tokens = len(records)
    tpf = decoded_tokens / forward_count if forward_count > 0 else 0.0
    avg_tokens_per_decoding_step = (
        decoded_tokens / decoding_steps_used if decoding_steps_used > 0 else 0.0
    )

    parallel_token_count = sum(
        1 for r in records if r.get("strategy") == "parallel"
    )

    single_token_count = sum(
        1 for r in records if r.get("strategy") == "single"
    )

    # step 层面的 parallel/single
    step_to_strategies = {}
    step_to_group_size = {}

    for r in records:
        step = r["step"]
        step_to_strategies.setdefault(step, set()).add(r.get("strategy"))

        if r.get("strategy") == "parallel":
            step_to_group_size[step] = r.get("parallel_group_size", 1)
        else:
            step_to_group_size.setdefault(step, 1)

    parallel_step_count = 0
    single_step_count = 0

    for step, strategies in step_to_strategies.items():
        if "parallel" in strategies:
            parallel_step_count += 1
        elif "single" in strategies:
            single_step_count += 1

    if len(step_to_group_size) > 0:
        avg_group_size = sum(step_to_group_size.values()) / len(step_to_group_size)
    else:
        avg_group_size = 0.0

    avg_threshold = (
        sum(r["threshold"] for r in records) / len(records)
        if records else 0.0
    )

    print(json.dumps(records))
    print(len(records))

    print("====== Decoding Statistics ======")
    print(f"Decoded tokens: {decoded_tokens}")
    print(f"Model forward calls: {forward_count}")
    print(f"Actual decoding steps with unmask: {decoding_steps_used}")
    print(f"TPF (tokens per forward): {tpf:.4f}")
    print(f"Avg tokens per decoding step: {avg_tokens_per_decoding_step:.4f}")
    print(f"Parallel decoded tokens: {parallel_token_count}")
    print(f"Single decoded tokens: {single_token_count}")
    print(f"Parallel decoding steps: {parallel_step_count}")
    print(f"Single decoding steps: {single_step_count}")
    print(f"Avg selected tokens per active decoding step: {avg_group_size:.4f}")
    print(f"Avg applied threshold: {avg_threshold:.4f}")


    return current_seq


def generate_token_threshold_parallel_straggler_aware(
    model,
    prompt,
    threshold_dict,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    cfg_scale=0.,
    remasking='low_confidence',
    mask_id=126336,
    log=False,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    max_threshold=0.9,
    min_threshold=0.05,
    default_threshold=0.9,
    min_parallel_tokens=1,
    max_parallel_tokens=100,
    # ===== straggler-aware release =====
    enable_straggler_release=True,
    # release_remaining_8=0.20,
    # release_remaining_4=0.40,
    # release_remaining_2=0.60,
    release_remaining_8=0.10,
    release_remaining_4=0.20,
    release_remaining_2=0.30,
    protect_special_tokens=True,
    special_token_ids=(126081, 126348),  # eos / eot
    **kwargs
):
    """
    Token-wise threshold adaptive parallel decoding with straggler-aware late-stage release.

    Main idea:
        Use token-specific threshold as base threshold.
        When the current block has only a few remaining masked tokens,
        slightly lower the threshold to release near-threshold stragglers.

    Args:
        threshold_dict: dict, {token_id: threshold}. Keys can be int or str.
        enable_straggler_release: whether to apply late-stage release.
        release_remaining_8: subtract this value when remaining masks in current block <= 8.
        release_remaining_4: subtract this value when remaining masks in current block <= 4.
        release_remaining_2: subtract this value when remaining masks in current block <= 2.
        protect_special_tokens: if True, do not apply release to special_token_ids.
        special_token_ids: tokens protected from release, e.g. EOS/EOT.

    Returns:
        current_seq
    """

    import json

    # Normalize threshold_dict keys to int.
    token_thresholds = {}
    for k, v in threshold_dict.items():
        try:
            token_thresholds[int(k)] = float(v)
        except Exception:
            continue

    special_token_ids = set(int(x) for x in special_token_ids)

    def get_base_threshold(token_id):
        tau = token_thresholds.get(int(token_id), default_threshold)
        tau = max(min_threshold, min(max_threshold, float(tau)))
        return tau

    def apply_straggler_release(token_id, base_tau, remaining_masks_in_block):
        """
        Apply late-stage threshold release according to the number of remaining masks.
        """
        tau = float(base_tau)

        if not enable_straggler_release:
            return tau, 0.0

        if protect_special_tokens and int(token_id) in special_token_ids:
            return tau, 0.0

        release = 0.0

        if remaining_masks_in_block <= 2:
            release = release_remaining_2
        elif remaining_masks_in_block <= 4:
            release = release_remaining_4
        elif remaining_masks_in_block <= 8:
            release = release_remaining_8

        tau = tau - release
        tau = max(min_threshold, min(max_threshold, tau))

        return tau, release

    # 初始化序列
    x = torch.full(
        (1, prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long
    ).to(model.device)

    x[:, :prompt.shape[1]] = prompt.clone()

    current_seq = x.clone()
    current_block = 0
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    records = []

    # ===== statistics =====
    forward_count = 0
    decoding_steps_used = 0

    if log:
        print("=== Token-threshold + Straggler-aware Decoding Start ===")
        print(f"Total blocks: {num_blocks}, Steps per block: {steps_per_block}")
        print(f"Default threshold: {default_threshold}")
        print(f"Min threshold: {min_threshold}, Max threshold: {max_threshold}")
        print(f"Loaded token thresholds: {len(token_thresholds)}")
        print(f"Straggler release: {enable_straggler_release}")
        print(f"Release <=8: {release_remaining_8}, <=4: {release_remaining_4}, <=2: {release_remaining_2}")
        print(f"Protect special tokens: {protect_special_tokens}, special_token_ids={special_token_ids}")
        print(f"Initial mask count: {(x == mask_id).sum().item()}")

    for global_step in range(steps):
        if not (current_seq == mask_id).any():
            if log:
                print(f"No masks remaining, early stopping at step {global_step + 1}")
            break

        if log:
            print(f"=== Step {global_step + 1}/{steps} ===")

        # 计算 logits
        with torch.no_grad():
            if cfg_scale > 0.:
                unconditional_seq = current_seq.clone()
                unconditional_seq[prompt_index] = mask_id

                combined_seq = torch.cat([current_seq, unconditional_seq], dim=0)
                combined_logits = model(combined_seq).logits

                conditional_logits, unconditional_logits = torch.chunk(
                    combined_logits,
                    2,
                    dim=0
                )

                logits = unconditional_logits + (
                    cfg_scale + 1
                ) * (conditional_logits - unconditional_logits)
            else:
                logits = model(current_seq).logits

        forward_count += 1

        if logits_eos_inf:
            logits[:, :, 126081] = -torch.inf

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

        if confidence_eos_eot_inf:
            logits_with_noise[:, :, 126081] = -torch.inf
            logits_with_noise[:, :, 126348] = -torch.inf

        x0 = torch.argmax(logits_with_noise, dim=-1)

        # 计算 confidence
        if remasking == 'low_confidence':
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(
                p,
                dim=-1,
                index=x0.unsqueeze(-1)
            ).squeeze(-1)
        elif remasking == 'random':
            x0_p = torch.rand(x0.shape, device=x0.device)
        else:
            raise NotImplementedError(remasking)

        # 当前 block 范围
        block_start = prompt.shape[1] + current_block * block_length
        block_end = prompt.shape[1] + (current_block + 1) * block_length

        # 获取 mask 和 confidence
        mask_index = (current_seq == mask_id)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        # 限制在当前 block 及之前
        confidence[:, prompt.shape[1] + (current_block + 1) * block_length:] = -np.inf

        # 当前 block 内的 mask positions
        block_mask_positions = (
            torch.where(mask_index[0, block_start:block_end])[0] + block_start
        )

        # 当前 block 没有 mask，移动到下一个 block
        if len(block_mask_positions) == 0:
            if current_block < num_blocks - 1:
                current_block += 1
                continue
            else:
                break

        remaining_masks_in_block = len(block_mask_positions)

        block_mask_confidence = confidence[0, block_mask_positions]
        block_mask_token_ids = x0[0, block_mask_positions]

        # 为每个 candidate position 取 token threshold，并做 straggler-aware release
        base_threshold_values = []
        threshold_values = []
        release_values = []

        for token_id in block_mask_token_ids.detach().cpu().tolist():
            base_tau = get_base_threshold(token_id)
            tau, release = apply_straggler_release(
                token_id=token_id,
                base_tau=base_tau,
                remaining_masks_in_block=remaining_masks_in_block,
            )

            base_threshold_values.append(base_tau)
            threshold_values.append(tau)
            release_values.append(release)

        threshold_tensor = torch.tensor(
            threshold_values,
            dtype=block_mask_confidence.dtype,
            device=block_mask_confidence.device
        )

        base_threshold_tensor = torch.tensor(
            base_threshold_values,
            dtype=block_mask_confidence.dtype,
            device=block_mask_confidence.device
        )

        release_tensor = torch.tensor(
            release_values,
            dtype=block_mask_confidence.dtype,
            device=block_mask_confidence.device
        )

        # token-wise threshold 判断
        high_confidence_mask = block_mask_confidence >= threshold_tensor
        high_confidence_indices = torch.where(high_confidence_mask)[0]

        if len(high_confidence_indices) >= min_parallel_tokens:
            # 策略 1：并行解码多个满足 token-wise threshold 的 token
            num_to_unmask = min(len(high_confidence_indices), max_parallel_tokens)

            top_probs, top_indices = torch.topk(
                block_mask_confidence[high_confidence_indices],
                num_to_unmask
            )

            selected_indices = high_confidence_indices[top_indices]

            if log:
                print(
                    f"Parallel decoding {num_to_unmask} tokens "
                    f"from {len(high_confidence_indices)} candidates; "
                    f"remaining_masks_in_block={remaining_masks_in_block}"
                )

            for idx in range(num_to_unmask):
                original_idx = selected_indices[idx].item()
                pos = block_mask_positions[original_idx].item()
                token = x0[0, pos].item()
                conf = block_mask_confidence[original_idx].item()
                tau = threshold_tensor[original_idx].item()
                base_tau = base_threshold_tensor[original_idx].item()
                release = release_tensor[original_idx].item()

                current_seq[0, pos] = token

                records.append({
                    "step": global_step + 1,
                    "position": pos,
                    "confidence": conf,
                    "threshold": tau,
                    "base_threshold": base_tau,
                    "release": release,
                    "remaining_masks_in_block": remaining_masks_in_block,
                    "token_id": token,
                    "strategy": "parallel",
                    "block": current_block,
                    "parallel_group_size": num_to_unmask,
                })

            decoding_steps_used += 1

            # 检查当前 block 是否完成
            if current_block < num_blocks - 1:
                current_block_mask = (
                    current_seq[:, block_start:block_end] == mask_id
                )
                if not current_block_mask.any():
                    current_block += 1

        else:
            # 策略 2：没有任何 token 达到对应 threshold，fallback 到单个最高 confidence token
            top_prob, top_idx = torch.max(block_mask_confidence, dim=0)

            pos = block_mask_positions[top_idx].item()
            token = x0[0, pos].item()
            conf = top_prob.item()
            tau = threshold_tensor[top_idx].item()
            base_tau = base_threshold_tensor[top_idx].item()
            release = release_tensor[top_idx].item()

            current_seq[0, pos] = token

            if log:
                print(
                    f"Single token fallback: position {pos}, "
                    f"confidence {conf:.4f}, threshold {tau:.4f}, "
                    f"base_threshold {base_tau:.4f}, release {release:.4f}, "
                    f"remaining_masks_in_block={remaining_masks_in_block}, "
                    f"token_id {token}"
                )

            records.append({
                "step": global_step + 1,
                "position": pos,
                "confidence": conf,
                "threshold": tau,
                "base_threshold": base_tau,
                "release": release,
                "remaining_masks_in_block": remaining_masks_in_block,
                "token_id": token,
                "strategy": "single",
                "block": current_block,
            })

            decoding_steps_used += 1

            # 检查当前 block 是否完成
            if current_block < num_blocks - 1:
                current_block_mask = (
                    current_seq[:, block_start:block_end] == mask_id
                )
                if not current_block_mask.any():
                    current_block += 1

        if log:
            remaining = (current_seq == mask_id).sum().item()
            print(f"Remaining masks: {remaining}")

    if log:
        print("=== Generation Complete ===")
        print(f"Final mask count: {(current_seq == mask_id).sum().item()}")
        print(f"Total decoded tokens: {len(records)}")

    # ===== final statistics =====
    decoded_tokens = len(records)
    tpf = decoded_tokens / forward_count if forward_count > 0 else 0.0
    avg_tokens_per_decoding_step = (
        decoded_tokens / decoding_steps_used if decoding_steps_used > 0 else 0.0
    )

    parallel_token_count = sum(
        1 for r in records if r.get("strategy") == "parallel"
    )

    single_token_count = sum(
        1 for r in records if r.get("strategy") == "single"
    )

    step_to_strategies = {}
    step_to_group_size = {}

    for r in records:
        step = r["step"]
        step_to_strategies.setdefault(step, set()).add(r.get("strategy"))

        if r.get("strategy") == "parallel":
            step_to_group_size[step] = r.get("parallel_group_size", 1)
        else:
            step_to_group_size.setdefault(step, 1)

    parallel_step_count = 0
    single_step_count = 0

    for step, strategies in step_to_strategies.items():
        if "parallel" in strategies:
            parallel_step_count += 1
        elif "single" in strategies:
            single_step_count += 1

    avg_group_size = (
        sum(step_to_group_size.values()) / len(step_to_group_size)
        if len(step_to_group_size) > 0 else 0.0
    )

    avg_threshold = (
        sum(r["threshold"] for r in records) / len(records)
        if records else 0.0
    )

    avg_base_threshold = (
        sum(r["base_threshold"] for r in records) / len(records)
        if records else 0.0
    )

    avg_release = (
        sum(r["release"] for r in records) / len(records)
        if records else 0.0
    )

    released_token_count = sum(1 for r in records if r.get("release", 0.0) > 0.0)

    print(json.dumps(records))
    print(len(records))

    print("====== Decoding Statistics ======")
    print(f"Decoded tokens: {decoded_tokens}")
    print(f"Model forward calls: {forward_count}")
    print(f"Actual decoding steps with unmask: {decoding_steps_used}")
    print(f"TPF (tokens per forward): {tpf:.4f}")
    print(f"Avg tokens per decoding step: {avg_tokens_per_decoding_step:.4f}")
    print(f"Parallel decoded tokens: {parallel_token_count}")
    print(f"Single decoded tokens: {single_token_count}")
    print(f"Parallel decoding steps: {parallel_step_count}")
    print(f"Single decoding steps: {single_step_count}")
    print(f"Avg selected tokens per active decoding step: {avg_group_size:.4f}")
    print(f"Avg applied threshold: {avg_threshold:.4f}")
    print(f"Avg base threshold: {avg_base_threshold:.4f}")
    print(f"Avg release: {avg_release:.4f}")
    print(f"Released selected tokens: {released_token_count}")

    return current_seq



def generate_soar_token_threshold(
    model,
    prompt,
    threshold_dict,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    cfg_scale=0.,
    remasking='low_confidence',
    mask_id=126336,
    log=False,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
):
    """
    SOAR decoding with token-level calibrated thresholds.

    Core logic:
        - Keep SOAR beam list.
        - Replace fixed global confidence threshold with token-level threshold.
        - If token_id exists in threshold_dict, use calibrated threshold.
        - Otherwise use default_threshold = 0.90.
        - If at least one token in current block satisfies its threshold,
          use SOAR parallel decoding branch.
        - If no token satisfies its threshold, use SOAR beam-search fallback.
        - Beam candidates are ranked by cumulative_log_prob.
        - Beam-search fallback keeps top-2 candidates.

    Args:
        model: Mask predictor.
        prompt: Tensor of shape (1, L).
        threshold_dict: dict, {token_id: threshold}. Keys can be int or str.
        steps: Sampling steps.
        gen_length: Generated answer length.
        block_length: Block length.
        temperature: Sampling temperature.
        cfg_scale: Classifier-free guidance scale.
        remasking: 'low_confidence' or 'random'.
        mask_id: [MASK] token id.
        log: Whether to print detailed logs.
        logits_eos_inf: Whether to set EOS logit to -inf.
        confidence_eos_eot_inf: Whether to suppress EOS/EOT for confidence.

    Returns:
        best_sequence: Final decoded sequence from the best beam.
    """

    import json

    print("======SOAR + token-level calibrated threshold, temperature: {:.1f}====".format(temperature))

    # =========================
    # Internal SOAR config
    # =========================
    default_threshold = 0.90
    min_parallel_tokens = 1
    max_parallel_tokens = 100
    max_beam_size = 2

    # =========================
    # Normalize threshold dict
    # =========================
    token_thresholds = {}
    for k, v in threshold_dict.items():
        try:
            token_thresholds[int(k)] = float(v)
        except Exception:
            continue

    def get_token_threshold(token_id):
        token_id = int(token_id)
        return float(token_thresholds.get(token_id, default_threshold))

    # =========================
    # Decoding statistics
    # =========================
    forward_count = 0
    actual_global_steps = 0

    # =========================
    # Initialize beam
    # Each beam item:
    # (sequence, cumulative_log_prob, current_block, records)
    # =========================
    x = torch.full(
        (1, prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long
    ).to(model.device)

    x[:, :prompt.shape[1]] = prompt.clone()

    beam = [(x.clone(), 0.0, 0, [])]
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    if log:
        print("=== SOAR + Token Threshold Generation Start ===")
        print(f"Total blocks: {num_blocks}, Steps per block: {steps_per_block}")
        print(f"Max beam size: {max_beam_size}")
        print(f"Default threshold: {default_threshold}")
        print(f"Loaded token thresholds: {len(token_thresholds)}")
        print(f"Initial mask count: {(x == mask_id).sum().item()}")

    for global_step in range(steps):
        if log:
            print(f"=== Global Step {global_step + 1}/{steps} ===")

        # Check whether any beam still has masks
        has_remaining_masks = False
        for seq, _, _, _ in beam:
            if (seq == mask_id).any():
                has_remaining_masks = True
                break

        if not has_remaining_masks:
            if log:
                print(f"No masks remaining in any beam, early stopping at step {global_step + 1}")
            break

        # Collect all beam sequences into a batch
        beam_sequences = [seq for seq, _, _, _ in beam]
        batch_sequences = torch.cat(beam_sequences, dim=0)

        if log:
            print(f"Processing batch of {len(beam)} beam sequences")
            total_masks = sum((seq == mask_id).sum().item() for seq in beam_sequences)
            print(f"Total remaining masks across beams: {total_masks}")

        # =========================
        # Batched forward pass
        # One model(...) call counts as one forward call.
        # =========================
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
                forward_count += 1

                conditional_logits, unconditional_logits = torch.chunk(batch_logits, 2, dim=0)
                batch_logits = unconditional_logits + (cfg_scale + 1) * (
                    conditional_logits - unconditional_logits
                )
            else:
                batch_logits = model(batch_sequences).logits
                forward_count += 1

        actual_global_steps += 1

        if logits_eos_inf:
            batch_logits[:, :, 126081] = -torch.inf

        # Add Gumbel noise and get predictions
        logits_with_noise = add_gumbel_noise(batch_logits, temperature=temperature)

        if confidence_eos_eot_inf:
            logits_with_noise[:, :, 126081] = -torch.inf
            logits_with_noise[:, :, 126348] = -torch.inf

        batch_x0 = torch.argmax(logits_with_noise, dim=-1)

        # Compute confidence
        if remasking == 'low_confidence':
            p = F.softmax(batch_logits, dim=-1)
            batch_x0_p = torch.gather(
                p,
                dim=-1,
                index=batch_x0.unsqueeze(-1)
            ).squeeze(-1)
        elif remasking == 'random':
            batch_x0_p = torch.rand(batch_x0.shape, device=batch_x0.device)
        else:
            raise NotImplementedError(remasking)

        new_beam_candidates = []
        has_multi_unmask_candidate = False

        # =========================
        # Process each beam
        # =========================
        for beam_idx, (seq, cumulative_log_prob, current_block, records) in enumerate(beam):
            x0 = batch_x0[beam_idx:beam_idx + 1]
            x0_p = batch_x0_p[beam_idx:beam_idx + 1]

            if log:
                print(f"--- Processing Beam {beam_idx + 1}/{len(beam)} ---")
                print(f"Current cumulative log prob: {cumulative_log_prob:.4f}")
                print(f"Current block progress: {current_block}/{num_blocks}")

            if not (seq == mask_id).any():
                new_beam_candidates.append((seq, cumulative_log_prob, current_block, records))
                if log:
                    print("    Sequence already complete")
                continue

            # Current block range
            block_start = prompt.shape[1] + current_block * block_length
            block_end = prompt.shape[1] + (current_block + 1) * block_length

            mask_index = (seq == mask_id)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # Only allow decoding up to the current block
            confidence[:, prompt.shape[1] + (current_block + 1) * block_length:] = -np.inf

            block_mask_positions = (
                torch.where(mask_index[0, block_start:block_end])[0] + block_start
            )

            # If current block has no masks, move to next block
            if len(block_mask_positions) == 0:
                new_current_block = min(current_block + 1, num_blocks - 1)
                new_beam_candidates.append(
                    (seq, cumulative_log_prob, new_current_block, records)
                )

                if log:
                    print(f"    No masks in block {current_block}, moving to block {new_current_block}")

                continue

            block_mask_confidence = confidence[0, block_mask_positions]
            block_mask_token_ids = x0[0, block_mask_positions]

            # =========================
            # Token-level calibrated thresholds
            # =========================
            threshold_values = []
            for token_id in block_mask_token_ids.detach().cpu().tolist():
                threshold_values.append(get_token_threshold(token_id))

            threshold_tensor = torch.tensor(
                threshold_values,
                dtype=block_mask_confidence.dtype,
                device=block_mask_confidence.device
            )

            if log:
                print(f"Block {current_block} mask positions: {block_mask_positions.detach().cpu().tolist()}")
                print(f"Block {current_block} confidences: {block_mask_confidence.detach().cpu().float().tolist()}")
                print(f"Block {current_block} thresholds: {threshold_tensor.detach().cpu().float().tolist()}")

            # =========================
            # Strategy 1:
            # Parallel decode tokens whose confidence >= token threshold
            # =========================
            high_confidence_mask = block_mask_confidence >= threshold_tensor
            high_confidence_indices = torch.where(high_confidence_mask)[0]

            if len(high_confidence_indices) >= min_parallel_tokens:
                if log:
                    print(
                        f"Strategy 1: Parallel decoding "
                        f"{len(high_confidence_indices)} calibrated high-confidence tokens"
                    )

                num_to_unmask = min(len(high_confidence_indices), max_parallel_tokens)

                top_probs, top_indices = torch.topk(
                    block_mask_confidence[high_confidence_indices],
                    num_to_unmask
                )

                selected_indices = high_confidence_indices[top_indices]

                new_seq = seq.clone()
                new_log_prob = cumulative_log_prob
                new_records = records.copy()

                for idx in range(num_to_unmask):
                    original_idx = selected_indices[idx].item()
                    pos = block_mask_positions[original_idx].item()
                    token = x0[0, pos].item()
                    prob = block_mask_confidence[original_idx].item()
                    tau = threshold_tensor[original_idx].item()

                    new_seq[0, pos] = token
                    new_log_prob += prob

                    new_records.append({
                        "step": global_step + 1,
                        "position": int(pos),
                        "confidence": float(prob),
                        "threshold": float(tau),
                        "token_id": int(token),
                        "strategy": "parallel",
                        "block": int(current_block),
                        "parallel_group_size": int(num_to_unmask),
                        "beam_idx": int(beam_idx),
                        "beam_size": int(len(beam)),
                    })

                new_current_block = current_block

                if new_current_block < num_blocks - 1:
                    current_block_mask = (new_seq[:, block_start:block_end] == mask_id)
                    if not current_block_mask.any():
                        new_current_block += 1
                        if log:
                            print(
                                f"    Block {current_block} completed after parallel decoding, "
                                f"moving to block {new_current_block}"
                            )

                new_beam_candidates.append(
                    (new_seq, new_log_prob, new_current_block, new_records)
                )

                has_multi_unmask_candidate = True

                if log:
                    print(f"    Parallel unmasked {num_to_unmask} tokens")
                    print(f"    New cumulative log prob: {new_log_prob:.4f}")

            # =========================
            # Strategy 2:
            # No token satisfies calibrated threshold.
            # Beam search over top-2 positions.
            # =========================
            else:
                k = min(max_beam_size, len(block_mask_confidence))

                if k == 0:
                    new_current_block = min(current_block + 1, num_blocks - 1)
                    new_beam_candidates.append(
                        (seq, cumulative_log_prob, new_current_block, records)
                    )

                    if log:
                        print(f"    No masks in current block, moving to block {new_current_block}")

                    continue

                top_probs, top_indices = torch.topk(block_mask_confidence, k)
                top_positions = block_mask_positions[top_indices]
                top_tokens = x0[0, top_positions]

                if log:
                    print(f"Strategy 2: Beam search fallback with k={k}")

                for idx in range(k):
                    new_seq = seq.clone()

                    pos = top_positions[idx].item()
                    token = top_tokens[idx].item()
                    prob = top_probs[idx].item()
                    tau = threshold_tensor[top_indices[idx]].item()

                    new_seq[0, pos] = token
                    new_log_prob = cumulative_log_prob + prob

                    new_current_block = current_block
                    if new_current_block < num_blocks - 1:
                        current_block_mask = (new_seq[:, block_start:block_end] == mask_id)
                        if not current_block_mask.any():
                            new_current_block += 1

                    new_records = records.copy()
                    new_records.append({
                        "step": global_step + 1,
                        "position": int(pos),
                        "confidence": float(prob),
                        "threshold": float(tau),
                        "token_id": int(token),
                        "strategy": "beam",
                        "block": int(current_block),
                        "parallel_group_size": 1,
                        "beam_idx": int(beam_idx),
                        "beam_size": int(len(beam)),
                    })

                    new_beam_candidates.append(
                        (new_seq, new_log_prob, new_current_block, new_records)
                    )

                    if log:
                        print(
                            f"    Beam candidate {idx + 1}/{k}: "
                            f"pos={pos}, token={token}, conf={prob:.4f}, "
                            f"tau={tau:.4f}, score={new_log_prob:.4f}"
                        )

        # If no new candidates are generated, stop
        if not new_beam_candidates:
            if log:
                print("No new beam candidates generated, early stopping")
            break

        if log:
            print(f"Total candidates before selection: {len(new_beam_candidates)}")

        # Sort by cumulative log prob
        new_beam_candidates.sort(key=lambda item: item[1], reverse=True)

        # Deduplicate candidates by sequence
        uniq_new_beam_candidates = []
        seen = set()

        for tensor, log_prob, block_progress, records in new_beam_candidates:
            tensor_tuple = tuple(tensor.flatten().detach().cpu().numpy().tolist())

            if tensor_tuple not in seen:
                seen.add(tensor_tuple)
                uniq_new_beam_candidates.append(
                    (tensor, log_prob, block_progress, records)
                )

        if log:
            print(f"Unique candidates after deduplication: {len(uniq_new_beam_candidates)}")

        # =========================
        # Dynamic beam size adjustment
        # Keep original SOAR behavior:
        # If parallel decoding happens, reduce beam to 1.
        # Otherwise keep top-2 candidates.
        # =========================
        if has_multi_unmask_candidate and uniq_new_beam_candidates:
            best_candidate = uniq_new_beam_candidates[0]
            best_seq, best_log_prob, best_block, best_records = best_candidate

            original_mask_count = (beam[0][0] == mask_id).sum().item()
            current_mask_count = (best_seq == mask_id).sum().item()
            masks_unmasked = original_mask_count - current_mask_count

            if masks_unmasked >= min_parallel_tokens:
                beam = [best_candidate]

                if log:
                    print(
                        f"Dynamic adjustment: Parallel unmasked {masks_unmasked} tokens, "
                        f"beam size reduced to 1"
                    )
            else:
                beam_size = min(max_beam_size, len(uniq_new_beam_candidates))
                beam = uniq_new_beam_candidates[:beam_size]

                if log:
                    print(f"Dynamic adjustment: Beam size set to {beam_size}")
        else:
            beam_size = min(max_beam_size, len(uniq_new_beam_candidates))
            beam = uniq_new_beam_candidates[:beam_size]

            if log:
                print(f"Dynamic adjustment: No parallel decoding, beam size set to {beam_size}")

        best_seq, best_score, best_block, best_records = beam[0]

        if log:
            print(f"Current beam size: {len(beam)}")
            print(f"Best sequence score: {best_score:.4f}")
            print(f"Best beam current block: {best_block}")
            print(f"Remaining mask count: {(best_seq == mask_id).sum().item()}")

    # =========================
    # Select final best sequence
    # =========================
    if beam:
        best_sequence, best_score, _, best_records = beam[0]

        if log:
            print("=== SOAR + Token Threshold Generation Complete ===")
            print(f"Final sequence score: {best_score:.4f}")
            print(f"Final mask count: {(best_sequence == mask_id).sum().item()}")
            print(f"Total decoding records: {len(best_records)}")

            if best_records:
                steps_used = max(r["step"] for r in best_records)
                avg_confidence = sum(r["confidence"] for r in best_records) / len(best_records)
                avg_threshold = sum(r["threshold"] for r in best_records) / len(best_records)
                print(f"Steps used: {steps_used}")
                print(f"Average confidence: {avg_confidence:.4f}")
                print(f"Average threshold: {avg_threshold:.4f}")

            if not (best_sequence == mask_id).any():
                print("✓ All masks have been filled!")
            else:
                print(f"⚠ Still has {(best_sequence == mask_id).sum().item()} masks remaining")
    else:
        best_sequence = x
        best_records = []

        if log:
            print("=== SOAR + Token Threshold Generation Complete (No valid sequences) ===")

    # =========================
    # Decoding statistics for top-1 beam
    # =========================
    decoded_tokens = len(best_records)

    if best_records:
        decoding_steps_used = len(set(r["step"] for r in best_records))
    else:
        decoding_steps_used = 0

    tpf = decoded_tokens / forward_count if forward_count > 0 else 0.0

    avg_tokens_per_decoding_step = (
        decoded_tokens / decoding_steps_used if decoding_steps_used > 0 else 0.0
    )

    parallel_token_count = sum(
        1 for r in best_records if r.get("strategy") == "parallel"
    )

    beam_token_count = sum(
        1 for r in best_records if r.get("strategy") == "beam"
    )

    step_to_strategies = {}
    step_to_group_size = {}

    for r in best_records:
        step = r["step"]
        step_to_strategies.setdefault(step, set()).add(r.get("strategy"))

        if r.get("strategy") == "parallel":
            step_to_group_size[step] = r.get("parallel_group_size", 1)
        else:
            step_to_group_size.setdefault(step, 1)

    parallel_step_count = 0
    beam_step_count = 0

    for _, strategies in step_to_strategies.items():
        if "parallel" in strategies:
            parallel_step_count += 1
        elif "beam" in strategies:
            beam_step_count += 1

    avg_group_size = (
        sum(step_to_group_size.values()) / len(step_to_group_size)
        if len(step_to_group_size) > 0 else 0.0
    )

    avg_threshold = (
        sum(r["threshold"] for r in best_records) / len(best_records)
        if best_records else 0.0
    )

    avg_confidence = (
        sum(r["confidence"] for r in best_records) / len(best_records)
        if best_records else 0.0
    )

    decoding_stats = {
        "decoded_tokens": decoded_tokens,
        "model_forward_calls": forward_count,
        "actual_decoding_steps_with_unmask": decoding_steps_used,
        "tpf": tpf,
        "avg_tokens_per_decoding_step": avg_tokens_per_decoding_step,
        "actual_global_steps": actual_global_steps,
        "parallel_decoded_tokens": parallel_token_count,
        "beam_decoded_tokens": beam_token_count,
        "parallel_decoding_steps": parallel_step_count,
        "beam_decoding_steps": beam_step_count,
        "avg_selected_tokens_per_active_decoding_step": avg_group_size,
        "avg_applied_threshold": avg_threshold,
        "avg_selected_confidence": avg_confidence,
        "default_threshold": default_threshold,
        "loaded_token_thresholds": len(token_thresholds),
        "max_beam_size": max_beam_size,
    }

    # Keep output compatible with your existing log parser:
    # records JSON first, then len(records), then human-readable stats.
    print(json.dumps(best_records))
    print(len(best_records))

    print("====== Decoding Statistics ======")
    print(f"Decoded tokens: {decoded_tokens}")
    print(f"Model forward calls: {forward_count}")
    print(f"Actual decoding steps with unmask: {decoding_steps_used}")
    print(f"TPF (tokens per forward): {tpf:.4f}")
    print(f"Avg tokens per decoding step: {avg_tokens_per_decoding_step:.4f}")
    print(f"Parallel decoded tokens: {parallel_token_count}")
    print(f"Beam decoded tokens: {beam_token_count}")
    print(f"Parallel decoding steps: {parallel_step_count}")
    print(f"Beam decoding steps: {beam_step_count}")
    print(f"Avg selected tokens per active decoding step: {avg_group_size:.4f}")
    print(f"Avg applied threshold: {avg_threshold:.4f}")
    print(f"Avg selected confidence: {avg_confidence:.4f}")

    # Extra machine-readable stats
    print(json.dumps(decoding_stats))

    return best_sequence


def main():
    import json
    device = 'cuda'
    
    model = AutoModel.from_pretrained("/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct", trust_remote_code=True)
    
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    m = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    
    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    # out = generate_adaptive_parallel(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    with open("token_threshold_stats/token_threshold_p25.json", "r") as f:
        raw = json.load(f)

    threshold_dict = {int(k): float(v) for k, v in raw.items()}

    out = generate_token_threshold_parallel(
        model,
        input_ids,
        threshold_dict=threshold_dict,
        steps=128,
        gen_length=128,
        block_length=32,
    )
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()