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
    with open("token_threshold_stats/token_threshold_p50.json", "r") as f:
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