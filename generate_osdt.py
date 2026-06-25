import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    """
    Precomputes the number of tokens that need to be transitioned at each step
    for a standard, non-dynamic generation.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

def _generate_for_data_collection(model, prompt, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336):
    """
    Generates text without dynamic thresholds to collect baseline confidence data.
    This is intended for the first problem in a batch.
    """
    B, L = prompt.shape
    block_step_confidences = []
    x = torch.full((B, L + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :L] = prompt.clone()
    prompt_index = (x != mask_id)

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_step_confidences.append([])
        block_mask_index = (x[:, L + num_block * block_length: L + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            logits = model(x).logits
            nfe += 1
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            x0_p[:, L + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            
            step_confidences = confidence[transfer_index].to(torch.float32).cpu().numpy().tolist()
            block_step_confidences[num_block].append(step_confidences)
            x[transfer_index] = x0[transfer_index]

    return x, nfe, block_step_confidences

def _get_transfer_index_factor_based(confidence, factor_value, mask_index):
    """
    Determines which tokens to unmask based on the factor-based strategy.
    Finds the largest n such that (n+1)(1 - c^(n)) < f.
    """
    transfer_index = torch.zeros_like(confidence, dtype=torch.bool, device=confidence.device)
    
    # Iterate over each item in the batch
    for j in range(confidence.shape[0]):
        current_mask = mask_index[j]
        if not current_mask.any():
            continue

        # Get confidences of only the masked tokens
        masked_confidences = confidence[j][current_mask]
        
        # Sort confidences in descending order
        sorted_conf, _ = torch.sort(masked_confidences, descending=True)
        
        num_masked = len(sorted_conf)
        
        # Find the largest n
        n = 0
        for k in range(1, num_masked + 1):
            # c_k is the k-th highest confidence (using 0-based index k-1)
            c_k = sorted_conf[k-1]
            if (k + 1) * (1 - c_k) < factor_value:
                n = k
            else:
                # The condition is not met, so we stop
                break
        
        # Always unmask at least the most confident token if n is 0 but there are masked tokens
        if n == 0 and num_masked > 0:
            n = 1
            
        if n > 0:
            # Find the indices of the top-n tokens within the masked part
            _, top_indices_in_masked = torch.topk(masked_confidences, k=n)
            
            # Convert these local indices back to global indices
            original_indices = torch.where(current_mask)[0]
            global_indices = original_indices[top_indices_in_masked]
            
            transfer_index[j, global_indices] = True
            
    return transfer_index


def _generate_block_dynamic_internal(
    model, prompt, gen_length, block_length, temperature, remasking, mask_id,
    thresholds, tokenizer, attention_mask, threshold_cap, epsilon_ratio,
    decoding_strategy, decoding_factor
):
    """
    Internal function for block-wise dynamic thresholding.
    """
    B, L = prompt.shape
    x = torch.full((B, L + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :L] = prompt.clone()
    num_blocks = gen_length // block_length

    batch_indices = torch.arange(B, device=x.device)

    # =========================
    # Statistics
    # =========================
    forward_count = 0
    decoded_tokens = 0
    decoding_steps_used = 0

    for num_block in range(num_blocks):
        current_block_start = L + num_block * block_length
        current_block_end = current_block_start + block_length

        if decoding_strategy == 'threshold':
            current_threshold_with_cap = min(
                thresholds[min(num_block, len(thresholds) - 1)],
                threshold_cap
            )
            epsilon = current_threshold_with_cap * epsilon_ratio
            effective_threshold = current_threshold_with_cap - epsilon

        is_block_finished = torch.zeros(B, dtype=torch.bool, device=x.device)

        while not is_block_finished.all():
            active_indices = batch_indices[~is_block_finished]

            # Run model only on active sequences
            active_x = x[active_indices]
            active_attention_mask = (active_x != tokenizer.pad_token_id)

            all_logits = model(active_x, attention_mask=active_attention_mask).logits
            forward_count += 1

            # Perform calculations on the dense, active-only logits for numerical stability
            all_logits_with_noise = add_gumbel_noise(all_logits, temperature=temperature)
            all_x0 = torch.argmax(all_logits_with_noise, dim=-1)
            all_p = F.softmax(all_logits, dim=-1)

            # Create full-sized tensors to scatter results into
            x0 = torch.full_like(x, tokenizer.pad_token_id)
            p = torch.zeros(
                (B, x.size(1), model.config.vocab_size),
                dtype=all_p.dtype,
                device=x.device
            )

            # Scatter the results back
            x0[active_indices] = all_x0
            p[active_indices] = all_p

            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)),
                -1
            )
            x0_p[:, L + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where((x == mask_id), x0, x)
            confidence = torch.where((x == mask_id), x0_p, -np.inf)

            if decoding_strategy == 'threshold':
                transfer_index = confidence > effective_threshold
            else:  # factor
                transfer_index = _get_transfer_index_factor_based(
                    confidence,
                    decoding_factor,
                    (x == mask_id)
                )

            # Fallback for sequences that are not finished but did not transfer any tokens
            stuck_mask = (
                ~is_block_finished
                & (x[:, current_block_start:current_block_end] == mask_id).any(dim=1)
                & ~transfer_index.any(dim=1)
            )

            if stuck_mask.any():
                stuck_indices = batch_indices[stuck_mask]

                # Get confidences for stuck sequences in the current block
                block_confidence_stuck = confidence[
                    stuck_indices,
                    current_block_start:current_block_end
                ]

                # Find most confident token for each stuck sequence
                most_confident_local_idx_stuck = torch.argmax(block_confidence_stuck, dim=1)
                most_confident_global_idx_stuck = (
                    most_confident_local_idx_stuck + current_block_start
                )

                # Force unmask for these specific tokens in the specific stuck sequences
                transfer_index[stuck_indices, most_confident_global_idx_stuck] = True

            # =========================
            # Statistics before commit
            # =========================
            num_selected = int(transfer_index.sum().item())

            if num_selected > 0:
                decoded_tokens += num_selected
                decoding_steps_used += 1

            x[transfer_index] = x0[transfer_index]

            is_block_now_full = torch.all(
                x[:, current_block_start:current_block_end] != mask_id,
                dim=1
            )
            is_block_finished = torch.logical_or(is_block_finished, is_block_now_full)

    # =========================
    # Decoding statistics
    # =========================
    nfe = forward_count
    tpf = decoded_tokens / forward_count if forward_count > 0 else 0.0
    avg_tokens_per_decoding_step = (
        decoded_tokens / decoding_steps_used if decoding_steps_used > 0 else 0.0
    )

    print("====== Decoding Statistics ======")
    print(f"Decoded tokens: {decoded_tokens}")
    print(f"Model forward calls: {forward_count}")
    print(f"Actual decoding steps with unmask: {decoding_steps_used}")
    print(f"TPF (tokens per forward): {tpf:.4f}")
    print(f"Avg tokens per decoding step: {avg_tokens_per_decoding_step:.4f}")

    return x, nfe


def _generate_step_block_dynamic_internal(
    model, prompt, gen_length, block_length, temperature, remasking, mask_id,
    thresholds, tokenizer, attention_mask, threshold_cap=0.9,
    epsilon_ratio=0.05, decoding_strategy='threshold', decoding_factor=1.0
):
    """
    Internal function for step-block-wise dynamic thresholding.
    """
    B, L = prompt.shape
    x = torch.full((B, L + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :L] = prompt.clone()
    num_blocks = gen_length // block_length
    batch_indices = torch.arange(B, device=x.device)

    # =========================
    # Statistics
    # =========================
    forward_count = 0
    decoded_tokens = 0
    decoding_steps_used = 0

    for num_block in range(num_blocks):
        current_block_start = L + num_block * block_length
        current_block_end = current_block_start + block_length

        if decoding_strategy == 'threshold':
            current_threshold_schedule = thresholds[min(num_block, len(thresholds) - 1)]
            current_threshold_schedule_with_cap = [min(t, threshold_cap) for t in current_threshold_schedule]

        is_block_finished = torch.zeros(B, dtype=torch.bool, device=x.device)
        step = 0

        while not is_block_finished.all():
            if decoding_strategy == 'threshold':
                current_threshold = current_threshold_schedule_with_cap[
                    min(step, len(current_threshold_schedule_with_cap) - 1)
                ]
                epsilon = current_threshold * epsilon_ratio
                effective_threshold = current_threshold - epsilon

            active_indices = batch_indices[~is_block_finished]

            # Run model only on active sequences
            active_x = x[active_indices]
            active_attention_mask = (active_x != tokenizer.pad_token_id)

            all_logits = model(active_x, attention_mask=active_attention_mask).logits
            forward_count += 1

            # Perform calculations on the dense, active-only logits for numerical stability
            all_logits_with_noise = add_gumbel_noise(all_logits, temperature=temperature)
            all_x0 = torch.argmax(all_logits_with_noise, dim=-1)
            all_p = F.softmax(all_logits, dim=-1)

            # Create full-sized tensors to scatter results into
            x0 = torch.full_like(x, tokenizer.pad_token_id)
            p = torch.zeros((B, x.size(1), model.config.vocab_size), dtype=all_p.dtype, device=x.device)

            # Scatter the results back
            x0[active_indices] = all_x0
            p[active_indices] = all_p

            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            x0_p[:, L + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where((x == mask_id), x0, x)
            confidence = torch.where((x == mask_id), x0_p, -np.inf)

            if decoding_strategy == 'threshold':
                transfer_index = confidence > effective_threshold
            else:  # factor
                transfer_index = _get_transfer_index_factor_based(
                    confidence,
                    decoding_factor,
                    (x == mask_id)
                )

            # Fallback for sequences that are not finished but did not transfer any tokens
            stuck_mask = (
                ~is_block_finished
                & (x[:, current_block_start:current_block_end] == mask_id).any(dim=1)
                & ~transfer_index.any(dim=1)
            )

            if stuck_mask.any():
                stuck_indices = batch_indices[stuck_mask]

                # Get confidences for stuck sequences in the current block
                block_confidence_stuck = confidence[stuck_indices, current_block_start:current_block_end]

                # Find most confident token for each stuck sequence
                most_confident_local_idx_stuck = torch.argmax(block_confidence_stuck, dim=1)
                most_confident_global_idx_stuck = most_confident_local_idx_stuck + current_block_start

                # Force unmask for these specific tokens in the specific stuck sequences
                transfer_index[stuck_indices, most_confident_global_idx_stuck] = True

            # =========================
            # Statistics before commit
            # =========================
            num_selected = int(transfer_index.sum().item())

            if num_selected > 0:
                decoded_tokens += num_selected
                decoding_steps_used += 1

            x[transfer_index] = x0[transfer_index]

            is_block_now_full = torch.all(x[:, current_block_start:current_block_end] != mask_id, dim=1)
            is_block_finished = torch.logical_or(is_block_finished, is_block_now_full)
            step += 1

    # =========================
    # Decoding statistics
    # =========================
    nfe = forward_count
    tpf = decoded_tokens / forward_count if forward_count > 0 else 0.0
    avg_tokens_per_decoding_step = (
        decoded_tokens / decoding_steps_used if decoding_steps_used > 0 else 0.0
    )

    print("====== Decoding Statistics ======")
    print(f"Decoded tokens: {decoded_tokens}")
    print(f"Model forward calls: {forward_count}")
    print(f"Actual decoding steps with unmask: {decoding_steps_used}")
    print(f"TPF (tokens per forward): {tpf:.4f}")
    print(f"Avg tokens per decoding step: {avg_tokens_per_decoding_step:.4f}")

    return x, nfe


def _calculate_thresholds(reference_confidence_data, dynamic_mode, metric):
    """
    Calculates dynamic thresholds from reference confidence data.
    """
    if metric == 'average':
        metric_func = np.mean
    elif metric == 'q1':
        metric_func = lambda x: np.quantile(x, 0.25)
    elif metric == 'q2':
        metric_func = np.median
    elif metric == 'q3':
        metric_func = lambda x: np.quantile(x, 0.75)
    elif metric == 'minimum_whiskers':
        def metric_func(x):
            if len(x) < 2: return np.mean(x) if len(x) > 0 else 0
            q1 = np.quantile(x, 0.25)
            q3 = np.quantile(x, 0.75)
            iqr = q3 - q1
            return q1 - 1.5 * iqr
    else:
        raise ValueError(f"Unknown metric for threshold calculation: {metric}")

    if dynamic_mode == 'block':
        thresholds = []
        for block_data in reference_confidence_data:
            all_block_confs = [conf for step_confs in block_data for conf in step_confs]
            if all_block_confs:
                thresholds.append(metric_func(all_block_confs))
            else:
                thresholds.append(0) # Default for empty block
        return thresholds
    elif dynamic_mode == 'step_block':
        step_block_thresholds = []
        for block_data in reference_confidence_data:
            block_thresholds = [metric_func(step_confs) if step_confs else 0 for step_confs in block_data]
            step_block_thresholds.append(block_thresholds)
        return step_block_thresholds
    else:
        raise ValueError(f"Unknown dynamic_mode: {dynamic_mode}")


def generate(model, tokenizer, prompt, gen_length=128, block_length=32, dynamic_mode='block', 
             reference_confidence_data=None, threshold_metric='average', 
             decoding_strategy='threshold', decoding_factor=1.0,
             threshold_cap=0.9, epsilon_ratio=0.05, **kwargs):
    """
    Main orchestration function for adaptive inference-time generation.
    - If reference_confidence_data is None, runs in data collection mode.
    - If reference_confidence_data is provided, runs in dynamic thresholding mode.
    """
    mask_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None else 126336
    
    if reference_confidence_data is None:
        # Data collection mode for the first problem
        print("Running in data collection mode (for problem 1)...")
        # Use pop() to retrieve the value and remove it from kwargs to avoid duplication.
        steps = kwargs.pop('steps', 128)
        output_ids, nfe, collected_confidences = _generate_for_data_collection(
            model, prompt, steps=steps, gen_length=gen_length, block_length=block_length, mask_id=mask_id, **kwargs
        )
        return output_ids, nfe, collected_confidences
    else:
        # Dynamic generation mode for subsequent problems
        if decoding_strategy == 'threshold':
            print(f"Running in dynamic thresholding mode ('{dynamic_mode}') with metric '{threshold_metric}'...")
            thresholds = _calculate_thresholds(reference_confidence_data, dynamic_mode, metric=threshold_metric)
        else: # factor strategy
            print(f"Running with fixed factor decoding strategy (factor={decoding_factor}).")
            # The 'thresholds' variable will not be used, but we pass a dummy value to maintain function signature.
            thresholds = []

        if dynamic_mode == 'block':
            output_ids, nfe = _generate_block_dynamic_internal(
                model, prompt, gen_length, block_length, kwargs.get('temperature', 0.), kwargs.get('remasking', 'low_confidence'), mask_id, thresholds, tokenizer, kwargs.get('attention_mask'),
                threshold_cap=threshold_cap, epsilon_ratio=epsilon_ratio, decoding_strategy=decoding_strategy, decoding_factor=decoding_factor
            )
        elif dynamic_mode == 'step_block':
            output_ids, nfe = _generate_step_block_dynamic_internal(
                model, prompt, gen_length, block_length, kwargs.get('temperature', 0.), kwargs.get('remasking', 'low_confidence'), mask_id, thresholds, tokenizer, kwargs.get('attention_mask'),
                threshold_cap=threshold_cap, epsilon_ratio=epsilon_ratio, decoding_strategy=decoding_strategy, decoding_factor=decoding_factor
            )
        else:
            raise ValueError(f"Unknown dynamic_mode: {dynamic_mode}")
            
        return output_ids, nfe, None
    


def main():
    device = 'cuda'
    model = AutoModel.from_pretrained('/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct', trust_remote_code=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompts = ["Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"]

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    print("Collecting reference confidence data from the first problem...")
    _, nfe_ref, reference_confidence_data = generate(
        model,
        tokenizer,
        prompt=input_ids,
        gen_length=256,
        block_length=32,
        steps=256,
        temperature=0.0,
        cfg_scale=0.0,
    )

    print("Reference data collected. Proceeding with dynamic generation.")
    out, nfe_dyn, _ = generate(
        model,
        tokenizer,
        prompt=input_ids,
        gen_length=256,
        block_length=32,
        dynamic_mode="block",
        threshold_metric='q1',
        decoding_strategy='threshold',
        decoding_factor=1.0,
        reference_confidence_data=reference_confidence_data,
        temperature=0.0,
        remasking="low_confidence",
        threshold_cap=0.75,
        epsilon_ratio=0.20
    )
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()
    



