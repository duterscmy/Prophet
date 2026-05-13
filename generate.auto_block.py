import torch
import json
import math
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    """Gumbel-Max sampling with float64 precision."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Calculate tokens to transfer at each step for uniform denoising."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def _safe_decode_token(tokenizer, token_id):
    if tokenizer is None:
        return None
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return None


def _rank_values(values):
    """
    Larger value gets larger rank.
    Only used for logging/analysis.
    """
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isnan(arr), -np.inf, arr)

    order = np.argsort(arr, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(arr), dtype=float)
    return ranks


def _get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_block_candidates(
    remaining,
    gen_length,
    adaptive_min_block_size=2,
    adaptive_max_block_size=None,
    adaptive_candidate_mode="power2",
    adaptive_candidate_stride=1,
):
    """
    Build candidate block sizes.

    Default:
      gen_length=256 -> max_block_size=128
      adaptive_candidate_mode="power2"
      candidates=[2,4,8,16,32,64,128]

    Dense mode:
      adaptive_candidate_mode="dense"
      adaptive_candidate_stride=1
      candidates=[2,3,4,...,128]
    """
    if remaining <= 0:
        return []

    if adaptive_max_block_size is None:
        adaptive_max_block_size = max(1, gen_length // 2)

    max_b = min(int(adaptive_max_block_size), int(remaining))
    min_b = min(int(adaptive_min_block_size), max_b)

    if remaining == 1:
        return [1]

    if adaptive_candidate_mode == "dense":
        stride = max(1, int(adaptive_candidate_stride))
        candidates = list(range(min_b, max_b + 1, stride))
        if max_b not in candidates:
            candidates.append(max_b)

    elif adaptive_candidate_mode == "power2":
        candidates = []
        b = 1
        while b < min_b:
            b *= 2

        while b <= max_b:
            candidates.append(b)
            b *= 2

        if not candidates:
            candidates = [max_b]

    else:
        raise ValueError(
            f"Unknown adaptive_candidate_mode={adaptive_candidate_mode}. "
            f"Use 'power2' or 'dense'."
        )

    if remaining < adaptive_min_block_size and remaining not in candidates:
        candidates = [remaining]

    return sorted(set(int(x) for x in candidates if 1 <= x <= remaining))


def _argmax_score_with_large_B_tiebreak(rows):
    """
    Select row with largest total_score.
    If tie, prefer larger B.
    """
    scores = np.asarray([r["total_score"] for r in rows], dtype=float)
    max_score = np.max(scores)

    candidate_indices = [
        i for i, s in enumerate(scores)
        if np.isclose(s, max_score, rtol=1e-8, atol=1e-12)
    ]

    best_idx = max(candidate_indices, key=lambda idx: rows[idx]["B"])
    return best_idx


def _compute_block_scores(
    conf,
    candidates,
    use_gap_score=True,
    use_length_compensation=True,
    positive_gap_only=True,
    gap_window_size=None,
    gap_context_mode="window",  # "window" or "block"
):
    """
    Adaptive block score.

    gap_context_mode="window":
        w = sqrt(B) by default.
        tail_mean(B) = mean(c_{B-w:B})
        post_mean(B) = mean(c_{B:B+w})
        gap(B)       = tail_mean(B) - post_mean(B)

    gap_context_mode="block":
        tail_mean(B) = mean(c_{0:B})
        post_mean(B) = mean(c_{B:2B})
        gap(B)       = tail_mean(B) - post_mean(B)

    Default final score:
        S(B) = max(0, gap(B)) * log(B)
    """
    conf = np.asarray(conf, dtype=float)

    if gap_context_mode not in ["window", "block"]:
        raise ValueError(
            f"Unknown gap_context_mode={gap_context_mode}. "
            f"Use 'window' or 'block'."
        )

    rows = []

    for B in candidates:
        B = int(B)
        if B < 1 or B > len(conf):
            continue

        if gap_context_mode == "window":
            if gap_window_size is None:
                w = max(2, int(math.sqrt(B)))
            else:
                w = max(1, int(gap_window_size))

            tail_start = max(0, B - w)
            tail_end = B
            post_start = B
            post_end = min(len(conf), B + w)

        else:
            # Compare the entire candidate block with the next same-length region.
            w = B
            tail_start = 0
            tail_end = B
            post_start = B
            post_end = min(len(conf), 2 * B)

        tail_conf = conf[tail_start:tail_end]
        post_conf = conf[post_start:post_end]

        if len(tail_conf) == 0:
            continue

        tail_mean = float(np.mean(tail_conf))
        tail_median = float(np.median(tail_conf))
        tail_min = float(np.min(tail_conf))
        tail_max = float(np.max(tail_conf))

        if len(post_conf) > 0:
            post_mean = float(np.mean(post_conf))
            post_median = float(np.median(post_conf))
            post_min = float(np.min(post_conf))
            post_max = float(np.max(post_conf))
            no_post_region = False
        else:
            # If there is no future region, this candidate reaches the end.
            # Treat post confidence as 0 so finishing can be selected.
            post_mean = 0.0
            post_median = None
            post_min = None
            post_max = None
            no_post_region = True

        gap = tail_mean - post_mean

        if positive_gap_only:
            gap_for_score = max(0.0, gap)
        else:
            gap_for_score = gap

        length_bonus = float(np.log(B))

        if use_gap_score and use_length_compensation:
            total_score = gap_for_score * length_bonus
        elif use_gap_score:
            total_score = gap_for_score
        elif use_length_compensation:
            total_score = length_bonus
        else:
            # If both are disabled, fall back to largest block through tie-break.
            total_score = 0.0

        rows.append({
            "B": B,
            "w": w,
            "gap_context_mode": gap_context_mode,

            "tail_start_offset": int(tail_start),
            "tail_end_offset": int(tail_end),
            "post_start_offset": int(post_start),
            "post_end_offset": int(post_end),

            "tail_mean": tail_mean,
            "tail_median": tail_median,
            "tail_min": tail_min,
            "tail_max": tail_max,

            "post_mean": post_mean,
            "post_median": post_median,
            "post_min": post_min,
            "post_max": post_max,
            "no_post_region": bool(no_post_region),

            "gap_tail_minus_post": float(gap),
            "gap_for_score": float(gap_for_score),
            "length_bonus_logB": length_bonus,
            "total_score": float(total_score),
        })

    if not rows:
        return None

    gap_values = [r["gap_tail_minus_post"] for r in rows]
    gap_for_score_values = [r["gap_for_score"] for r in rows]
    length_values = [r["length_bonus_logB"] for r in rows]
    total_values = [r["total_score"] for r in rows]

    gap_rank = _rank_values(gap_values)
    gap_for_score_rank = _rank_values(gap_for_score_values)
    length_rank = _rank_values(length_values)
    total_rank = _rank_values(total_values)

    for idx, row in enumerate(rows):
        row["rank_gap"] = float(gap_rank[idx])
        row["rank_gap_for_score"] = float(gap_for_score_rank[idx])
        row["rank_length"] = float(length_rank[idx])
        row["rank_total"] = float(total_rank[idx])

    # If all scores are zero, there is no clear positive gap.
    # Prefer the largest candidate for efficiency.
    if max(r["total_score"] for r in rows) <= 0:
        best_idx = max(range(len(rows)), key=lambda idx: rows[idx]["B"])
    else:
        best_idx = _argmax_score_with_large_B_tiebreak(rows)

    return {
        "selected_B": int(rows[best_idx]["B"]),
        "candidate_rows": rows,
        "selected_row": rows[best_idx],
    }


def _choose_adaptive_block_size(
    conf,
    coarse_candidates,
    use_gap_score=True,
    use_length_compensation=True,
    positive_gap_only=True,
    gap_window_size=None,
    gap_context_mode="window",
    adaptive_refine_candidates=False,
    adaptive_refine_stride=1,
):
    """
    First score coarse candidates.

    Optional refinement:
      If coarse best is 16 and its strongest neighbor is 32,
      search every integer in [16,32].
    """
    coarse_result = _compute_block_scores(
        conf=conf,
        candidates=coarse_candidates,
        use_gap_score=use_gap_score,
        use_length_compensation=use_length_compensation,
        positive_gap_only=positive_gap_only,
        gap_window_size=gap_window_size,
        gap_context_mode=gap_context_mode,
    )

    if coarse_result is None:
        return None

    final_result = coarse_result
    refine_info = None

    if adaptive_refine_candidates and len(coarse_candidates) >= 2:
        coarse_rows = coarse_result["candidate_rows"]
        coarse_Bs = [r["B"] for r in coarse_rows]
        coarse_scores = np.array([r["total_score"] for r in coarse_rows], dtype=float)

        best_B = coarse_result["selected_B"]
        best_idx = coarse_Bs.index(best_B)

        neighbor_indices = []
        if best_idx - 1 >= 0:
            neighbor_indices.append(best_idx - 1)
        if best_idx + 1 < len(coarse_Bs):
            neighbor_indices.append(best_idx + 1)

        if neighbor_indices:
            # Pick the higher-scoring neighbor.
            neighbor_idx = max(
                neighbor_indices,
                key=lambda idx: (coarse_scores[idx], coarse_Bs[idx])
            )
            neighbor_B = coarse_Bs[neighbor_idx]

            low = min(best_B, neighbor_B)
            high = max(best_B, neighbor_B)

            if high > low:
                stride = max(1, int(adaptive_refine_stride))
                refined_candidates = list(range(low, high + 1, stride))
                if high not in refined_candidates:
                    refined_candidates.append(high)

                refined_candidates = [
                    b for b in refined_candidates
                    if 1 <= b <= len(conf)
                ]

                refined_result = _compute_block_scores(
                    conf=conf,
                    candidates=refined_candidates,
                    use_gap_score=use_gap_score,
                    use_length_compensation=use_length_compensation,
                    positive_gap_only=positive_gap_only,
                    gap_window_size=gap_window_size,
                    gap_context_mode=gap_context_mode,
                )

                if refined_result is not None:
                    final_result = refined_result
                    refine_info = {
                        "enabled": True,
                        "coarse_best_B": int(best_B),
                        "neighbor_B": int(neighbor_B),
                        "refine_range": [int(low), int(high)],
                        "refine_stride": int(stride),
                        "refined_candidates": [int(b) for b in refined_candidates],
                        "refined_selected_B": int(refined_result["selected_B"]),
                    }

    return {
        "selected_B": int(final_result["selected_B"]),
        "coarse_result": coarse_result,
        "final_result": final_result,
        "refine_info": refine_info,
    }


def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    cfg_scale=0.,
    remasking='low_confidence',
    mask_id=126336,
    constraints=None,
    log=False,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    tokenizer=None,

    # Adaptive block-size options
    adaptive_block_size=True,
    adaptive_min_block_size=2,
    adaptive_max_block_size=None,
    adaptive_candidate_mode="power2",   # "power2" or "dense"
    adaptive_candidate_stride=1,

    # Fine-grained refinement
    adaptive_refine_candidates=False,
    adaptive_refine_stride=1,

    # Window/block gap score options
    use_gap_score=True,
    use_length_compensation=True,
    positive_gap_only=True,
    gap_window_size=None,
    gap_context_mode="window",  # "window" or "block"

    # Logging options
    log_all_probe_positions=True,
    print_probe_positions=False,
    dump_json_logs=False,
    return_logs=False,
):
    """
    Adaptive block diffusion generation.

    gap_context_mode="window":
        G(B) = mean(c_{B-w:B}) - mean(c_{B:B+w})

    gap_context_mode="block":
        G(B) = mean(c_{0:B}) - mean(c_{B:2B})

    Default:
        S(B) = max(0, G(B)) * log(B)

    Candidate block sizes:
        Default power-of-two:
            gen_length=256 -> [2,4,8,16,32,64,128]

        Dense:
            adaptive_candidate_mode="dense"
            adaptive_candidate_stride=1
            gen_length=256 -> [2,3,4,...,128]
    """
    print("======adaptive block generation, temperature: {:.1f}====".format(temperature))

    device = _get_model_device(model)
    prompt_len = prompt.shape[1]

    x = torch.full(
        (1, prompt_len + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    # Apply constraints before generation.
    if constraints is not None:
        for pos, token_id in constraints.items():
            absolute_pos = prompt_len + pos
            if absolute_pos < x.shape[1]:
                x[:, absolute_pos] = token_id

    prompt_index = (x != mask_id)

    block_selection_records = []
    token_unmask_records = []

    summary_records = {
        "gen_length": int(gen_length),
        "total_steps_budget": int(steps),
        "adaptive_block_size": bool(adaptive_block_size),
        "adaptive_min_block_size": int(adaptive_min_block_size),
        "adaptive_max_block_size": (
            int(adaptive_max_block_size)
            if adaptive_max_block_size is not None
            else int(max(1, gen_length // 2))
        ),
        "adaptive_candidate_mode": adaptive_candidate_mode,
        "adaptive_candidate_stride": int(adaptive_candidate_stride),
        "adaptive_refine_candidates": bool(adaptive_refine_candidates),
        "adaptive_refine_stride": int(adaptive_refine_stride),
        "score_terms": {
            "use_gap_score": bool(use_gap_score),
            "use_length_compensation": bool(use_length_compensation),
            "positive_gap_only": bool(positive_gap_only),
            "gap_window_size": gap_window_size,
            "gap_context_mode": gap_context_mode,
        },
        "selected_block_sizes": [],
        "block_step_counts": [],
    }

    def compute_logits(current_x):
        if cfg_scale > 0.:
            un_x = current_x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([current_x, un_x], dim=0)
            logits_ = model(x_).logits
            logits_cond, logits_uncond = torch.chunk(logits_, 2, dim=0)
            logits_ = logits_uncond + (cfg_scale + 1) * (logits_cond - logits_uncond)
        else:
            logits_ = model(current_x).logits

        if logits_eos_inf:
            logits_ = logits_.clone()
            logits_[:, :, 126081] = -torch.inf

        return logits_

    def logits_for_confidence(logits_):
        if confidence_eos_eot_inf:
            logits_ = logits_.clone()
            logits_[:, :, 126081] = -torch.inf
            logits_[:, :, 126348] = -torch.inf
        return logits_

    cursor = 0
    block_id = 0
    global_denoise_step = 0

    while cursor < gen_length:
        block_id += 1
        remaining = gen_length - cursor
        block_start = prompt_len + cursor

        # =========================================================
        # 1. Select block size
        # =========================================================
        if adaptive_block_size:
            with torch.no_grad():
                probe_logits = compute_logits(x)
                probe_logits_conf = logits_for_confidence(probe_logits)
                probe_probs = F.softmax(probe_logits_conf, dim=-1)
                probe_conf, probe_token = torch.max(probe_probs, dim=-1)

            future_start = prompt_len + cursor
            future_end = prompt_len + gen_length

            future_conf_tensor = probe_conf[0, future_start:future_end]
            future_token_tensor = probe_token[0, future_start:future_end]

            future_conf = future_conf_tensor.detach().float().cpu().numpy()
            future_token = future_token_tensor.detach().long().cpu().numpy()

            coarse_candidates = _build_block_candidates(
                remaining=remaining,
                gen_length=gen_length,
                adaptive_min_block_size=adaptive_min_block_size,
                adaptive_max_block_size=adaptive_max_block_size,
                adaptive_candidate_mode=adaptive_candidate_mode,
                adaptive_candidate_stride=adaptive_candidate_stride,
            )

            choice = _choose_adaptive_block_size(
                conf=future_conf,
                coarse_candidates=coarse_candidates,
                use_gap_score=use_gap_score,
                use_length_compensation=use_length_compensation,
                positive_gap_only=positive_gap_only,
                gap_window_size=gap_window_size,
                gap_context_mode=gap_context_mode,
                adaptive_refine_candidates=adaptive_refine_candidates,
                adaptive_refine_stride=adaptive_refine_stride,
            )

            if choice is None:
                selected_block_size = min(block_length, remaining)
                choice_record = None
            else:
                selected_block_size = min(int(choice["selected_B"]), remaining)
                choice_record = choice

            probe_position_records = []
            if log_all_probe_positions:
                for offset in range(len(future_conf)):
                    abs_pos = int(future_start + offset)
                    gen_pos = int(cursor + offset)
                    tok_id = int(future_token[offset])
                    probe_position_records.append({
                        "offset_from_current": int(offset),
                        "gen_position": gen_pos,
                        "absolute_position": abs_pos,
                        "confidence": float(future_conf[offset]),
                        "pred_token_id": tok_id,
                        "pred_token": _safe_decode_token(tokenizer, tok_id),
                        "is_currently_masked": bool(x[0, abs_pos].item() == mask_id),
                    })

            block_selection_record = {
                "block_id": int(block_id),
                "cursor_gen_position": int(cursor),
                "remaining": int(remaining),
                "coarse_candidates": [int(b) for b in coarse_candidates],
                "selected_block_size": int(selected_block_size),
                "selected_block_start_gen_position": int(cursor),
                "selected_block_end_gen_position_exclusive": int(cursor + selected_block_size),
                "choice": choice_record,
                "probe_positions": probe_position_records,
            }

        else:
            selected_block_size = min(int(block_length), remaining)
            block_selection_record = {
                "block_id": int(block_id),
                "cursor_gen_position": int(cursor),
                "remaining": int(remaining),
                "fixed_block_length": int(block_length),
                "selected_block_size": int(selected_block_size),
                "choice": None,
                "probe_positions": [],
            }

        block_selection_records.append(block_selection_record)

        if log:
            print("\n========== Block Selection ==========")
            print(f"block_id={block_id}")
            print(f"cursor={cursor}, remaining={remaining}")
            print(f"selected_block_size={selected_block_size}")
            print(f"gap_context_mode={gap_context_mode}")

            if adaptive_block_size and block_selection_record["choice"] is not None:
                print("coarse candidates:", block_selection_record["coarse_candidates"])

                print("\ncoarse candidate rows:")
                for row in block_selection_record["choice"]["coarse_result"]["candidate_rows"]:
                    print(row)

                if block_selection_record["choice"]["refine_info"] is not None:
                    print("\nrefine info:")
                    print(block_selection_record["choice"]["refine_info"])

                    print("\nrefined candidate rows:")
                    for row in block_selection_record["choice"]["final_result"]["candidate_rows"]:
                        print(row)

                print("\nselected row:")
                print(block_selection_record["choice"]["final_result"]["selected_row"])

            if print_probe_positions and log_all_probe_positions:
                print("\nprobe positions:")
                for item in probe_position_records:
                    print(item)

        # =========================================================
        # 2. Decode selected block
        # =========================================================
        block_end = block_start + selected_block_size
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_mask_in_block = int(block_mask_index.sum().item())

        if num_mask_in_block == 0:
            summary_records["selected_block_sizes"].append(int(selected_block_size))
            summary_records["block_step_counts"].append(0)
            cursor += selected_block_size
            continue

        # Distribute the global step budget proportionally to selected block size.
        # If steps == gen_length, this gives roughly one unmasking step per token.
        block_steps = max(
            1,
            int(round(float(steps) * selected_block_size / float(gen_length)))
        )
        block_steps = min(block_steps, num_mask_in_block)

        summary_records["selected_block_sizes"].append(int(selected_block_size))
        summary_records["block_step_counts"].append(int(block_steps))

        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, block_steps)

        if log:
            print(f"block token range: [{block_start}, {block_end})")
            print(f"num_mask_in_block={num_mask_in_block}, block_steps={block_steps}")

        for local_step in range(block_steps):
            global_denoise_step += 1

            with torch.no_grad():
                mask_index = (x == mask_id)
                logits = compute_logits(x)

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                logits_conf = logits_for_confidence(logits)

                if remasking == 'low_confidence':
                    p = F.softmax(logits_conf, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)),
                        -1,
                    )
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # Only allow positions inside current selected block to be transferred.
                x0_p[:, :block_start] = -np.inf
                x0_p[:, block_end:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

                for j in range(confidence.shape[0]):
                    k = int(num_transfer_tokens[j, local_step].item())
                    available = int((x[j, block_start:block_end] == mask_id).sum().item())
                    k = min(k, available)

                    if k <= 0:
                        continue

                    selected_conf, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True

                    for pos, conf_val in zip(select_index, selected_conf):
                        pos_int = int(pos.item())
                        token_id = int(x0[j, pos].item())
                        conf_float = float(conf_val.item())

                        token_unmask_records.append({
                            "global_denoise_step": int(global_denoise_step),
                            "block_id": int(block_id),
                            "local_step_in_block": int(local_step + 1),
                            "block_steps": int(block_steps),
                            "selected_block_size": int(selected_block_size),
                            "block_start_abs": int(block_start),
                            "block_end_abs_exclusive": int(block_end),
                            "position": pos_int,
                            "gen_position": int(pos_int - prompt_len),
                            "confidence": conf_float,
                            "token_id": token_id,
                            "token": _safe_decode_token(tokenizer, token_id),
                        })

                x[transfer_index] = x0[transfer_index]

                # Maintain constraints.
                if constraints is not None:
                    for pos, token_id in constraints.items():
                        absolute_pos = prompt_len + pos
                        if absolute_pos < x.shape[1]:
                            x[:, absolute_pos] = token_id

        cursor += selected_block_size

    selected_Bs = summary_records["selected_block_sizes"]
    summary_records["num_blocks"] = int(len(selected_Bs))
    summary_records["avg_block_size"] = float(np.mean(selected_Bs)) if selected_Bs else None
    summary_records["min_block_size"] = int(np.min(selected_Bs)) if selected_Bs else None
    summary_records["max_block_size"] = int(np.max(selected_Bs)) if selected_Bs else None
    summary_records["num_unmasked_tokens"] = int(len(token_unmask_records))

    log_payload = {
        "summary": summary_records,
        "block_selection_records": block_selection_records,
        "token_unmask_records": token_unmask_records,
    }

    if dump_json_logs:
        print(json.dumps(log_payload, ensure_ascii=False))
    else:
        print("num_unmask_records:", len(token_unmask_records))
        print("selected_block_sizes:", selected_Bs)

    if return_logs:
        return x, log_payload

    return x


def main():
    device = "cuda"

    model_path = "/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct"

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    prompt = """from typing import List
        def has_close_elements(numbers: List[float], threshold: float) -> bool:
         Check if in given list of numbers, are any two numbers closer to each other than
        given threshold.
        >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
        False
        >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
        True
"""

    m = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        m,
        add_generation_prompt=True,
        tokenize=False,
    )

    input_ids = tokenizer(prompt)["input_ids"]
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out, logs = generate(
        model,
        input_ids,
        gen_length=256,
        steps=256,
        temperature=0.,
        adaptive_block_size=True,

        # Candidate setting.
        adaptive_min_block_size=2,
        adaptive_max_block_size=None,

        # "power2": [2,4,8,16,32,64,128]
        # "dense":  [2,3,4,...,128]
        adaptive_candidate_mode="dense",
        adaptive_candidate_stride=1,

        # Fine-grained refinement.
        adaptive_refine_candidates=False,
        adaptive_refine_stride=1,

        # Score:
        # window mode:
        #   G(B)=mean(c_{B-w:B}) - mean(c_{B:B+w})
        # block mode:
        #   G(B)=mean(c_{0:B}) - mean(c_{B:2B})
        gap_context_mode="block",  # change to "block" for full block comparison
        # gap_context_mode="block",
        use_gap_score=True,
        use_length_compensation=False,
        positive_gap_only=True,
        gap_window_size=None,

        tokenizer=tokenizer,
        return_logs=True,
        log=True,
        log_all_probe_positions=True,
        print_probe_positions=False,
        dump_json_logs=False,
    )

    print(
        tokenizer.batch_decode(
            out[:, input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]
    )


if __name__ == "__main__":
    main()