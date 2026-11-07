"""Clear reference implementation of standard decoding, APD, and TC-APD."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import torch
import torch.nn.functional as F

from .thresholds import TokenThresholdTable


DecodeMode = Literal["standard", "fixed_apd", "tc_apd"]


@dataclass(frozen=True)
class DecodeConfig:
    mode: DecodeMode = "standard"
    steps: int = 256
    gen_length: int = 256
    block_length: int = 32
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    mask_id: int = 126336

    confidence_threshold: float = 0.9
    default_threshold: float = 0.9
    min_threshold: float = 0.05
    max_threshold: float = 0.9
    min_parallel_tokens: int = 1
    max_parallel_tokens: int = 100
    collect_traces: bool = True

    def validate(self) -> None:
        if self.mode not in {"standard", "fixed_apd", "tc_apd"}:
            raise ValueError(f"Unsupported decoder mode: {self.mode}")
        if self.steps <= 0 or self.gen_length <= 0 or self.block_length <= 0:
            raise ValueError("steps, gen_length, and block_length must be positive")
        if self.gen_length % self.block_length != 0:
            raise ValueError("gen_length must be divisible by block_length")
        num_blocks = self.gen_length // self.block_length
        if self.mode == "standard" and self.steps % num_blocks != 0:
            raise ValueError("standard-decoding steps must be divisible by num_blocks")
        if self.remasking not in {"low_confidence", "random"}:
            raise ValueError(f"Unsupported remasking strategy: {self.remasking}")
        if not 0.0 <= self.min_threshold <= self.max_threshold <= 1.0:
            raise ValueError("threshold bounds must satisfy 0 <= min <= max <= 1")
        if self.min_parallel_tokens < 1:
            raise ValueError("min_parallel_tokens must be at least 1")
        if self.max_parallel_tokens < self.min_parallel_tokens:
            raise ValueError("max_parallel_tokens must be >= min_parallel_tokens")


@dataclass
class DecodeResult:
    sequences: torch.Tensor
    mode: str
    nfe: int
    elapsed_seconds: float
    selected_records: list[dict[str, Any]] = field(default_factory=list)
    candidate_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def stats(self) -> dict[str, Any]:
        decoded_tokens = len(self.selected_records)
        parallel_tokens = sum(
            record.get("strategy") == "parallel" for record in self.selected_records
        )
        fallback_tokens = sum(
            record.get("strategy") == "single" for record in self.selected_records
        )
        return {
            "decoder": self.mode,
            "decoded_tokens": decoded_tokens,
            "model_forward_calls": self.nfe,
            "steps": self.nfe,
            "tpf": decoded_tokens / self.nfe if self.nfe else 0.0,
            "parallel_decoded_tokens": parallel_tokens,
            "single_decoded_tokens": fallback_tokens,
            "num_all_token_records": len(self.candidate_records),
            "elapsed_seconds": self.elapsed_seconds,
        }

    def trace_payload(self) -> dict[str, Any]:
        return {
            "selected_records": self.selected_records,
            "all_token_records": self.candidate_records,
            "stats": self.stats,
        }

    def print_report(self, *, include_traces: bool = True) -> None:
        stats = self.stats
        print("====== Decoding Statistics ======")
        print(f"Decoder: {self.mode}")
        print(f"Decoded tokens: {stats['decoded_tokens']}")
        print(f"Model forward calls: {self.nfe}")
        print(f"TPF (tokens per forward): {stats['tpf']:.4f}")
        print(f"Decode time: {self.elapsed_seconds:.4f}s")
        if include_traces:
            print(json.dumps(self.trace_payload(), ensure_ascii=False))
            # Keep the selected-only line consumed by older analysis scripts.
            print(json.dumps(self.selected_records, ensure_ascii=False))
            print(len(self.selected_records))


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** temperature)


def _transfer_schedule(
    mask_count: int, steps: int, device: torch.device
) -> torch.Tensor:
    base, remainder = divmod(mask_count, steps)
    schedule = torch.full((steps,), base, dtype=torch.long, device=device)
    schedule[:remainder] += 1
    return schedule


def _forward_logits(
    model: Any,
    sequences: torch.Tensor,
    prompt_index: torch.Tensor,
    config: DecodeConfig,
) -> torch.Tensor:
    if config.cfg_scale <= 0:
        return model(sequences).logits

    unconditional = sequences.clone()
    unconditional[prompt_index] = config.mask_id
    combined = torch.cat([sequences, unconditional], dim=0)
    conditional_logits, unconditional_logits = torch.chunk(
        model(combined).logits, 2, dim=0
    )
    return unconditional_logits + (config.cfg_scale + 1.0) * (
        conditional_logits - unconditional_logits
    )


def _predict(
    logits: torch.Tensor,
    config: DecodeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = torch.argmax(_add_gumbel_noise(logits, config.temperature), dim=-1)
    if config.remasking == "random":
        confidence = torch.rand(predictions.shape, device=predictions.device)
    else:
        probabilities = F.softmax(logits.float(), dim=-1)
        confidence = probabilities.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
    return predictions, confidence


def _apply_constraints(
    sequences: torch.Tensor,
    prompt_length: int,
    constraints: Mapping[int, int] | None,
) -> None:
    if not constraints:
        return
    for relative_position, token_id in constraints.items():
        position = prompt_length + int(relative_position)
        if prompt_length <= position < sequences.shape[1]:
            sequences[:, position] = int(token_id)


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@torch.no_grad()
def decode(
    model: Any,
    prompt: torch.Tensor,
    config: DecodeConfig,
    *,
    threshold_table: TokenThresholdTable | None = None,
    constraints: Mapping[int, int] | None = None,
) -> DecodeResult:
    """Decode one prompt with standard, fixed APD, or TC-APD.

    TC-APD follows Algorithm A1 from the paper: commit all candidates whose
    confidence meets the predicted token's calibrated threshold; if none are
    eligible, commit the single highest-confidence candidate.
    """
    config.validate()
    if prompt.shape[0] != 1:
        raise ValueError("The reference decoder currently expects batch size 1")
    if config.mode == "tc_apd" and threshold_table is None:
        raise ValueError("tc_apd requires a token threshold table")

    device = prompt.device
    _sync(device)
    started_at = time.perf_counter()
    prompt_length = prompt.shape[1]
    sequences = torch.full(
        (1, prompt_length + config.gen_length),
        config.mask_id,
        dtype=prompt.dtype,
        device=device,
    )
    sequences[:, :prompt_length] = prompt
    _apply_constraints(sequences, prompt_length, constraints)
    prompt_index = sequences != config.mask_id

    selected_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    nfe = 0
    num_blocks = config.gen_length // config.block_length
    steps_per_block = config.steps // num_blocks

    for block_index in range(num_blocks):
        block_start = prompt_length + block_index * config.block_length
        block_end = block_start + config.block_length
        initial_masks = int(
            (sequences[0, block_start:block_end] == config.mask_id).sum().item()
        )
        standard_schedule = (
            _transfer_schedule(initial_masks, steps_per_block, device)
            if config.mode == "standard"
            else None
        )
        local_step = 0

        while (sequences[0, block_start:block_end] == config.mask_id).any():
            if config.mode == "standard" and local_step >= steps_per_block:
                break
            if config.mode != "standard" and nfe >= config.steps:
                break

            local_step += 1
            mask_index = sequences == config.mask_id
            candidate_positions = (
                torch.where(mask_index[0, block_start:block_end])[0] + block_start
            )
            logits = _forward_logits(model, sequences, prompt_index, config)
            nfe += 1
            predictions, confidence = _predict(logits, config)
            candidate_confidence = confidence[0, candidate_positions]
            candidate_token_ids = predictions[0, candidate_positions]

            raw_thresholds: list[float | None] = []
            applied_thresholds: list[float | None] = []
            used_defaults: list[bool] = []
            clipped: list[bool] = []

            if config.mode == "standard":
                assert standard_schedule is not None
                commit_count = min(
                    int(standard_schedule[local_step - 1].item()),
                    len(candidate_positions),
                )
                selected_indices = torch.topk(
                    candidate_confidence, k=commit_count
                ).indices
                strategy = "standard"
                raw_thresholds = [None] * len(candidate_positions)
                applied_thresholds = [None] * len(candidate_positions)
                used_defaults = [False] * len(candidate_positions)
                clipped = [False] * len(candidate_positions)
            else:
                if config.mode == "fixed_apd":
                    raw_thresholds = [config.confidence_threshold] * len(
                        candidate_positions
                    )
                    applied_thresholds = list(raw_thresholds)
                    used_defaults = [False] * len(candidate_positions)
                    clipped = [False] * len(candidate_positions)
                else:
                    assert threshold_table is not None
                    for token_id in candidate_token_ids.tolist():
                        raw, applied, used_default, was_clipped = (
                            threshold_table.lookup(
                                token_id,
                                minimum=config.min_threshold,
                                maximum=config.max_threshold,
                                default_threshold=config.default_threshold,
                            )
                        )
                        raw_thresholds.append(raw)
                        applied_thresholds.append(applied)
                        used_defaults.append(used_default)
                        clipped.append(was_clipped)

                thresholds = torch.tensor(
                    applied_thresholds,
                    dtype=candidate_confidence.dtype,
                    device=device,
                )
                eligible = torch.where(candidate_confidence >= thresholds)[0]
                if len(eligible) >= config.min_parallel_tokens:
                    commit_count = min(len(eligible), config.max_parallel_tokens)
                    best = torch.topk(
                        candidate_confidence[eligible], k=commit_count
                    ).indices
                    selected_indices = eligible[best]
                    strategy = "parallel"
                else:
                    selected_indices = torch.argmax(candidate_confidence).view(1)
                    strategy = "single"

            selected_set = set(selected_indices.tolist())
            global_step = nfe
            for candidate_index, position in enumerate(candidate_positions.tolist()):
                threshold = applied_thresholds[candidate_index]
                record = {
                    "global_step": global_step,
                    "local_step": local_step,
                    "block": block_index + 1,
                    "position": position,
                    "block_relative_position": position - block_start,
                    "generation_relative_position": position - prompt_length,
                    "confidence": float(candidate_confidence[candidate_index].item()),
                    "token_id": int(candidate_token_ids[candidate_index].item()),
                    "raw_threshold": raw_thresholds[candidate_index],
                    "threshold": threshold,
                    "used_default_threshold": used_defaults[candidate_index],
                    "threshold_clipped": clipped[candidate_index],
                    "eligible": (
                        True
                        if config.mode == "standard"
                        else float(candidate_confidence[candidate_index].item())
                        >= float(threshold)
                    ),
                    "selected": candidate_index in selected_set,
                    "strategy": (strategy if candidate_index in selected_set else None),
                    "parallel_group_size": (
                        len(selected_indices)
                        if candidate_index in selected_set
                        else None
                    ),
                    "remaining_masks_in_block": len(candidate_positions),
                }
                if config.collect_traces:
                    candidate_records.append(record)
                if candidate_index in selected_set:
                    selected_records.append({**record, "step": local_step})

            for candidate_index in selected_indices.tolist():
                position = int(candidate_positions[candidate_index].item())
                sequences[0, position] = candidate_token_ids[candidate_index]
            _apply_constraints(sequences, prompt_length, constraints)

        if (sequences[0, block_start:block_end] == config.mask_id).any():
            raise RuntimeError(
                f"Decoder exhausted its step budget with masks remaining in block "
                f"{block_index + 1}; increase steps"
            )

    _sync(device)
    return DecodeResult(
        sequences=sequences,
        mode=config.mode,
        nfe=nfe,
        elapsed_seconds=time.perf_counter() - started_at,
        selected_records=selected_records,
        candidate_records=candidate_records,
    )
