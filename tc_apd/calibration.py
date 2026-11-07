"""Offline calibration utilities for TC-APD lookup tables.

The calibration path is deliberately independent of PyTorch so that large
standard-decoding logs can be processed on a CPU-only machine.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


Observation = tuple[float, int]
ObservationTable = dict[int, list[Observation]]


@dataclass(frozen=True)
class ThresholdStats:
    """Calibration statistics for one sufficiently frequent token."""

    token_id: int
    threshold: float | None
    coverage: int
    success: int
    precision: float | None
    status: str
    total_count: int


def _as_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    resolved = [Path(path).expanduser() for path in paths]
    if not resolved:
        raise ValueError("At least one standard-decoding log is required")
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"Standard-decoding log not found: {path}")
    return resolved


def parse_trace_payload(line: str) -> dict[str, Any] | None:
    """Extract a trace JSON object from a possibly prefixed log line."""
    if '"selected_records"' not in line or '"all_token_records"' not in line:
        return None
    start = line.find("{")
    if start < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(line[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if "selected_records" not in payload or "all_token_records" not in payload:
        return None
    return payload


def iter_trace_payloads(
    paths: str | Path | Iterable[str | Path],
    *,
    max_traces_per_log: int | None = None,
) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    """Stream trace payloads without loading complete log files into memory."""
    if max_traces_per_log is not None and max_traces_per_log <= 0:
        raise ValueError("max_traces_per_log must be positive or None")
    for path in _as_paths(paths):
        trace_index = 0
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                payload = parse_trace_payload(line)
                if payload is None:
                    continue
                yield path, trace_index, payload
                trace_index += 1
                if max_traces_per_log is not None and trace_index >= max_traces_per_log:
                    break


def iter_final_consistency_records(
    paths: str | Path | Iterable[str | Path],
    *,
    max_traces_per_log: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield labeled intermediate predictions from standard traces.

    The label is one exactly when the intermediate top-1 token equals the
    token eventually committed at the same generation-relative position.
    """
    for path, trace_index, payload in iter_trace_payloads(
        paths, max_traces_per_log=max_traces_per_log
    ):
        final_token_by_position: dict[int, int] = {}
        commit_step_by_position: dict[int, int] = {}
        for selected in payload["selected_records"]:
            position = int(selected["generation_relative_position"])
            final_token_by_position[position] = int(selected["token_id"])
            if "global_step" in selected:
                commit_step_by_position[position] = int(selected["global_step"])

        for candidate in payload["all_token_records"]:
            position = int(candidate["generation_relative_position"])
            final_token_id = final_token_by_position.get(position)
            if final_token_id is None:
                continue
            confidence = float(candidate["confidence"])
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                continue
            token_id = int(candidate["token_id"])
            yield {
                **candidate,
                "source_log": str(path),
                "problem_idx": trace_index,
                "generation_relative_position": position,
                "confidence": confidence,
                "token_id": token_id,
                "final_token_id": final_token_id,
                "commit_step": commit_step_by_position.get(position),
                "final_consistent": token_id == final_token_id,
            }


def collect_final_consistency(
    paths: str | Path | Iterable[str | Path],
    *,
    max_traces_per_log: int | None = None,
) -> ObservationTable:
    """Group labeled ``(confidence, final_consistent)`` pairs by token id."""
    observations: dict[int, list[Observation]] = defaultdict(list)
    for record in iter_final_consistency_records(
        paths, max_traces_per_log=max_traces_per_log
    ):
        observations[record["token_id"]].append(
            (record["confidence"], int(record["final_consistent"]))
        )
    if not observations:
        raise RuntimeError(
            "No valid trace payloads were found. The log must contain both "
            "selected_records and all_token_records."
        )
    return dict(observations)


def _threshold_grid(start: float, end: float, step: float) -> list[float]:
    if not 0.0 <= end <= start <= 1.0:
        raise ValueError("Threshold range must satisfy 0 <= end <= start <= 1")
    if not 0.0 < step <= 1.0:
        raise ValueError("threshold_step must be in (0, 1]")
    count = int(math.floor((start - end) / step + 1e-9))
    descending = [round(start - index * step, 10) for index in range(count + 1)]
    return [threshold for threshold in descending if threshold >= end - 1e-10]


def build_grid_thresholds(
    observations: Mapping[int, Sequence[Observation]],
    *,
    target_precision: float = 0.995,
    min_count: int = 200,
    min_accepted: int = 100,
    threshold_start: float = 1.0,
    threshold_end: float = 0.01,
    threshold_step: float = 0.005,
) -> tuple[dict[int, float], list[ThresholdStats]]:
    """Choose the lowest feasible grid threshold for every eligible token."""
    if target_precision > 1.0:
        target_precision /= 100.0
    if not 0.0 < target_precision <= 1.0:
        raise ValueError("target_precision must be in (0, 1] or a percentage")
    if min_count <= 0 or min_accepted <= 0:
        raise ValueError("min_count and min_accepted must be positive")

    # The notebook scanned high to low and overwrote successful choices. This
    # ascending scan is equivalent and stops at the first feasible threshold.
    grid = list(
        reversed(_threshold_grid(threshold_start, threshold_end, threshold_step))
    )
    threshold_table: dict[int, float] = {}
    rows: list[ThresholdStats] = []

    for raw_token_id in sorted(observations):
        token_id = int(raw_token_id)
        valid = [
            (float(confidence), int(bool(consistent)))
            for confidence, consistent in observations[raw_token_id]
            if math.isfinite(float(confidence)) and 0.0 <= float(confidence) <= 1.0
        ]
        total_count = len(valid)
        if total_count < min_count:
            continue

        ordered = sorted(valid, key=lambda item: item[0])
        confidences = [item[0] for item in ordered]
        suffix_success = [0] * (total_count + 1)
        for index in range(total_count - 1, -1, -1):
            suffix_success[index] = suffix_success[index + 1] + ordered[index][1]

        chosen: ThresholdStats | None = None
        for threshold in grid:
            start_index = bisect.bisect_left(confidences, threshold)
            coverage = total_count - start_index
            if coverage < min_accepted:
                continue
            success = suffix_success[start_index]
            precision = success / coverage
            if precision >= target_precision:
                chosen = ThresholdStats(
                    token_id=token_id,
                    threshold=threshold,
                    coverage=coverage,
                    success=success,
                    precision=precision,
                    status="ok",
                    total_count=total_count,
                )
                threshold_table[token_id] = threshold
                break

        rows.append(
            chosen
            or ThresholdStats(
                token_id=token_id,
                threshold=None,
                coverage=0,
                success=0,
                precision=None,
                status="fallback_no_valid_threshold",
                total_count=total_count,
            )
        )
    return threshold_table, rows


def calibrate_thresholds(
    observations: Mapping[int, Sequence[Observation]],
    *,
    rho: float,
    min_token_observations: int,
    min_accepted_observations: int,
    grid_step: float,
    threshold_start: float = 1.0,
    threshold_end: float = 0.01,
) -> dict[int, float]:
    """Backward-compatible threshold-only calibration interface."""
    thresholds, _ = build_grid_thresholds(
        observations,
        target_precision=rho,
        min_count=min_token_observations,
        min_accepted=min_accepted_observations,
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        threshold_step=grid_step,
    )
    return thresholds


def save_calibration(
    thresholds: Mapping[int, float],
    rows: Sequence[ThresholdStats],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    """Save the inference lookup JSON and the complete calibration audit CSV."""
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                str(token_id): float(threshold)
                for token_id, threshold in sorted(thresholds.items())
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )
        stream.write("\n")

    fieldnames = [
        "token_id",
        "threshold",
        "coverage",
        "success",
        "precision",
        "status",
        "total_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def load_final_consistency_dataframe(
    paths: str | Path | Iterable[str | Path],
    *,
    max_traces_per_log: int | None = None,
):
    """Load labeled records as a DataFrame for optional notebook analysis."""
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - optional analysis path
        raise ImportError("pandas is required for DataFrame analysis") from error
    records = list(
        iter_final_consistency_records(paths, max_traces_per_log=max_traces_per_log)
    )
    if not records:
        raise RuntimeError("No valid all_token_records found")
    return pd.DataFrame.from_records(records)


def add_confidence_bins(dataframe, bins):
    """Add the confidence-bin columns used by analysis_confidence.ipynb."""
    import numpy as np
    import pandas as pd

    result = dataframe.copy()
    result["conf_bin"] = pd.cut(
        result["confidence"], bins=bins, include_lowest=True, right=False
    )
    result["conf_bin_mid"] = result["conf_bin"].apply(
        lambda interval: (
            (interval.left + interval.right) / 2 if pd.notna(interval) else np.nan
        )
    )
    result["conf_bin_label"] = result["conf_bin"].astype(str)
    return result


def compute_bin_stats(dataframe, group_columns):
    """Aggregate final consistency for confidence-bin visualizations."""
    return (
        dataframe.groupby(group_columns, observed=True)
        .agg(
            final_consistency_rate=("final_consistent", "mean"),
            count=("final_consistent", "size"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
