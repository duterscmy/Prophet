#!/usr/bin/env python3
"""Build a conservative TC-APD table using a binomial lower bound."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tc_apd.calibration import iter_final_consistency_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--empirical-csv", required=True, type=Path)
    parser.add_argument("--traces", type=int, default=5000)
    parser.add_argument("--rho", type=float, default=0.995)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bound", choices=["wilson", "clopper-pearson"], default="wilson")
    parser.add_argument("--min-count", type=int, default=200)
    parser.add_argument("--min-accepted", type=int, default=100)
    parser.add_argument("--threshold-start", type=float, default=1.0)
    parser.add_argument("--threshold-end", type=float, default=0.01)
    parser.add_argument("--grid-step", type=float, default=0.005)
    return parser.parse_args()


def lower_bound(success: int, total: int, *, confidence_level: float, method: str) -> float:
    """One-sided binomial lower confidence bound."""
    if total <= 0:
        return math.nan
    if method == "clopper-pearson":
        if success == 0:
            return 0.0
        return float(beta.ppf(1.0 - confidence_level, success, total - success + 1))
    z = NormalDist().inv_cdf(confidence_level)
    p = success / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius)


def threshold_grid(start: float, end: float, step: float) -> list[float]:
    count = int(math.floor((start - end) / step + 1e-9))
    descending = [round(start - i * step, 10) for i in range(count + 1)]
    return list(reversed([x for x in descending if x >= end - 1e-10]))


def read_empirical(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.rho < 1.0:
        raise ValueError("rho must be in (0, 1)")
    if not 0.5 < args.confidence_level < 1.0:
        raise ValueError("confidence-level must be in (0.5, 1)")

    observations: dict[int, list[tuple[float, int]]] = defaultdict(list)
    seen_traces: set[int] = set()
    candidate_records = 0
    for record in iter_final_consistency_records(args.log, max_traces_per_log=args.traces):
        trace_index = int(record["problem_idx"])
        seen_traces.add(trace_index)
        observations[int(record["token_id"])].append(
            (float(record["confidence"]), int(bool(record["final_consistent"])))
        )
        candidate_records += 1
    if len(seen_traces) != args.traces:
        raise RuntimeError(f"Expected {args.traces} traces, found {len(seen_traces)}")

    grid = threshold_grid(args.threshold_start, args.threshold_end, args.grid_step)
    thresholds: dict[int, float] = {}
    rows: list[dict] = []
    for token_id in sorted(observations):
        valid = sorted(observations[token_id], key=lambda item: item[0])
        total_count = len(valid)
        if total_count < args.min_count:
            continue
        confidences = [item[0] for item in valid]
        suffix_success = [0] * (total_count + 1)
        for index in range(total_count - 1, -1, -1):
            suffix_success[index] = suffix_success[index + 1] + valid[index][1]
        chosen = None
        for threshold in grid:
            start_index = bisect.bisect_left(confidences, threshold)
            coverage = total_count - start_index
            if coverage < args.min_accepted:
                continue
            success = suffix_success[start_index]
            precision = success / coverage
            bound = lower_bound(
                success,
                coverage,
                confidence_level=args.confidence_level,
                method=args.bound,
            )
            if bound >= args.rho:
                chosen = {
                    "token_id": token_id,
                    "threshold": threshold,
                    "coverage": coverage,
                    "success": success,
                    "precision": precision,
                    "lower_bound": bound,
                    "status": "ok",
                    "total_count": total_count,
                }
                thresholds[token_id] = threshold
                break
        rows.append(chosen or {
            "token_id": token_id,
            "threshold": None,
            "coverage": 0,
            "success": 0,
            "precision": None,
            "lower_bound": None,
            "status": "fallback_no_valid_lower_bound",
            "total_count": total_count,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = "wilson95lower" if args.bound == "wilson" and args.confidence_level == 0.95 else f"{args.bound}_lower"
    stem = (
        f"math_n{args.traces}_{label}_p99.5_"
        f"mincount{args.min_count}_minaccepted{args.min_accepted}"
    )
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path.write_text(
        json.dumps({str(k): v for k, v in sorted(thresholds.items())}, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    empirical_rows = read_empirical(args.empirical_csv)
    empirical_ok = [row for row in empirical_rows if row["status"] == "ok"]
    conservative_ok = [row for row in rows if row["status"] == "ok"]
    empirical_thresholds = {int(row["token_id"]): float(row["threshold"]) for row in empirical_ok}
    common = sorted(set(empirical_thresholds) & set(thresholds))
    conservative_values = list(thresholds.values())
    summary = {
        "source_log": str(args.log),
        "traces": args.traces,
        "candidate_records": candidate_records,
        "bound": args.bound,
        "one_sided_confidence_level": args.confidence_level,
        "rho": args.rho,
        "min_count": args.min_count,
        "min_accepted": args.min_accepted,
        "eligible_tokens": len(rows),
        "empirical_valid_tokens": len(empirical_ok),
        "conservative_valid_tokens": len(conservative_ok),
        "valid_token_retention_pct": 100 * len(conservative_ok) / len(empirical_ok),
        "empirical_selected_accepted_records": sum(int(row["coverage"]) for row in empirical_ok),
        "conservative_selected_accepted_records": sum(int(row["coverage"]) for row in conservative_ok),
        "common_valid_tokens": len(common),
        "common_tokens_conservative_threshold_ge_empirical_pct": 100 * sum(
            thresholds[token] >= empirical_thresholds[token] for token in common
        ) / len(common) if common else math.nan,
        "threshold_min": min(conservative_values) if conservative_values else None,
        "threshold_median": quantile(conservative_values, 0.5),
        "threshold_max": max(conservative_values) if conservative_values else None,
        "thresholds_above_runtime_0.90": sum(x > 0.90 for x in conservative_values),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = [
        "# P1.3 Confidence-bound calibration summary",
        "",
        f"A one-sided {100*args.confidence_level:.0f}% {args.bound.title()} lower bound must reach the target consistency {100*args.rho:.1f}% at the selected threshold.",
        "",
        "| Statistic | Empirical rule | Confidence-bound rule |",
        "|---|---:|---:|",
        f"| Eligible token types | {len(empirical_rows):,} | {len(rows):,} |",
        f"| Valid token thresholds | {len(empirical_ok):,} | {len(conservative_ok):,} |",
        f"| Selected accepted calibration records | {summary['empirical_selected_accepted_records']:,} | {summary['conservative_selected_accepted_records']:,} |",
        f"| Thresholds above 0.90 | — | {summary['thresholds_above_runtime_0.90']:,} |",
        "",
        f"The conservative table retains {summary['valid_token_retention_pct']:.2f}% of the empirical table's valid token types. Among {len(common):,} common token types, {summary['common_tokens_conservative_threshold_ge_empirical_pct']:.2f}% receive an equal or higher threshold.",
        "",
        "Inference preserves conservative thresholds up to 1.0 instead of clipping them to 0.9. Tokens without a valid confidence-bound threshold retain the existing default fallback 0.9, so this experiment evaluates a conservative calibrated subset and is not a formal vocabulary-wide guarantee.",
        "",
    ]
    (args.output_dir / "P13_CONFIDENCE_BOUND_CALIBRATION.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
