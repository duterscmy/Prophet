#!/usr/bin/env python3
"""Independent held-out validation of TC-APD Standard-trace consistency."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tc_apd.calibration import (
    build_grid_thresholds,
    iter_final_consistency_records,
    save_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--total-traces", type=int, default=5000)
    parser.add_argument("--estimation-traces", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--rho", type=float, default=0.995)
    parser.add_argument("--min-count", type=int, default=200)
    parser.add_argument("--min-accepted", type=int, default=100)
    parser.add_argument("--threshold-start", type=float, default=1.0)
    parser.add_argument("--threshold-end", type=float, default=0.01)
    parser.add_argument("--grid-step", type=float, default=0.005)
    parser.add_argument("--runtime-min-threshold", type=float, default=0.01)
    parser.add_argument("--runtime-max-threshold", type=float, default=0.90)
    parser.add_argument("--default-threshold", type=float, default=0.90)
    return parser.parse_args()


def wilson_interval(success: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = success / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def safe_rate(success: int, total: int) -> float:
    return success / total if total else math.nan


def support_group(value: int) -> str:
    if value == 0:
        return "0"
    if value < 10:
        return "1–9"
    if value < 25:
        return "10–24"
    if value < 50:
        return "25–49"
    if value < 100:
        return "50–99"
    return "≥100"


def estimation_support_group(value: int) -> str:
    if value < 200:
        return "100–199"
    if value < 500:
        return "200–499"
    if value < 1000:
        return "500–999"
    return "≥1000"


def aggregate_group(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    order = (
        ["100–199", "200–499", "500–999", "≥1000"]
        if group_column == "estimation_support_group"
        else ["0", "1–9", "10–24", "25–49", "50–99", "≥100"]
    )
    rows: list[dict[str, Any]] = []
    for label in order:
        group = frame[frame[group_column] == label]
        if group.empty:
            continue
        accepted_raw = int(group.heldout_accepted_raw.sum())
        success_raw = int(group.heldout_success_raw.sum())
        accepted_deployed = int(group.heldout_accepted_deployed.sum())
        success_deployed = int(group.heldout_success_deployed.sum())
        valid_raw = group[group.heldout_accepted_raw > 0]
        valid_deployed = group[group.heldout_accepted_deployed > 0]
        rows.append(
            {
                group_column: label,
                "token_types": len(group),
                "heldout_accepted_raw": accepted_raw,
                "heldout_precision_raw": safe_rate(success_raw, accepted_raw),
                "heldout_macro_precision_raw": valid_raw.heldout_precision_raw.mean(),
                "tokens_meeting_rho_raw_pct": 100 * valid_raw.meets_rho_raw.mean() if len(valid_raw) else math.nan,
                "heldout_accepted_deployed": accepted_deployed,
                "heldout_precision_deployed": safe_rate(success_deployed, accepted_deployed),
                "heldout_macro_precision_deployed": valid_deployed.heldout_precision_deployed.mean(),
                "tokens_meeting_rho_deployed_pct": 100 * valid_deployed.meets_rho_deployed.mean() if len(valid_deployed) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_policy(
    log: Path,
    *,
    total_traces: int,
    validation_indices: set[int],
    thresholds: dict[int, float],
    runtime_min: float,
    runtime_max: float,
    default_threshold: float,
) -> tuple[dict[int, list[int]], dict[str, int], set[int]]:
    # Per calibrated token: total, raw accepted/success, deployed accepted/success.
    per_token: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    policy = {
        "total_records": 0,
        "accepted": 0,
        "success": 0,
        "accepted_calibrated_token": 0,
        "success_calibrated_token": 0,
        "accepted_default_token": 0,
        "success_default_token": 0,
    }
    seen_traces: set[int] = set()
    for record in iter_final_consistency_records(log, max_traces_per_log=total_traces):
        trace_index = int(record["problem_idx"])
        if trace_index not in validation_indices:
            continue
        seen_traces.add(trace_index)
        token_id = int(record["token_id"])
        confidence = float(record["confidence"])
        consistent = int(bool(record["final_consistent"]))
        policy["total_records"] += 1
        if token_id in thresholds:
            raw_threshold = thresholds[token_id]
            deployed_threshold = min(runtime_max, max(runtime_min, raw_threshold))
            counts = per_token[token_id]
            counts[0] += 1
            if confidence >= raw_threshold:
                counts[1] += 1
                counts[2] += consistent
            if confidence >= deployed_threshold:
                counts[3] += 1
                counts[4] += consistent
            policy_threshold = deployed_threshold
            calibrated = True
        else:
            policy_threshold = min(runtime_max, max(runtime_min, default_threshold))
            calibrated = False
        if confidence >= policy_threshold:
            policy["accepted"] += 1
            policy["success"] += consistent
            branch = "calibrated_token" if calibrated else "default_token"
            policy[f"accepted_{branch}"] += 1
            policy[f"success_{branch}"] += consistent
    return dict(per_token), policy, seen_traces


def configure_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
        }
    )
    return plt


def make_figure(
    output_dir: Path,
    per_token: pd.DataFrame,
    support_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
    rho: float,
) -> None:
    plt = configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.3), constrained_layout=True)

    axes[0, 0].hist(per_token.threshold_raw, bins=30, color="#56B4E9", alpha=0.8)
    axes[0, 0].axvline(0.90, color="#D55E00", linestyle="--", label="Runtime max=0.90")
    axes[0, 0].set_title("(a) Estimated token thresholds")
    axes[0, 0].set_xlabel("Raw selected threshold")
    axes[0, 0].set_ylabel("Token types")
    axes[0, 0].legend(frameon=False)

    x = np.arange(len(support_summary))
    axes[0, 1].plot(x, 100 * support_summary.heldout_precision_raw, marker="o", label="Raw threshold")
    axes[0, 1].plot(x, 100 * support_summary.heldout_precision_deployed, marker="s", label="Runtime-clipped")
    axes[0, 1].axhline(100 * rho, color="black", linestyle="--", label=f"Target {100*rho:.1f}%")
    axes[0, 1].set_xticks(x, support_summary.estimation_support_group)
    axes[0, 1].set_title("(b) Held-out consistency by estimation support")
    axes[0, 1].set_xlabel("Accepted estimation observations")
    axes[0, 1].set_ylabel("Micro consistency (%)")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(True, axis="y")

    heldout_counts = per_token.heldout_support_group.value_counts()
    heldout_order = [x for x in ["0", "1–9", "10–24", "25–49", "50–99", "≥100"] if x in heldout_counts]
    axes[1, 0].bar(heldout_order, [heldout_counts[x] for x in heldout_order], color="#009E73")
    axes[1, 0].set_title("(c) Independent accepted support per token")
    axes[1, 0].set_xlabel("Held-out accepted observations (raw threshold)")
    axes[1, 0].set_ylabel("Token types")
    axes[1, 0].grid(True, axis="y")

    labels = policy_summary.label.tolist()
    values = 100 * policy_summary.precision.to_numpy()
    lower = 100 * policy_summary.ci95_low.to_numpy()
    upper = 100 * policy_summary.ci95_high.to_numpy()
    axes[1, 1].bar(np.arange(len(labels)), values, color=["#0072B2", "#D55E00", "#CC79A7"])
    axes[1, 1].errorbar(np.arange(len(labels)), values, yerr=[values-lower, upper-values], fmt="none", color="black", capsize=3)
    axes[1, 1].axhline(100 * rho, color="black", linestyle="--", label=f"Target {100*rho:.1f}%")
    axes[1, 1].set_xticks(np.arange(len(labels)), labels, rotation=12, ha="right")
    axes[1, 1].set_title("(d) Held-out aggregate consistency")
    axes[1, 1].set_ylabel("Consistency (%)")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(True, axis="y")

    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"figure_p04_heldout_consistency.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0 < args.estimation_traces < args.total_traces:
        raise ValueError("estimation-traces must be between zero and total-traces")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(args.total_traces))
    random.Random(args.seed).shuffle(indices)
    estimation_indices = set(indices[: args.estimation_traces])
    validation_indices = set(indices[args.estimation_traces :])

    observations: dict[int, list[tuple[float, int]]] = defaultdict(list)
    estimation_records = 0
    estimation_seen: set[int] = set()
    for record in iter_final_consistency_records(
        args.log, max_traces_per_log=args.total_traces
    ):
        trace_index = int(record["problem_idx"])
        if trace_index not in estimation_indices:
            continue
        estimation_seen.add(trace_index)
        observations[int(record["token_id"])].append(
            (float(record["confidence"]), int(bool(record["final_consistent"])))
        )
        estimation_records += 1
    if estimation_seen != estimation_indices:
        raise RuntimeError(
            f"Expected {len(estimation_indices)} estimation traces; found {len(estimation_seen)}"
        )

    thresholds, calibration_rows = build_grid_thresholds(
        observations,
        target_precision=args.rho,
        min_count=args.min_count,
        min_accepted=args.min_accepted,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        threshold_step=args.grid_step,
    )
    save_calibration(
        thresholds,
        calibration_rows,
        json_path=args.output_dir / "estimated_thresholds.json",
        csv_path=args.output_dir / "estimated_thresholds_audit.csv",
    )

    validation, policy, validation_seen = evaluate_policy(
        args.log,
        total_traces=args.total_traces,
        validation_indices=validation_indices,
        thresholds=thresholds,
        runtime_min=args.runtime_min_threshold,
        runtime_max=args.runtime_max_threshold,
        default_threshold=args.default_threshold,
    )
    if validation_seen != validation_indices:
        raise RuntimeError(
            f"Expected {len(validation_indices)} held-out traces; found {len(validation_seen)}"
        )

    calibration_by_token = {row.token_id: row for row in calibration_rows if row.status == "ok"}
    rows: list[dict[str, Any]] = []
    for token_id, threshold in sorted(thresholds.items()):
        estimate = calibration_by_token[token_id]
        total, accepted_raw, success_raw, accepted_deployed, success_deployed = validation.get(token_id, [0, 0, 0, 0, 0])
        raw_low, raw_high = wilson_interval(success_raw, accepted_raw)
        deployed_low, deployed_high = wilson_interval(success_deployed, accepted_deployed)
        raw_precision = safe_rate(success_raw, accepted_raw)
        deployed_precision = safe_rate(success_deployed, accepted_deployed)
        rows.append(
            {
                "token_id": token_id,
                "threshold_raw": threshold,
                "threshold_deployed": min(args.runtime_max_threshold, max(args.runtime_min_threshold, threshold)),
                "was_runtime_clipped": threshold > args.runtime_max_threshold or threshold < args.runtime_min_threshold,
                "estimation_total_count": estimate.total_count,
                "estimation_accepted": estimate.coverage,
                "estimation_success": estimate.success,
                "estimation_precision": estimate.precision,
                "estimation_support_group": estimation_support_group(estimate.coverage),
                "heldout_total_count": total,
                "heldout_accepted_raw": accepted_raw,
                "heldout_success_raw": success_raw,
                "heldout_precision_raw": raw_precision,
                "heldout_ci95_low_raw": raw_low,
                "heldout_ci95_high_raw": raw_high,
                "meets_rho_raw": raw_precision >= args.rho if accepted_raw else False,
                "wilson_low_meets_rho_raw": raw_low >= args.rho if accepted_raw else False,
                "heldout_support_group": support_group(accepted_raw),
                "heldout_accepted_deployed": accepted_deployed,
                "heldout_success_deployed": success_deployed,
                "heldout_precision_deployed": deployed_precision,
                "heldout_ci95_low_deployed": deployed_low,
                "heldout_ci95_high_deployed": deployed_high,
                "meets_rho_deployed": deployed_precision >= args.rho if accepted_deployed else False,
                "wilson_low_meets_rho_deployed": deployed_low >= args.rho if accepted_deployed else False,
            }
        )
    per_token = pd.DataFrame(rows)
    per_token.to_csv(args.output_dir / "per_token_heldout_validation.csv", index=False)

    support_summary = aggregate_group(per_token, "estimation_support_group")
    support_summary.to_csv(args.output_dir / "estimation_support_group_summary.csv", index=False)
    heldout_support_summary = aggregate_group(per_token, "heldout_support_group")
    heldout_support_summary.to_csv(args.output_dir / "heldout_support_group_summary.csv", index=False)

    aggregate_specs = [
        (
            "Raw selected thresholds",
            int(per_token.heldout_success_raw.sum()),
            int(per_token.heldout_accepted_raw.sum()),
        ),
        (
            "Runtime-clipped thresholds",
            int(per_token.heldout_success_deployed.sum()),
            int(per_token.heldout_accepted_deployed.sum()),
        ),
        ("Full lookup policy", policy["success"], policy["accepted"]),
    ]
    policy_rows = []
    for label, success, accepted in aggregate_specs:
        low, high = wilson_interval(success, accepted)
        policy_rows.append(
            {
                "label": label,
                "success": success,
                "accepted": accepted,
                "precision": safe_rate(success, accepted),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    policy_summary = pd.DataFrame(policy_rows)
    policy_summary.to_csv(args.output_dir / "aggregate_policy_summary.csv", index=False)

    valid_raw = per_token[per_token.heldout_accepted_raw > 0]
    valid_deployed = per_token[per_token.heldout_accepted_deployed > 0]
    summary = {
        "source_log": str(args.log),
        "seed": args.seed,
        "rho": args.rho,
        "split": {
            "total_traces": args.total_traces,
            "estimation_traces": len(estimation_indices),
            "heldout_traces": len(validation_indices),
            "estimation_candidate_records": estimation_records,
            "heldout_candidate_records": policy["total_records"],
        },
        "calibration": {
            "observed_token_ids": len(observations),
            "eligible_token_ids": len(calibration_rows),
            "calibrated_token_ids": len(thresholds),
            "fallback_no_valid_threshold_token_ids": sum(row.status != "ok" for row in calibration_rows),
            "runtime_clipped_token_ids": int(per_token.was_runtime_clipped.sum()),
        },
        "heldout_token_level": {
            "tokens_with_raw_accepted_support": len(valid_raw),
            "tokens_meeting_empirical_rho_raw_pct": 100 * valid_raw.meets_rho_raw.mean(),
            "tokens_wilson_low_meeting_rho_raw_pct": 100 * valid_raw.wilson_low_meets_rho_raw.mean(),
            "macro_precision_raw": valid_raw.heldout_precision_raw.mean(),
            "tokens_with_deployed_accepted_support": len(valid_deployed),
            "tokens_meeting_empirical_rho_deployed_pct": 100 * valid_deployed.meets_rho_deployed.mean(),
            "tokens_wilson_low_meeting_rho_deployed_pct": 100 * valid_deployed.wilson_low_meets_rho_deployed.mean(),
            "macro_precision_deployed": valid_deployed.heldout_precision_deployed.mean(),
        },
        "full_lookup_policy": {
            **policy,
            "precision": safe_rate(policy["success"], policy["accepted"]),
            "accepted_coverage_pct_of_candidate_records": 100 * policy["accepted"] / policy["total_records"],
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    raw_aggregate = policy_summary.iloc[0]
    deployed_aggregate = policy_summary.iloc[1]
    full_aggregate = policy_summary.iloc[2]
    markdown = [
        "# P0.4 Independent held-out consistency validation",
        "",
        f"Math Standard traces were split once with seed {args.seed}: {len(estimation_indices):,} traces for threshold estimation and {len(validation_indices):,} disjoint traces for validation.",
        "",
        "| Validation policy | Accepted records | Realized consistency | 95% Wilson CI |",
        "|---|---:|---:|---:|",
    ]
    for row in policy_summary.itertuples():
        markdown.append(
            f"| {row.label} | {row.accepted:,} | {100*row.precision:.3f}% | [{100*row.ci95_low:.3f}%, {100*row.ci95_high:.3f}%] |"
        )
    markdown.extend(
        [
            "",
            "## Token-level diagnostics",
            "",
            f"- Estimated valid token thresholds: {len(thresholds):,}.",
            f"- Tokens with nonzero held-out accepted support: {len(valid_raw):,}.",
            f"- Tokens meeting empirical {100*args.rho:.1f}% consistency on held-out data: {100*valid_raw.meets_rho_raw.mean():.2f}% (raw thresholds).",
            f"- Tokens whose 95% Wilson lower bound reaches {100*args.rho:.1f}%: {100*valid_raw.wilson_low_meets_rho_raw.mean():.2f}%.",
            f"- Runtime clipping changes {int(per_token.was_runtime_clipped.sum()):,} token thresholds.",
            "",
            "## Interpretation boundary",
            "",
            "This validates the offline consistency estimator on independent Standard-decoding traces. It does not establish consistency under TC-APD-generated trajectories and must not be presented as a formal deployment-time guarantee.",
        ]
    )
    (args.output_dir / "P04_HELDOUT_CONSISTENCY_SUMMARY.md").write_text("\n".join(markdown) + "\n")

    make_figure(args.output_dir, per_token, support_summary, policy_summary, args.rho)
    print(json.dumps(summary, indent=2))
    print(f"Wrote held-out validation artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
