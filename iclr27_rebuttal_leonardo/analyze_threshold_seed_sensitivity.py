#!/usr/bin/env python3
"""Compare independently sampled threshold tables of a given subset size."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from transformers import AutoTokenizer


SEEDS = [2027, 2028, 2029]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--size", type=int, default=100)
    return parser.parse_args()


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {pattern}, found {matches}")
    return matches[0]


def load_inputs(root: Path, size: int):
    tables: dict[int, dict[int, float]] = {}
    audits: dict[int, dict[int, dict[str, str]]] = {}
    paths: dict[int, dict[str, str]] = {}
    for seed in SEEDS:
        stem = f"math_n{size}_seed{seed}_token_threshold_grid_p99.5_mincount200_minaccepted100"
        json_path = find_one(root, f"{stem}.json")
        csv_path = find_one(root, f"{stem}.csv")
        raw = json.loads(json_path.read_text())
        tables[seed] = {int(k): float(v) for k, v in raw.items()}
        with csv_path.open(newline="") as handle:
            audits[seed] = {
                int(row["token_id"]): row for row in csv.DictReader(handle)
            }
        paths[seed] = {"json": str(json_path), "csv": str(csv_path)}
    return tables, audits, paths


def configure_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "savefig.dpi": 300,
        }
    )
    return plt


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables, audits, paths = load_inputs(args.threshold_root, args.size)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )

    union = set().union(*(set(table) for table in tables.values()))
    intersection = set.intersection(*(set(table) for table in tables.values()))
    membership_rows: list[dict[str, Any]] = []
    for token_id in sorted(union):
        row: dict[str, Any] = {
            "token_id": token_id,
            "token_piece": tokenizer.convert_ids_to_tokens(token_id),
            "decoded_text": tokenizer.decode([token_id], skip_special_tokens=False),
        }
        present = []
        values = []
        supports = []
        for seed in SEEDS:
            value = tables[seed].get(token_id, math.nan)
            is_present = token_id in tables[seed]
            present.append(is_present)
            row[f"present_seed{seed}"] = is_present
            row[f"threshold_seed{seed}"] = value
            audit = audits[seed].get(token_id)
            support = int(audit["total_count"]) if audit else 0
            accepted = int(audit["coverage"]) if audit and audit["status"] == "ok" else 0
            row[f"total_count_seed{seed}"] = support
            row[f"accepted_seed{seed}"] = accepted
            if is_present:
                values.append(value)
                supports.append(support)
        row["num_seeds_present"] = sum(present)
        row["membership"] = "".join("1" if value else "0" for value in present)
        row["threshold_mean_present"] = float(np.mean(values))
        row["threshold_range_present"] = float(max(values) - min(values))
        row["min_total_count_present"] = min(supports)
        row["mean_total_count_present"] = float(np.mean(supports))
        membership_rows.append(row)
    membership = pd.DataFrame(membership_rows)
    membership.to_csv(args.output_dir / "token_membership_and_thresholds.csv", index=False)

    per_seed_rows = []
    for seed in SEEDS:
        audit = audits[seed]
        valid = len(tables[seed])
        eligible = len(audit)
        per_seed_rows.append(
            {
                "seed": seed,
                "valid_token_thresholds": valid,
                "eligible_token_ids": eligible,
                "fallback_no_valid_threshold": eligible - valid,
                "valid_among_eligible_pct": 100 * valid / eligible,
                "union_coverage_pct": 100 * valid / len(union),
                "all_seed_intersection_coverage_pct": 100 * len(intersection) / valid,
            }
        )
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(args.output_dir / "per_seed_table_summary.csv", index=False)

    pairwise_rows = []
    for left, right in combinations(SEEDS, 2):
        left_ids, right_ids = set(tables[left]), set(tables[right])
        shared = sorted(left_ids & right_ids)
        left_values = np.asarray([tables[left][token] for token in shared])
        right_values = np.asarray([tables[right][token] for token in shared])
        delta = np.abs(left_values - right_values)
        pairwise_rows.append(
            {
                "seed_left": left,
                "seed_right": right,
                "left_tokens": len(left_ids),
                "right_tokens": len(right_ids),
                "shared_tokens": len(shared),
                "union_tokens": len(left_ids | right_ids),
                "jaccard": len(shared) / len(left_ids | right_ids),
                "overlap_coefficient": len(shared) / min(len(left_ids), len(right_ids)),
                "exact_threshold_agreement_pct": 100 * float(np.mean(delta == 0)),
                "within_001_pct": 100 * float(np.mean(delta <= 0.01 + 1e-12)),
                "within_005_pct": 100 * float(np.mean(delta <= 0.05 + 1e-12)),
                "mean_abs_threshold_diff": float(np.mean(delta)),
                "median_abs_threshold_diff": float(np.median(delta)),
                "p90_abs_threshold_diff": float(np.quantile(delta, 0.9)),
                "spearman_threshold_correlation": float(spearmanr(left_values, right_values).statistic),
            }
        )
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(args.output_dir / "pairwise_threshold_sensitivity.csv", index=False)

    pattern_counts = (
        membership.groupby("membership", as_index=False)
        .agg(token_types=("token_id", "size"))
        .sort_values("token_types", ascending=False)
    )
    pattern_counts.to_csv(args.output_dir / "membership_pattern_counts.csv", index=False)
    common = membership[membership.num_seeds_present == 3].sort_values(
        ["min_total_count_present", "mean_total_count_present"], ascending=False
    )
    common.head(100).to_csv(args.output_dir / "top100_common_tokens.csv", index=False)
    for seed_index, seed in enumerate(SEEDS):
        pattern = [False] * len(SEEDS)
        pattern[seed_index] = True
        encoded = "".join("1" if x else "0" for x in pattern)
        membership[membership.membership == encoded].sort_values(
            "mean_total_count_present", ascending=False
        ).to_csv(args.output_dir / f"tokens_unique_to_seed{seed}.csv", index=False)

    shared_ranges = common.threshold_range_present.to_numpy()
    summary = {
        "seeds": SEEDS,
        "source_tables": paths,
        "union_token_ids": len(union),
        "intersection_token_ids": len(intersection),
        "intersection_over_union_pct": 100 * len(intersection) / len(union),
        "tokens_in_exactly_one_seed": int((membership.num_seeds_present == 1).sum()),
        "tokens_in_exactly_two_seeds": int((membership.num_seeds_present == 2).sum()),
        "tokens_in_all_three_seeds": int((membership.num_seeds_present == 3).sum()),
        "all_three_threshold_exact_agreement_pct": 100 * float(np.mean(shared_ranges == 0)),
        "all_three_threshold_range_le_0.01_pct": 100 * float(np.mean(shared_ranges <= 0.01 + 1e-12)),
        "all_three_threshold_range_le_0.05_pct": 100 * float(np.mean(shared_ranges <= 0.05 + 1e-12)),
        "all_three_mean_threshold_range": float(np.mean(shared_ranges)),
        "all_three_median_threshold_range": float(np.median(shared_ranges)),
        "all_three_p90_threshold_range": float(np.quantile(shared_ranges, 0.9)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    markdown = [
        f"# P1.2 Threshold sensitivity across {args.size}-trace subsets",
        "",
        f"Three independently sampled {args.size}-of-5000 Math Standard-trace subsets were calibrated with seeds 2027, 2028, and 2029.",
        "",
        "## Table size and overlap",
        "",
        "| Seed | Valid thresholds | Eligible tokens | Valid / eligible | Coverage of union |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed.itertuples():
        markdown.append(
            f"| {row.seed} | {row.valid_token_thresholds:,} | {row.eligible_token_ids:,} | {row.valid_among_eligible_pct:.2f}% | {row.union_coverage_pct:.2f}% |"
        )
    markdown.extend(
        [
            "",
            f"Union: **{len(union):,}** token ids; all-three intersection: **{len(intersection):,}** ({100*len(intersection)/len(union):.2f}% of union).",
            "",
            "## Pairwise threshold sensitivity",
            "",
            "| Seeds | Shared | Jaccard | Exact agreement | Within 0.05 | Mean |Δτ| | P90 |Δτ| | Spearman |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pairwise.itertuples():
        markdown.append(
            f"| {row.seed_left}/{row.seed_right} | {row.shared_tokens} | {row.jaccard:.3f} | {row.exact_threshold_agreement_pct:.2f}% | {row.within_005_pct:.2f}% | {row.mean_abs_threshold_diff:.3f} | {row.p90_abs_threshold_diff:.3f} | {row.spearman_threshold_correlation:.3f} |"
        )
    markdown.extend(
        [
            "",
            "## Main overlapping tokens",
            "",
            "The 100 highest-support tokens shared by all seeds are in `top100_common_tokens.csv`, including decoded token text, per-seed thresholds, and support. Full membership and seed-unique token lists are provided as CSV files.",
            "",
            "## Interpretation",
            "",
            "This analysis measures sampling sensitivity of lookup-table support and selected thresholds. Downstream accuracy/NFE sensitivity is evaluated separately on GSM8K, MATH500, and HumanEval.",
        ]
    )
    (args.output_dir / "P12_THRESHOLD_SEED_SENSITIVITY.md").write_text("\n".join(markdown) + "\n")

    plt = configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    axes[0, 0].bar(pattern_counts.membership, pattern_counts.token_types, color="#56B4E9")
    axes[0, 0].set_title("(a) Token-table membership patterns")
    axes[0, 0].set_xlabel("Seeds 2027 / 2028 / 2029 (1 = present)")
    axes[0, 0].set_ylabel("Token types")
    axes[0, 0].grid(True, axis="y")

    axes[0, 1].bar([str(x) for x in per_seed.seed], per_seed.valid_token_thresholds, color="#009E73")
    axes[0, 1].set_title("(b) Valid token thresholds")
    axes[0, 1].set_xlabel("Subset seed")
    axes[0, 1].set_ylabel("Token types")
    axes[0, 1].grid(True, axis="y")

    axes[1, 0].hist(shared_ranges, bins=25, color="#E69F00", alpha=0.85)
    axes[1, 0].set_title("(c) Threshold range for all-seed shared tokens")
    axes[1, 0].set_xlabel("max(τ) − min(τ)")
    axes[1, 0].set_ylabel("Token types")

    jaccard = np.eye(len(SEEDS))
    for row in pairwise.itertuples():
        i, j = SEEDS.index(row.seed_left), SEEDS.index(row.seed_right)
        jaccard[i, j] = jaccard[j, i] = row.jaccard
    image = axes[1, 1].imshow(jaccard, vmin=0, vmax=1, cmap="Blues")
    axes[1, 1].set_xticks(range(len(SEEDS)), SEEDS)
    axes[1, 1].set_yticks(range(len(SEEDS)), SEEDS)
    axes[1, 1].set_title("(d) Pairwise token-set Jaccard")
    for i in range(len(SEEDS)):
        for j in range(len(SEEDS)):
            axes[1, 1].text(j, i, f"{jaccard[i,j]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046)
    for suffix in ["png", "pdf"]:
        fig.savefig(args.output_dir / f"figure_p12_threshold_seed_sensitivity.{suffix}", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Wrote seed-sensitivity artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
