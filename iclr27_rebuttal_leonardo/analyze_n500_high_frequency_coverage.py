#!/usr/bin/env python3
"""Analyze whether the 500-trace table captures high-frequency token mass."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tc_apd.calibration import iter_final_consistency_records


def load_table(path: Path) -> dict[int, float]:
    return {int(key): float(value) for key, value in json.loads(path.read_text()).items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--table-500", required=True, type=Path)
    parser.add_argument("--table-5000", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table500 = load_table(args.table_500)
    table5000 = load_table(args.table_5000)
    ids500, ids5000 = set(table500), set(table5000)

    counts: Counter[int] = Counter()
    total_records = 0
    for record in iter_final_consistency_records(args.log, max_traces_per_log=5000):
        counts[int(record["token_id"])] += 1
        total_records += 1
    ranked = sorted(counts, key=lambda token: (-counts[token], token))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    top_rows = []
    for rank, token_id in enumerate(ranked[:500], start=1):
        top_rows.append(
            {
                "frequency_rank": rank,
                "token_id": token_id,
                "token_piece": tokenizer.convert_ids_to_tokens(token_id),
                "decoded_text": tokenizer.decode([token_id], skip_special_tokens=False),
                "candidate_count_5000_traces": counts[token_id],
                "candidate_mass_pct": 100 * counts[token_id] / total_records,
                "in_n500_table": token_id in ids500,
                "threshold_n500": table500.get(token_id, ""),
                "in_n5000_table": token_id in ids5000,
                "threshold_n5000": table5000.get(token_id, ""),
            }
        )
    write_csv(args.output_dir / "top500_candidate_tokens.csv", top_rows)

    topk_rows = []
    for k in [10, 25, 50, 100, 200, 500, 1000, 2000]:
        top = set(ranked[: min(k, len(ranked))])
        top_mass = sum(counts[token] for token in top)
        target = top & ids5000
        target_mass = sum(counts[token] for token in target)
        recovered = target & ids500
        topk_rows.append(
            {
                "top_k": min(k, len(ranked)),
                "top_k_candidate_mass_pct": 100 * top_mass / total_records,
                "n500_tokens_in_top_k": len(top & ids500),
                "n500_top_k_type_coverage_pct": 100 * len(top & ids500) / len(top),
                "n5000_tokens_in_top_k": len(target),
                "n5000_top_k_type_coverage_pct": 100 * len(target) / len(top),
                "n500_recall_of_n5000_valid_types_pct": 100 * len(recovered) / len(target) if target else 0,
                "n500_recall_of_n5000_valid_mass_pct": 100 * sum(counts[token] for token in recovered) / target_mass if target_mass else 0,
            }
        )
    write_csv(args.output_dir / "topk_high_frequency_coverage.csv", topk_rows)

    def mass(token_ids: set[int]) -> int:
        return sum(counts[token] for token in token_ids)

    shared = ids500 & ids5000
    union = ids500 | ids5000
    values500 = np.asarray([table500[token] for token in sorted(shared)])
    values5000 = np.asarray([table5000[token] for token in sorted(shared)])
    delta = np.abs(values500 - values5000)
    mass500 = mass(ids500)
    mass5000 = mass(ids5000)
    shared_mass = mass(shared)
    summary = {
        "candidate_records_5000_traces": total_records,
        "observed_candidate_token_types": len(counts),
        "n500_valid_thresholds": len(ids500),
        "n5000_valid_thresholds": len(ids5000),
        "shared_valid_thresholds": len(shared),
        "union_valid_thresholds": len(union),
        "n500_type_recall_of_n5000_pct": 100 * len(shared) / len(ids5000),
        "n500_table_retained_in_n5000_pct": 100 * len(shared) / len(ids500),
        "table_jaccard": len(shared) / len(union),
        "n500_candidate_mass_coverage_pct": 100 * mass500 / total_records,
        "n5000_candidate_mass_coverage_pct": 100 * mass5000 / total_records,
        "shared_candidate_mass_coverage_pct": 100 * shared_mass / total_records,
        "n500_recovered_fraction_of_n5000_candidate_mass_pct": 100 * shared_mass / mass5000,
        "n500_only_candidate_mass_pct": 100 * mass(ids500 - ids5000) / total_records,
        "n5000_only_candidate_mass_pct": 100 * mass(ids5000 - ids500) / total_records,
        "shared_exact_threshold_agreement_pct": 100 * float(np.mean(delta == 0)),
        "shared_threshold_within_0.05_pct": 100 * float(np.mean(delta <= 0.05 + 1e-12)),
        "shared_mean_abs_threshold_diff": float(np.mean(delta)),
        "shared_median_abs_threshold_diff": float(np.median(delta)),
        "shared_p90_abs_threshold_diff": float(np.quantile(delta, 0.9)),
        "shared_spearman_threshold_correlation": float(spearmanr(values500, values5000).statistic),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# High-frequency token coverage: 500 vs 5,000 Math traces",
        "",
        "Candidate frequency is measured over all intermediate top-1 token records in the same 5,000 Standard-decoding Math traces used for calibration.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Valid thresholds (500 / 5,000) | {len(ids500):,} / {len(ids5000):,} |",
        f"| Shared valid token ids | {len(shared):,} |",
        f"| 500 type recall of 5,000 table | {summary['n500_type_recall_of_n5000_pct']:.2f}% |",
        f"| Candidate mass covered by 500 table | {summary['n500_candidate_mass_coverage_pct']:.2f}% |",
        f"| Candidate mass covered by 5,000 table | {summary['n5000_candidate_mass_coverage_pct']:.2f}% |",
        f"| 5,000-table candidate mass recovered by shared 500 tokens | {summary['n500_recovered_fraction_of_n5000_candidate_mass_pct']:.2f}% |",
        "",
        "## Frequency-ranked coverage",
        "",
        "| Top K token types | Candidate mass in Top K | 500 covered types | 5,000 covered types | 500 recall of 5,000 valid types | 500 recall of 5,000 valid mass |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in topk_rows:
        lines.append(
            f"| {row['top_k']:,} | {row['top_k_candidate_mass_pct']:.2f}% | "
            f"{row['n500_tokens_in_top_k']:,} | {row['n5000_tokens_in_top_k']:,} | "
            f"{row['n500_recall_of_n5000_valid_types_pct']:.2f}% | "
            f"{row['n500_recall_of_n5000_valid_mass_pct']:.2f}% |"
        )
    lines += [
        "",
        "## Shared-threshold stability",
        "",
        f"Among {len(shared):,} shared token ids, mean absolute threshold difference is {summary['shared_mean_abs_threshold_diff']:.3f}, median is {summary['shared_median_abs_threshold_diff']:.3f}, and {summary['shared_threshold_within_0.05_pct']:.2f}% are within 0.05.",
        "",
        "The 500-trace table should therefore be described as a compact high-frequency approximation, not as an identical reconstruction of the 5,000-trace table.",
    ]
    (args.output_dir / "N500_HIGH_FREQUENCY_COVERAGE.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
