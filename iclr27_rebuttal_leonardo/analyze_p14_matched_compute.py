#!/usr/bin/env python3
"""Paired Fixed(0.77) vs TC-APD(Math-5000, max 0.9) comparison."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np


TASKS = {
    "gsm8k": {
        "label": "GSM8K",
        "filter": "flexible-extract",
        "metric": "exact_match",
        "tc_result": "gsm8k_tc_apd_math_n5000_seed2027",
        "tc_log": "logs/gsm8k/gsm8k_dynamic_threshold_from_math_n5000_seed2027_c99.5_mincount200_minaccepted100_len256_block32_maxthr0.90_minthr0.01.log",
    },
    "math500": {
        "label": "MATH500",
        "filter": "none",
        "metric": "math_verify",
        "tc_result": "math500_tc_apd_math_n5000_seed2027",
        "tc_log": "logs/math500/math500_dynamic_threshold_from_math_n5000_seed2027_c99.5_mincount200_minaccepted100_len256_block32_maxthr0.90_minthr0.01.log",
    },
    "humaneval": {
        "label": "HumanEval",
        "filter": "create_test",
        "metric": "pass@1",
        "tc_result": "humaneval_tc_apd_math_n5000_seed2027",
        "tc_log": "logs/humaneval/humaneval_dynamic_threshold_from_math_n5000_seed2027_c99.5_mincount200_minaccepted100_len256_block32_maxthr0.90_minthr0.01.log",
    },
}


def locate_samples(root: Path) -> Path:
    matches = sorted(glob.glob(str(root / "**" / "samples_*.jsonl"), recursive=True))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one samples file under {root}, found {matches}")
    return Path(matches[0])


def load_metric(path: Path, filter_name: str, metric: str) -> dict[int, float]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("filter") == filter_name:
                result[int(row["doc_id"])] = float(row[metric])
    if not result:
        raise RuntimeError(f"No {filter_name}/{metric} records in {path}")
    return result


def load_efficiency(path: Path, expected: int) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text(encoding="utf-8", errors="replace")
    nfe = np.asarray([int(x) for x in re.findall(r"^Model forward calls:\s*(\d+)\s*$", text, re.M)], dtype=float)
    seconds = np.asarray([float(x) for x in re.findall(r"^Decode time:\s*([0-9.]+)s\s*$", text, re.M)], dtype=float)
    if len(nfe) != expected or len(seconds) != expected:
        raise RuntimeError(f"Expected {expected} decode records in {path}; got NFE={len(nfe)}, time={len(seconds)}")
    return nfe, seconds


def bootstrap_difference(fixed: np.ndarray, tc: np.ndarray, *, repeats: int, rng: np.random.Generator) -> tuple[float, float, float]:
    difference = tc - fixed
    n = len(difference)
    # Chunk resampling to bound peak memory for the 1,319-example task.
    estimates = np.empty(repeats, dtype=float)
    chunk = 1000
    for start in range(0, repeats, chunk):
        stop = min(repeats, start + chunk)
        indices = rng.integers(0, n, size=(stop - start, n))
        estimates[start:stop] = difference[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(difference.mean()), float(low), float(high)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-root", required=True, type=Path)
    parser.add_argument("--tc-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary_rows: list[dict] = []
    transition_rows: list[dict] = []
    per_sample_rows: list[dict] = []

    for benchmark, config in TASKS.items():
        fixed_samples = locate_samples(args.fixed_root / "results" / f"{benchmark}_fixed_apd_thr0.77")
        tc_samples = locate_samples(args.tc_root / "results" / config["tc_result"])
        fixed_metric = load_metric(fixed_samples, config["filter"], config["metric"])
        tc_metric = load_metric(tc_samples, config["filter"], config["metric"])
        doc_ids = sorted(set(fixed_metric) & set(tc_metric))
        if len(doc_ids) != len(fixed_metric) or len(doc_ids) != len(tc_metric):
            raise RuntimeError(f"Metric doc_id mismatch for {benchmark}")
        fixed_acc = np.asarray([fixed_metric[i] for i in doc_ids])
        tc_acc = np.asarray([tc_metric[i] for i in doc_ids])
        fixed_nfe, fixed_seconds = load_efficiency(
            args.fixed_root / "logs" / benchmark / f"{benchmark}_fixed_threshold_thr0.77_len256_block32.log",
            len(doc_ids),
        )
        tc_nfe, tc_seconds = load_efficiency(args.tc_root / config["tc_log"], len(doc_ids))

        metrics = {
            "accuracy": (fixed_acc, tc_acc),
            "nfe": (fixed_nfe, tc_nfe),
            "seconds_per_item": (fixed_seconds, tc_seconds),
        }
        row = {
            "benchmark": config["label"],
            "num_samples": len(doc_ids),
        }
        for name, (fixed, tc) in metrics.items():
            diff, low, high = bootstrap_difference(fixed, tc, repeats=args.bootstrap_repeats, rng=rng)
            row[f"fixed_{name}"] = float(fixed.mean())
            row[f"tc_apd_{name}"] = float(tc.mean())
            row[f"tc_minus_fixed_{name}"] = diff
            row[f"tc_minus_fixed_{name}_ci95_low"] = low
            row[f"tc_minus_fixed_{name}_ci95_high"] = high
        row["nfe_relative_gap_pct"] = 100 * abs(row["tc_apd_nfe"] - row["fixed_nfe"]) / ((row["tc_apd_nfe"] + row["fixed_nfe"]) / 2)
        row["wallclock_relative_gap_pct"] = 100 * abs(row["tc_apd_seconds_per_item"] - row["fixed_seconds_per_item"]) / ((row["tc_apd_seconds_per_item"] + row["fixed_seconds_per_item"]) / 2)
        summary_rows.append(row)

        fixed_binary = fixed_acc.astype(bool)
        tc_binary = tc_acc.astype(bool)
        transition_rows.append({
            "benchmark": config["label"],
            "fixed_correct_tc_correct": int(np.sum(fixed_binary & tc_binary)),
            "fixed_correct_tc_incorrect": int(np.sum(fixed_binary & ~tc_binary)),
            "fixed_incorrect_tc_correct": int(np.sum(~fixed_binary & tc_binary)),
            "fixed_incorrect_tc_incorrect": int(np.sum(~fixed_binary & ~tc_binary)),
        })
        for index, doc_id in enumerate(doc_ids):
            per_sample_rows.append({
                "benchmark": config["label"],
                "doc_id": doc_id,
                "fixed_correct": fixed_acc[index],
                "tc_apd_correct": tc_acc[index],
                "fixed_nfe": fixed_nfe[index],
                "tc_apd_nfe": tc_nfe[index],
                "fixed_seconds": fixed_seconds[index],
                "tc_apd_seconds": tc_seconds[index],
            })

    write_csv(args.output_dir / "paired_bootstrap_summary.csv", summary_rows)
    write_csv(args.output_dir / "paired_accuracy_transitions.csv", transition_rows)
    write_csv(args.output_dir / "paired_per_sample.csv", per_sample_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"bootstrap_repeats": args.bootstrap_repeats, "seed": args.seed, "summary": summary_rows, "transitions": transition_rows}, handle, indent=2)

    lines = [
        "# P1.4 Matched-compute paired comparison",
        "",
        "Fixed APD uses threshold 0.77; TC-APD uses the Math-5000 empirical table with maximum/default threshold 0.90. All differences and confidence intervals are paired over identical examples and defined as **TC-APD − Fixed**.",
        "",
        "| Benchmark | Fixed acc. | TC acc. | Accuracy Δ (95% paired CI) | Fixed NFE | TC NFE | NFE Δ (95% CI) | Fixed s/item | TC s/item | Time Δ (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['benchmark']} | {100*row['fixed_accuracy']:.2f}% | {100*row['tc_apd_accuracy']:.2f}% | "
            f"{100*row['tc_minus_fixed_accuracy']:+.2f} pp [{100*row['tc_minus_fixed_accuracy_ci95_low']:+.2f}, {100*row['tc_minus_fixed_accuracy_ci95_high']:+.2f}] | "
            f"{row['fixed_nfe']:.2f} | {row['tc_apd_nfe']:.2f} | {row['tc_minus_fixed_nfe']:+.2f} [{row['tc_minus_fixed_nfe_ci95_low']:+.2f}, {row['tc_minus_fixed_nfe_ci95_high']:+.2f}] | "
            f"{row['fixed_seconds_per_item']:.3f} | {row['tc_apd_seconds_per_item']:.3f} | {row['tc_minus_fixed_seconds_per_item']:+.3f} [{row['tc_minus_fixed_seconds_per_item_ci95_low']:+.3f}, {row['tc_minus_fixed_seconds_per_item_ci95_high']:+.3f}] |"
        )
    lines += [
        "",
        f"Bootstrap uses {args.bootstrap_repeats:,} paired resamples with seed {args.seed}. NFE and time relative gaps are retained in `paired_bootstrap_summary.csv` to assess whether the selected operating points are sufficiently matched.",
        "",
    ]
    (args.output_dir / "P14_MATCHED_COMPUTE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
