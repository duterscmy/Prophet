#!/usr/bin/env python3
"""P1.2 calibration-subset sampling robustness across sizes 100-5000.

For every calibration size the three independently sampled subsets
(seeds 2027 / 2028 / 2029) are evaluated on GSM8K, MATH500 and HumanEval.
The script joins lm-eval ``results_*.json`` accuracy with the decoder
statistics printed into the matching inference log, then reports per-seed
rows plus a ``Mean +- Std`` row using the sample standard deviation
(``ddof=1``).

Definitions, kept identical to the P0.1 calibration-size tables:

* ``NFE`` and ``s/it`` are means over per-example decoder statistics.
* ``SpeedUp`` is the wall-clock ratio ``Standard s/it / setting s/it``
  computed per task; the ``Avg`` column averages the three per-task
  ratios.  The NFE ratio is reported separately and never mixed in.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUN_ROOT = Path("/leonardo_scratch/large/userexternal/mcao0000/prophet-runs")
SIZE_ABLATION = RUN_ROOT / "rebuttal-size-ablation"

SIZES = [100, 200, 500, 1000, 2000, 5000]
SEEDS = [2027, 2028, 2029]
BENCHMARKS = ["gsm8k", "math500", "humaneval"]
DISPLAY = {"gsm8k": "GSM8K", "math500": "MATH500", "humaneval": "HumanEval"}

EXTRA_SEED_ROOT = {
    100: RUN_ROOT / "rebuttal-seed-sensitivity",
    500: RUN_ROOT / "rebuttal-seed-sensitivity-n500",
    200: RUN_ROOT / "rebuttal-seed-sensitivity-multisize",
    1000: RUN_ROOT / "rebuttal-seed-sensitivity-multisize",
    2000: RUN_ROOT / "rebuttal-seed-sensitivity-multisize",
    5000: RUN_ROOT / "rebuttal-seed-sensitivity-multisize",
}

METRIC_PREFIX = {
    "gsm8k": "exact_match,flexible-extract",
    "math500": "math_verify,",
    "humaneval": "pass@1,",
}

TC_LOG_TEMPLATE = (
    "{benchmark}_dynamic_threshold_from_math_n{size}_seed{seed}_"
    "c99.5_mincount200_minaccepted100_len256_block32_maxthr0.90_minthr0.01.log"
)
THRESHOLD_STEM = (
    "math_n{size}_seed{seed}_token_threshold_grid_p99.5_mincount200_minaccepted100"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("iclr27_rebuttal_leonardo/analysis/multisize_seed_sensitivity"),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="report incomplete settings instead of failing (for in-flight arrays)",
    )
    return parser.parse_args()


def run_paths(benchmark: str, size: int, seed: int) -> tuple[Path, Path, Path]:
    """Return (result_dir, inference_log, threshold_table) for one run."""
    log_name = TC_LOG_TEMPLATE.format(benchmark=benchmark, size=size, seed=seed)
    stem = THRESHOLD_STEM.format(size=size, seed=seed)
    if seed == 2027:
        return (
            SIZE_ABLATION / "results" / f"{benchmark}_tc_apd_math_n{size}_seed2027",
            SIZE_ABLATION / "logs" / benchmark / log_name,
            SIZE_ABLATION / "thresholds" / f"{stem}.json",
        )
    root = EXTRA_SEED_ROOT[size]
    return (
        root / "results" / f"{benchmark}_tc_apd_math_n{size}_seed{seed}",
        root / "logs" / benchmark / log_name,
        root / "thresholds" / f"seed{seed}" / f"{stem}.json",
    )


def baseline_paths(benchmark: str, setting: str) -> tuple[Path, Path]:
    if setting == "standard":
        return (
            SIZE_ABLATION / "results" / f"{benchmark}_standard",
            SIZE_ABLATION / "logs" / benchmark / f"{benchmark}_standard_len256_block32.log",
        )
    return (
        SIZE_ABLATION / "results" / f"{benchmark}_fixed_apd_thr0.90",
        SIZE_ABLATION
        / "logs"
        / benchmark
        / f"{benchmark}_fixed_threshold_thr0.90_len256_block32.log",
    )


def latest_result(directory: Path) -> Path | None:
    matches = sorted(directory.rglob("results_*.json"))
    return matches[-1] if matches else None


def extract_accuracy(benchmark: str, payload: dict[str, Any]) -> tuple[float, int]:
    task_name, result = next(iter(payload["results"].items()))
    prefix = METRIC_PREFIX[benchmark]
    value_key = next(
        key for key in result if key.startswith(prefix) and "stderr" not in key
    )
    examples = payload.get("n-samples", {}).get(task_name, {}).get("effective", 0)
    return 100.0 * float(result[value_key]), int(examples)


def parse_decode_log(path: Path) -> tuple[float, float, int]:
    text = path.read_text(errors="replace")
    nfes = [int(x) for x in re.findall(r"^Model forward calls:\s*(\d+)\s*$", text, re.M)]
    times = [float(x) for x in re.findall(r"^Decode time:\s*([0-9.]+)s\s*$", text, re.M)]
    if not nfes or len(nfes) != len(times):
        raise ValueError(
            f"Incomplete decode statistics in {path.name}: "
            f"{len(nfes)} NFE and {len(times)} decode times"
        )
    return float(np.mean(nfes)), float(np.mean(times)), len(nfes)


def load_run(
    benchmark: str, result_dir: Path, log_path: Path, allow_missing: bool
) -> dict[str, Any] | None:
    payload_path = latest_result(result_dir)
    if payload_path is None or not log_path.is_file():
        if not allow_missing:
            raise FileNotFoundError(
                f"Missing run for {benchmark}: results={result_dir}, log={log_path}"
            )
        return None
    try:
        mean_nfe, mean_seconds, logged = parse_decode_log(log_path)
    except ValueError as error:
        if not allow_missing:
            raise
        print(f"[incomplete] {error}")
        return None
    accuracy, examples = extract_accuracy(
        benchmark, json.loads(payload_path.read_text())
    )
    return {
        "accuracy_pct": accuracy,
        "mean_nfe": mean_nfe,
        "seconds_per_item": mean_seconds,
        "evaluated_examples": examples,
        "logged_generations": logged,
        "result_file": str(payload_path),
        "log_file": str(log_path),
    }


def mean_std(values: list[float]) -> tuple[float, float, int]:
    """Mean, sample standard deviation (ddof=1), and the number of seeds used."""
    array = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if array.size == 0:
        return math.nan, math.nan, 0
    std = float(np.std(array, ddof=1)) if array.size > 1 else math.nan
    return float(array.mean()), std, int(array.size)


def macro(values: list[float]) -> float:
    present = [v for v in values if not math.isnan(v)]
    return float(np.mean(present)) if len(present) == len(values) else math.nan


def threshold_values(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text())
    table = payload["thresholds"] if "thresholds" in payload else payload
    return {int(k): float(v) for k, v in table.items()}


def fmt(value: float, digits: int = 2) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def pm(mean: float, std: float, digits: int = 2, count: int = 3) -> str:
    """Render ``mean ± std``; a single seed carries no standard deviation."""
    if math.isnan(mean):
        return "n/a"
    if count < 2:
        return f"{mean:.{digits}f} (1 seed)"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baselines: dict[str, dict[str, dict[str, Any]]] = {}
    for setting in ["standard", "fixed"]:
        baselines[setting] = {}
        for benchmark in BENCHMARKS:
            result_dir, log_path = baseline_paths(benchmark, setting)
            run = load_run(benchmark, result_dir, log_path, allow_missing=False)
            assert run is not None
            baselines[setting][benchmark] = run

    per_run: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    runs: dict[tuple[int, int, str], dict[str, Any]] = {}
    for size in SIZES:
        for seed in SEEDS:
            for benchmark in BENCHMARKS:
                result_dir, log_path, table_path = run_paths(benchmark, size, seed)
                run = load_run(benchmark, result_dir, log_path, args.allow_missing)
                if run is None:
                    missing.append(
                        {
                            "calibration_size": size,
                            "seed": seed,
                            "benchmark": benchmark,
                            "expected_result_dir": str(result_dir),
                            "expected_log": str(log_path),
                        }
                    )
                    continue
                standard = baselines["standard"][benchmark]
                run.update(
                    {
                        "calibration_size": size,
                        "seed": seed,
                        "benchmark": benchmark,
                        "task": DISPLAY[benchmark],
                        "nfe_speedup_vs_standard": standard["mean_nfe"] / run["mean_nfe"],
                        "wallclock_speedup_vs_standard": (
                            standard["seconds_per_item"] / run["seconds_per_item"]
                        ),
                        "threshold_table": str(table_path),
                        "threshold_table_present": int(table_path.is_file()),
                    }
                )
                runs[(size, seed, benchmark)] = run
                per_run.append(run)

    per_run_fields = [
        "calibration_size",
        "seed",
        "benchmark",
        "task",
        "accuracy_pct",
        "mean_nfe",
        "seconds_per_item",
        "nfe_speedup_vs_standard",
        "wallclock_speedup_vs_standard",
        "evaluated_examples",
        "logged_generations",
        "threshold_table_present",
        "result_file",
        "log_file",
        "threshold_table",
    ]
    with (args.output_dir / "multiseed_per_run.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=per_run_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            sorted(per_run, key=lambda r: (r["calibration_size"], r["seed"], r["benchmark"]))
        )

    if missing:
        with (args.output_dir / "missing_runs.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(missing[0]))
            writer.writeheader()
            writer.writerows(missing)

    # ---- the requested wide table -------------------------------------
    columns = [
        f"{DISPLAY[benchmark]} {field}"
        for benchmark in BENCHMARKS
        for field in ["Acc", "NFE", "s/it", "SpeedUp"]
    ] + ["Avg Acc", "Avg NFE", "Avg s/it", "Avg SpeedUp"]

    def baseline_row(setting: str, label: str) -> dict[str, Any]:
        row: dict[str, Any] = {"setting": label, "seeds_used": ""}
        accs, nfes, secs, ups = [], [], [], []
        for benchmark in BENCHMARKS:
            run = baselines[setting][benchmark]
            standard = baselines["standard"][benchmark]
            speedup = standard["seconds_per_item"] / run["seconds_per_item"]
            row[f"{DISPLAY[benchmark]} Acc"] = fmt(run["accuracy_pct"])
            row[f"{DISPLAY[benchmark]} NFE"] = fmt(run["mean_nfe"])
            row[f"{DISPLAY[benchmark]} s/it"] = fmt(run["seconds_per_item"], 3)
            row[f"{DISPLAY[benchmark]} SpeedUp"] = fmt(speedup)
            accs.append(run["accuracy_pct"])
            nfes.append(run["mean_nfe"])
            secs.append(run["seconds_per_item"])
            ups.append(speedup)
        row["Avg Acc"] = fmt(macro(accs))
        row["Avg NFE"] = fmt(macro(nfes))
        row["Avg s/it"] = fmt(macro(secs), 3)
        row["Avg SpeedUp"] = fmt(macro(ups))
        return row

    table_rows = [
        baseline_row("standard", "Standard"),
        baseline_row("fixed", "Fixed threshold 0.90"),
    ]

    stats_rows: list[dict[str, Any]] = []
    for size in SIZES:
        per_seed_macro: dict[int, dict[str, float]] = {}
        for seed in SEEDS:
            row: dict[str, Any] = {"setting": f"Math-{size} seed {seed}", "seeds_used": "1"}
            accs, nfes, secs, ups = [], [], [], []
            for benchmark in BENCHMARKS:
                run = runs.get((size, seed, benchmark))
                label = DISPLAY[benchmark]
                if run is None:
                    row[f"{label} Acc"] = row[f"{label} NFE"] = "n/a"
                    row[f"{label} s/it"] = row[f"{label} SpeedUp"] = "n/a"
                    accs.append(math.nan)
                    nfes.append(math.nan)
                    secs.append(math.nan)
                    ups.append(math.nan)
                    continue
                row[f"{label} Acc"] = fmt(run["accuracy_pct"])
                row[f"{label} NFE"] = fmt(run["mean_nfe"])
                row[f"{label} s/it"] = fmt(run["seconds_per_item"], 3)
                row[f"{label} SpeedUp"] = fmt(run["wallclock_speedup_vs_standard"])
                accs.append(run["accuracy_pct"])
                nfes.append(run["mean_nfe"])
                secs.append(run["seconds_per_item"])
                ups.append(run["wallclock_speedup_vs_standard"])
            row["Avg Acc"] = fmt(macro(accs))
            row["Avg NFE"] = fmt(macro(nfes))
            row["Avg s/it"] = fmt(macro(secs), 3)
            row["Avg SpeedUp"] = fmt(macro(ups))
            per_seed_macro[seed] = {
                "acc": macro(accs),
                "nfe": macro(nfes),
                "sec": macro(secs),
                "up": macro(ups),
            }
            table_rows.append(row)

        summary_row: dict[str, Any] = {"setting": f"Math-{size} Mean ± Std"}
        stats_entry: dict[str, Any] = {"calibration_size": size}
        used = 0
        for benchmark in BENCHMARKS:
            label = DISPLAY[benchmark]
            present = [
                runs[(size, seed, benchmark)]
                for seed in SEEDS
                if (size, seed, benchmark) in runs
            ]
            used = max(used, len(present))
            for field, key, digits in [
                ("Acc", "accuracy_pct", 2),
                ("NFE", "mean_nfe", 2),
                ("s/it", "seconds_per_item", 3),
                ("SpeedUp", "wallclock_speedup_vs_standard", 2),
            ]:
                mean, std, count = mean_std([run[key] for run in present])
                summary_row[f"{label} {field}"] = pm(mean, std, digits, count)
                stats_entry[f"{benchmark}_{key}_mean"] = mean
                stats_entry[f"{benchmark}_{key}_std"] = std
                stats_entry[f"{benchmark}_{key}_seeds"] = count
        for field, key, digits in [
            ("Acc", "acc", 2),
            ("NFE", "nfe", 2),
            ("s/it", "sec", 3),
            ("SpeedUp", "up", 2),
        ]:
            mean, std, count = mean_std(
                [per_seed_macro[seed][key] for seed in SEEDS]
            )
            summary_row[f"Avg {field}"] = pm(mean, std, digits, count)
            stats_entry[f"macro_{key}_mean"] = mean
            stats_entry[f"macro_{key}_std"] = std
            stats_entry[f"macro_{key}_seeds"] = count
        summary_row["seeds_used"] = str(used)
        stats_entry["seeds_used"] = used
        table_rows.append(summary_row)
        stats_rows.append(stats_entry)

    with (args.output_dir / "multiseed_full_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["setting"] + columns + ["seeds_used"]
        )
        writer.writeheader()
        writer.writerows(table_rows)

    with (args.output_dir / "multiseed_mean_std_statistics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats_rows[0]))
        writer.writeheader()
        writer.writerows(stats_rows)

    # ---- threshold-table stability ------------------------------------
    stability_rows: list[dict[str, Any]] = []
    for size in SIZES:
        tables: dict[int, dict[int, float]] = {}
        for seed in SEEDS:
            _, _, table_path = run_paths(BENCHMARKS[0], size, seed)
            if table_path.is_file():
                tables[seed] = threshold_values(table_path)
        if len(tables) < 2:
            continue
        keys = {seed: set(table) for seed, table in tables.items()}
        union = set().union(*keys.values())
        intersection = set.intersection(*keys.values())
        pair_abs: list[float] = []
        pair_within_005: list[float] = []
        for left, right in combinations(sorted(tables), 2):
            common = keys[left] & keys[right]
            if not common:
                continue
            diffs = np.array(
                [abs(tables[left][token] - tables[right][token]) for token in sorted(common)]
            )
            pair_abs.append(float(diffs.mean()))
            pair_within_005.append(float((diffs <= 0.05).mean()))
        row = {
            "calibration_size": size,
            "seeds_available": len(tables),
            "union_tokens": len(union),
            "intersection_tokens": len(intersection),
            "jaccard_intersection_over_union": len(intersection) / len(union)
            if union
            else math.nan,
            "pairwise_mean_abs_threshold_diff": float(np.mean(pair_abs))
            if pair_abs
            else math.nan,
            "pairwise_within_0.05_fraction": float(np.mean(pair_within_005))
            if pair_within_005
            else math.nan,
        }
        for seed in SEEDS:
            row[f"valid_thresholds_seed{seed}"] = len(tables[seed]) if seed in tables else ""
        row["subsets_identical"] = int(
            len(tables) == len(SEEDS)
            and len(intersection) == len(union)
            and bool(pair_abs)
            and max(pair_abs) == 0.0
        )
        stability_rows.append(row)

    with (args.output_dir / "multiseed_threshold_stability.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stability_rows[0]))
        writer.writeheader()
        writer.writerows(stability_rows)

    summary = {
        "sizes": SIZES,
        "seeds": SEEDS,
        "benchmarks": BENCHMARKS,
        "std_convention": "sample standard deviation, ddof=1",
        "speedup_convention": (
            "per-task wall-clock ratio Standard (s/it) / setting (s/it); "
            "Avg SpeedUp is the mean of the three per-task ratios"
        ),
        "completed_runs": len(per_run),
        "missing_runs": missing,
        "mean_std_statistics": stats_rows,
        "threshold_stability": stability_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )

    lines = [
        "# P1.2 Calibration-subset sampling robustness (sizes 100-5000, 3 seeds)",
        "",
        "Each calibration size is re-sampled with three independent subset seeds",
        "(2027 / 2028 / 2029) from the same pool of Math Standard-decoding traces,",
        "recalibrated, and re-evaluated on all three main benchmarks at",
        "`gen_length=256, steps=256, block_length=32, max/default threshold=0.90,",
        "min threshold=0.01`.",
        "",
        "`±` is the sample standard deviation across the three subset seeds",
        "(`ddof=1`). `SpeedUp` is the per-task wall-clock ratio",
        "`Standard (s/it) / setting (s/it)`; `Avg SpeedUp` averages the three",
        "per-task ratios. NFE ratios are kept separate in",
        "`multiseed_per_run.csv` and are never mixed into this column.",
        "",
        "| Setting | "
        + " | ".join(columns)
        + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for row in table_rows:
        lines.append(
            f"| {row['setting']} | "
            + " | ".join(str(row[column]) for column in columns)
            + " |"
        )
    lines += [
        "",
        "## Threshold-table stability across subset seeds",
        "",
        "| Size | Valid thresholds (2027 / 2028 / 2029) | Union | Intersection | "
        "Jaccard | Pairwise mean \\|Δτ\\| | Pairwise within 0.05 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in stability_rows:
        counts = " / ".join(
            str(row.get(f"valid_thresholds_seed{seed}", "")) for seed in SEEDS
        )
        lines.append(
            f"| {row['calibration_size']} | {counts} | {row['union_tokens']} "
            f"| {row['intersection_tokens']} "
            f"| {fmt(row['jaccard_intersection_over_union'] * 100, 2)}% "
            f"| {fmt(row['pairwise_mean_abs_threshold_diff'], 3)} "
            f"| {fmt(row['pairwise_within_0.05_fraction'] * 100, 1)}% |"
        )
    saturated = [row["calibration_size"] for row in stability_rows if row["subsets_identical"]]
    if saturated:
        lines += [
            "",
            "**Saturated sizes.** At calibration size "
            + ", ".join(str(size) for size in saturated)
            + " the requested subset is as large as the available pool of Math"
            " Standard-decoding traces, so the three subset seeds select the same"
            " traces and produce byte-identical threshold tables. The rows for"
            " those sizes therefore report run-to-run wall-clock jitter under"
            " greedy decoding, not calibration-subset sampling variability, and"
            " must not be presented as evidence that larger calibration sets are"
            " insensitive to subset sampling.",
        ]
    if missing:
        lines += [
            "",
            f"## Incomplete settings ({len(missing)})",
            "",
            "The following size/seed/benchmark runs had no aggregate result or no",
            "complete decoder statistics when this report was generated; they are",
            "excluded from the corresponding `Mean ± Std` entries, and the",
            "`seeds_used` column records how many seeds each row aggregates.",
            "",
        ]
        for entry in missing:
            lines.append(
                f"- Math-{entry['calibration_size']} seed {entry['seed']} "
                f"{DISPLAY[entry['benchmark']]}"
            )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Threshold membership and individual threshold values are sensitive to which",
        "calibration subset is drawn, while aggregate NFE and wall-clock efficiency are",
        "considerably more stable. Accuracy differences across subset seeds are within",
        "the range expected from benchmark sampling noise at these sample sizes and",
        "should not be read as a monotone effect of calibration size.",
        "",
        "## Files",
        "",
        "- `multiseed_full_table.csv`: the table above, including the `Mean ± Std` rows.",
        "- `multiseed_mean_std_statistics.csv`: numeric means and sample standard deviations.",
        "- `multiseed_per_run.csv`: one row per size/seed/benchmark with result and log paths.",
        "- `multiseed_threshold_stability.csv`: per-size threshold membership and value agreement.",
        "- `missing_runs.csv`: settings that were not complete at generation time (if any).",
        "- `summary.json`: machine-readable dump.",
    ]
    (args.output_dir / "P12_MULTISIZE_SEED_SENSITIVITY.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(
        f"completed_runs={len(per_run)} missing_runs={len(missing)} -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
