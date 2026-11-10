#!/usr/bin/env python3
"""Compute P2.1 offline calibration cost and break-even deployment size."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd


SIZES = [100, 200, 500, 1000, 2000, 5000]
BENCHMARK_LABELS = {
    "gsm8k": "GSM8K",
    "math500": "MATH500",
    "humaneval": "HumanEval",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-cpu-seconds", type=float, default=124.0)
    parser.add_argument("--threshold-cpus", type=int, default=8)
    parser.add_argument("--source-run-gpu-hours", type=float, default=15.622)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.metrics_csv)
    standard = frame[frame.setting == "standard"].set_index("benchmark")
    dynamic = frame[frame.setting == "dynamic"].copy()
    dynamic["calibration_size"] = dynamic.calibration_size.astype(int)

    # Same-hardware normalization: calibration uses Standard decoding on Math
    # with exactly the same len=256, steps=256, block=32 configuration.
    calibration_seconds_per_trace = float(standard.loc["math500", "mean_seconds_per_item"])
    detail_rows: list[dict] = []
    size_rows: list[dict] = []

    for size in SIZES:
        calibration_seconds = size * calibration_seconds_per_trace
        calibration_gpu_hours = calibration_seconds / 3600
        part = dynamic[dynamic.calibration_size == size].set_index("benchmark")
        break_evens = []
        row = {
            "calibration_size": size,
            "calibration_gpu_hours_a100": calibration_gpu_hours,
        }
        for benchmark, label in BENCHMARK_LABELS.items():
            standard_seconds = float(standard.loc[benchmark, "mean_seconds_per_item"])
            tc_seconds = float(part.loc[benchmark, "mean_seconds_per_item"])
            saved_seconds = standard_seconds - tc_seconds
            break_even = calibration_seconds / saved_seconds
            break_evens.append(break_even)
            row[f"{benchmark}_break_even_sequences"] = break_even
            row[f"{benchmark}_break_even_sequences_ceil"] = math.ceil(break_even)
            detail_rows.append({
                "calibration_size": size,
                "benchmark": label,
                "hardware": "1x NVIDIA A100 (Leonardo)",
                "calibration_seconds_per_trace": calibration_seconds_per_trace,
                "calibration_gpu_hours": calibration_gpu_hours,
                "standard_seconds_per_sequence": standard_seconds,
                "tc_apd_seconds_per_sequence": tc_seconds,
                "saved_seconds_per_sequence": saved_seconds,
                "wallclock_speedup": standard_seconds / tc_seconds,
                "break_even_sequences": break_even,
                "break_even_sequences_ceil": math.ceil(break_even),
            })
        row["mean_break_even_sequences"] = sum(break_evens) / len(break_evens)
        row["min_break_even_sequences"] = min(break_evens)
        row["max_break_even_sequences"] = max(break_evens)
        size_rows.append(row)

    write_csv(args.output_dir / "amortization_by_task.csv", detail_rows)
    write_csv(args.output_dir / "amortization_by_calibration_size.csv", size_rows)

    source_seconds_per_trace = args.source_run_gpu_hours * 3600 / 5000
    conservative_scale = source_seconds_per_trace / calibration_seconds_per_trace
    cpu_hours = args.threshold_cpu_seconds * args.threshold_cpus / 3600
    lines = [
        "# P2.1 Offline calibration cost and amortization",
        "",
        "## Cost model",
        "",
        "All primary costs are normalized to **one NVIDIA A100 on Leonardo**, so numerator and denominator use the same hardware and wall-clock unit. Math calibration traces use Standard decoding with `gen_length=256`, `steps=256`, and `block_length=32`. The measured MATH500 Standard cost is "
        f"**{calibration_seconds_per_trace:.3f} GPU-seconds per trace**.",
        "",
        "For benchmark \(b\) and calibration size \(m\):",
        "",
        "\\[C_{cal}(m)=m\\,t_{Math,Standard},\\qquad N_b^*(m)=\\frac{C_{cal}(m)}{t_{b,Standard}-t_{b,TC-APD(m)}}.\\]",
        "",
        "| Calibration trajectories | Calibration A100 GPU-hours | GSM8K break-even | MATH500 break-even | HumanEval break-even | Mean break-even |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in size_rows:
        lines.append(
            f"| {row['calibration_size']:,} | {row['calibration_gpu_hours_a100']:.3f} | "
            f"{row['gsm8k_break_even_sequences_ceil']:,} | "
            f"{row['math500_break_even_sequences_ceil']:,} | "
            f"{row['humaneval_break_even_sequences_ceil']:,} | "
            f"{math.ceil(row['mean_break_even_sequences']):,} |"
        )

    first = size_rows[0]
    last = size_rows[-1]
    lines += [
        "",
        "Break-even values are rounded up to the first whole deployed sequence that repays the offline GPU cost. For example, the 100-trace table costs only "
        f"{first['calibration_gpu_hours_a100']:.3f} A100 GPU-hours and amortizes after roughly "
        f"{first['min_break_even_sequences']:.0f}–{first['max_break_even_sequences']:.0f} generations across the three tasks. The 5,000-trace table costs "
        f"{last['calibration_gpu_hours_a100']:.3f} A100 GPU-hours and needs about "
        f"{last['min_break_even_sequences']:.0f}–{last['max_break_even_sequences']:.0f} generations.",
        "",
        "## Threshold-table construction overhead",
        "",
        f"Generating all six threshold tables from the saved traces took **{args.threshold_cpu_seconds:.0f} seconds on {args.threshold_cpus} CPU cores** ({cpu_hours:.3f} CPU-hours) and **0 GPU-hours**. This is negligible relative to trajectory collection and is excluded from the GPU break-even numerator.",
        "",
        "The six rows are alternative calibration budgets, not additive costs. A single 5,000-trace collection can be subsampled to produce every smaller table; its total collection cost remains the 5,000-row cost rather than the sum of all rows.",
        "If all six datasets were collected independently, their normalized costs would sum to 25.591 A100 GPU-hours, but that is not how this ablation was constructed.",
        "",
        "## Conservative source-log cross-check",
        "",
        f"The uploaded 5,000-trace source run spans approximately **{args.source_run_gpu_hours:.3f} single-GPU hours** end to end, including model setup and task-side postprocessing. That is {conservative_scale:.3f}× the Leonardo-normalized generation estimate. Because the original log does not record an unambiguous GPU model and includes non-generation overhead, it is not mixed into the primary formula. Multiplying the reported break-even counts by {conservative_scale:.3f} gives a conservative end-to-end estimate.",
        "",
        "## When recalibration is needed",
        "",
        "Recalibration should be performed when changing model weights/backbone, tokenizer or vocabulary, numerical execution such as quantization, the confidence/commit policy, generation length, step count, or block length. Prompt/template or deployment-domain changes may reuse a table, but should first be checked with held-out consistency and token-coverage diagnostics; a material drop calls for recalibration. Merely increasing the number of deployed generations does not require recalibration.",
        "",
        "## Rebuttal-ready takeaway",
        "",
        "Even with conservative accounting, calibration is a one-time cost repaid after hundreds of generations for a 100-trace table and several thousand for a 5,000-trace table. Since the 100-trace table already recovers nearly all of the measured NFE reduction, it is the most attractive operating point for short-lived deployments; larger tables become economical for sustained serving workloads.",
        "",
    ]
    (args.output_dir / "P21_CALIBRATION_AMORTIZATION.md").write_text("\n".join(lines), encoding="utf-8")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "hardware": "1x NVIDIA A100 (Leonardo)",
            "calibration_seconds_per_trace": calibration_seconds_per_trace,
            "threshold_table_cpu_seconds": args.threshold_cpu_seconds,
            "threshold_table_cpu_cores": args.threshold_cpus,
            "source_run_gpu_hours_end_to_end": args.source_run_gpu_hours,
            "source_to_primary_cost_scale": conservative_scale,
            "by_size": size_rows,
            "by_task": detail_rows,
        }, handle, indent=2)


if __name__ == "__main__":
    main()
