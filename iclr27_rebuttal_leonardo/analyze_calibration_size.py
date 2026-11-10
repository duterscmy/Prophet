#!/usr/bin/env python3
"""Build the P0.1 calibration-size tables, statistics, and figures.

The analysis consumes the immutable lm-eval outputs and calibration audit
files produced on Leonardo.  Runtime fallback coverage is estimated on final
generated text after re-tokenization because the size-ablation jobs disabled
candidate-level trace collection.  This distinction is preserved in all
column names and generated notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


SIZES = [100, 200, 500, 1000, 2000, 5000]
BENCHMARKS = ["gsm8k", "math500", "humaneval"]
DISPLAY = {"gsm8k": "GSM8K", "math500": "MATH500", "humaneval": "HumanEval"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/"
            "rebuttal-size-ablation"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/leonardo_scratch/large/userexternal/mcao0000/models/"
            "LLaDA-8B-Instruct"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "iclr27_rebuttal_leonardo/analysis/calibration_size"
        ),
    )
    return parser.parse_args()


def latest_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} below {root}")
    return matches[-1]


def result_dir_name(benchmark: str, setting: str, size: int | None = None) -> str:
    if setting == "standard":
        return f"{benchmark}_standard"
    if setting == "fixed":
        return f"{benchmark}_fixed_apd_thr0.90"
    if setting == "dynamic" and size is not None:
        return f"{benchmark}_tc_apd_math_n{size}_seed2027"
    raise ValueError((benchmark, setting, size))


def log_name(benchmark: str, setting: str, size: int | None = None) -> str:
    if setting == "standard":
        return f"{benchmark}_standard_len256_block32.log"
    if setting == "fixed":
        return f"{benchmark}_fixed_threshold_thr0.90_len256_block32.log"
    if setting == "dynamic" and size is not None:
        return (
            f"{benchmark}_dynamic_threshold_from_math_n{size}_seed2027_"
            "c99.5_mincount200_minaccepted100_len256_block32_"
            "maxthr0.90_minthr0.01.log"
        )
    raise ValueError((benchmark, setting, size))


def extract_metric(benchmark: str, payload: dict[str, Any]) -> tuple[float, float]:
    result = next(iter(payload["results"].values()))
    if benchmark == "humaneval":
        prefix = "pass@1,"
    elif benchmark == "math500":
        prefix = "math_verify,"
    else:
        prefix = "exact_match,flexible-extract"
    value_key = next(
        key for key in result if key.startswith(prefix) and "stderr" not in key
    )
    stderr_key = next(
        (
            key
            for key in result
            if key.startswith(prefix.replace(",", "_stderr,", 1))
        ),
        None,
    )
    if stderr_key is None:
        stderr_key = value_key.replace(",", "_stderr,", 1)
    return float(result[value_key]), float(result.get(stderr_key, math.nan))


def parse_decode_log(path: Path) -> tuple[float, float, int]:
    text = path.read_text(errors="replace")
    nfes = [int(x) for x in re.findall(r"^Model forward calls:\s*(\d+)\s*$", text, re.M)]
    times = [float(x) for x in re.findall(r"^Decode time:\s*([0-9.]+)s\s*$", text, re.M)]
    if not nfes or len(nfes) != len(times):
        raise ValueError(
            f"Incomplete decode statistics in {path}: {len(nfes)} NFE, {len(times)} times"
        )
    return float(np.mean(nfes)), float(np.mean(times)), len(nfes)


def load_evaluation_table(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    settings = [("standard", None), ("fixed", None)] + [
        ("dynamic", size) for size in SIZES
    ]
    for setting, size in settings:
        for benchmark in BENCHMARKS:
            result_root = run_root / "results" / result_dir_name(benchmark, setting, size)
            payload = json.loads(latest_file(result_root, "results_*.json").read_text())
            accuracy, stderr = extract_metric(benchmark, payload)
            mean_nfe, mean_seconds, samples = parse_decode_log(
                run_root / "logs" / benchmark / log_name(benchmark, setting, size)
            )
            rows.append(
                {
                    "setting": setting,
                    "calibration_size": size,
                    "benchmark": benchmark,
                    "accuracy": accuracy,
                    "accuracy_pct": 100 * accuracy,
                    "accuracy_stderr_pct": 100 * stderr,
                    "accuracy_ci95_low_pct": 100 * max(0.0, accuracy - 1.96 * stderr),
                    "accuracy_ci95_high_pct": 100 * min(1.0, accuracy + 1.96 * stderr),
                    "mean_nfe": mean_nfe,
                    "mean_seconds_per_item": mean_seconds,
                    "items_per_second": 1.0 / mean_seconds,
                    "num_eval_samples": samples,
                }
            )
    frame = pd.DataFrame(rows)
    for benchmark in BENCHMARKS:
        base = frame[(frame.setting == "standard") & (frame.benchmark == benchmark)].iloc[0]
        mask = frame.benchmark == benchmark
        frame.loc[mask, "nfe_speedup_vs_standard"] = (
            base.mean_nfe / frame.loc[mask, "mean_nfe"]
        )
        frame.loc[mask, "wallclock_speedup_vs_standard"] = (
            base.mean_seconds_per_item / frame.loc[mask, "mean_seconds_per_item"]
        )
    return frame


def threshold_stem(size: int) -> str:
    return (
        f"math_n{size}_seed2027_token_threshold_grid_p99.5_"
        "mincount200_minaccepted100"
    )


def load_threshold_statistics(run_root: Path) -> tuple[pd.DataFrame, dict[int, dict[int, float]]]:
    rows: list[dict[str, Any]] = []
    tables: dict[int, dict[int, float]] = {}
    for size in SIZES:
        stem = threshold_stem(size)
        raw = json.loads((run_root / "thresholds" / f"{stem}.json").read_text())
        table = {int(token): float(threshold) for token, threshold in raw.items()}
        tables[size] = table
        with (run_root / "thresholds" / f"{stem}.csv").open(newline="") as handle:
            audit = list(csv.DictReader(handle))
        statuses: dict[str, int] = {}
        for row in audit:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        values = np.asarray(list(table.values()), dtype=float)
        ok = statuses.get("ok", 0)
        fallback = statuses.get("fallback_no_valid_threshold", 0)
        rows.append(
            {
                "calibration_size": size,
                "valid_token_thresholds": len(table),
                "audited_supported_tokens": len(audit),
                "fallback_no_valid_threshold_tokens": fallback,
                "valid_among_audited_pct": 100 * ok / len(audit),
                "fallback_no_valid_among_audited_pct": 100 * fallback / len(audit),
                "unique_threshold_values": len(set(values.tolist())),
                "threshold_mean_raw": float(np.mean(values)),
                "threshold_q25_raw": float(np.quantile(values, 0.25)),
                "threshold_median_raw": float(np.median(values)),
                "threshold_q75_raw": float(np.quantile(values, 0.75)),
                "threshold_min_raw": float(np.min(values)),
                "threshold_max_raw": float(np.max(values)),
                "tokens_capped_at_runtime_max_0.90": int(np.sum(values > 0.90)),
                "tokens_capped_at_runtime_max_0.90_pct": 100 * float(np.mean(values > 0.90)),
            }
        )
    return pd.DataFrame(rows), tables


def unwrap_response(value: Any) -> str:
    while isinstance(value, list):
        if not value:
            return ""
        value = value[0]
    return value if isinstance(value, str) else str(value)


def output_token_coverage(
    run_root: Path,
    tokenizer: Any,
    tables: dict[int, dict[int, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for size in SIZES:
        supported = set(tables[size])
        for benchmark in BENCHMARKS:
            result_root = run_root / "results" / result_dir_name(
                benchmark, "dynamic", size
            )
            sample_path = latest_file(result_root, "samples_*.jsonl")
            texts: list[str] = []
            with sample_path.open() as handle:
                for line in handle:
                    sample = json.loads(line)
                    texts.append(unwrap_response(sample.get("resps", "")))
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]
            total = sum(len(ids) for ids in encoded)
            covered = sum(sum(token in supported for token in ids) for ids in encoded)
            sample_coverages = [
                (sum(token in supported for token in ids) / len(ids))
                for ids in encoded
                if ids
            ]
            rows.append(
                {
                    "calibration_size": size,
                    "benchmark": benchmark,
                    "num_outputs": len(texts),
                    "retokenized_output_tokens": total,
                    "retokenized_supported_tokens": covered,
                    "retokenized_output_token_coverage_pct": 100 * covered / total,
                    "retokenized_output_token_fallback_pct": 100 * (total - covered) / total,
                    "mean_per_output_token_coverage_pct": 100 * float(np.mean(sample_coverages)),
                }
            )
    frame = pd.DataFrame(rows)
    pooled = (
        frame.groupby("calibration_size", as_index=False)[
            ["retokenized_output_tokens", "retokenized_supported_tokens"]
        ]
        .sum()
    )
    pooled["pooled_retokenized_output_token_coverage_pct"] = (
        100
        * pooled.retokenized_supported_tokens
        / pooled.retokenized_output_tokens
    )
    pooled["pooled_retokenized_output_token_fallback_pct"] = (
        100 - pooled.pooled_retokenized_output_token_coverage_pct
    )
    return frame.merge(pooled, on="calibration_size", how="left")


def make_macro_table(
    evaluation: pd.DataFrame,
    thresholds: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    dynamic = evaluation[evaluation.setting == "dynamic"]
    standard = evaluation[evaluation.setting == "standard"]
    standard_macro_mean_nfe = standard.mean_nfe.mean()
    standard_macro_mean_seconds = standard.mean_seconds_per_item.mean()
    macro = (
        dynamic.groupby("calibration_size", as_index=False)
        .agg(
            macro_accuracy_pct=("accuracy_pct", "mean"),
            macro_mean_nfe=("mean_nfe", "mean"),
            macro_seconds_per_item=("mean_seconds_per_item", "mean"),
        )
    )
    # Speedup is the macro-averaged NFE (or seconds) ratio, i.e. average NFE
    # across benchmarks first, then divide -- not the mean of three separate
    # per-benchmark speedup ratios, which would overweight benchmarks with a
    # small denominator.
    macro["macro_nfe_speedup"] = standard_macro_mean_nfe / macro["macro_mean_nfe"]
    macro["macro_wallclock_speedup"] = (
        standard_macro_mean_seconds / macro["macro_seconds_per_item"]
    )
    pooled = coverage[
        [
            "calibration_size",
            "pooled_retokenized_output_token_coverage_pct",
            "pooled_retokenized_output_token_fallback_pct",
        ]
    ].drop_duplicates()
    result = macro.merge(thresholds, on="calibration_size").merge(
        pooled, on="calibration_size"
    )
    result["calibration_size"] = result["calibration_size"].astype(int)
    return result


def make_full_comparison(
    evaluation: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    standard_macro_mean_nfe = (
        evaluation[evaluation.setting == "standard"].mean_nfe.mean()
    )
    settings = [("Standard", "standard", None), ("Fixed threshold 0.90", "fixed", None)]
    settings += [
        (f"Math calibration, {size} trajectories", "dynamic", size)
        for size in SIZES
    ]
    for label, setting, size in settings:
        part = evaluation[evaluation.setting == setting]
        if size is not None:
            part = part[part.calibration_size == size]
        row: dict[str, Any] = {"setting": label}
        for benchmark in BENCHMARKS:
            item = part[part.benchmark == benchmark].iloc[0]
            prefix = DISPLAY[benchmark]
            row[f"{prefix} Acc"] = item.accuracy_pct
            row[f"{prefix} NFE"] = item.mean_nfe
            row[f"{prefix} s/it"] = item.mean_seconds_per_item
            row[f"{prefix} Speedup"] = item.nfe_speedup_vs_standard
        row["Macro Acc"] = part.accuracy_pct.mean()
        row["Macro NFE"] = part.mean_nfe.mean()
        row["Macro s/it"] = part.mean_seconds_per_item.mean()
        # Ratio of the macro-averaged NFE values (average NFE across the three
        # benchmarks first, then divide), not the mean of three separate
        # per-benchmark speedup ratios.
        row["Macro Speedup"] = standard_macro_mean_nfe / row["Macro NFE"]
        if size is None:
            row["Valid token thresholds"] = math.nan
            row["Output-token coverage"] = math.nan
        else:
            summary = macro[macro.calibration_size == size].iloc[0]
            row["Valid token thresholds"] = summary.valid_token_thresholds
            row["Output-token coverage"] = summary.pooled_retokenized_output_token_coverage_pct
        rows.append(row)
    return pd.DataFrame(rows)


def configure_plotting() -> Any:
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


def save_tradeoff_figure(
    output_dir: Path,
    evaluation: pd.DataFrame,
    macro: pd.DataFrame,
) -> None:
    plt = configure_plotting()
    colors = {"gsm8k": "#0072B2", "math500": "#D55E00", "humaneval": "#009E73"}
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), constrained_layout=True)
    dynamic = evaluation[evaluation.setting == "dynamic"]
    for benchmark in BENCHMARKS:
        part = dynamic[dynamic.benchmark == benchmark].sort_values("calibration_size")
        axes[0, 0].plot(
            part.calibration_size,
            part.accuracy_pct,
            marker="o",
            color=colors[benchmark],
            label=DISPLAY[benchmark],
        )
        axes[0, 1].plot(
            part.calibration_size,
            part.mean_nfe,
            marker="o",
            color=colors[benchmark],
            label=DISPLAY[benchmark],
        )
        axes[1, 0].plot(
            part.calibration_size,
            part.mean_seconds_per_item,
            marker="o",
            color=colors[benchmark],
            label=DISPLAY[benchmark],
        )
    axes[0, 0].plot(
        macro.calibration_size,
        macro.macro_accuracy_pct,
        marker="D",
        linestyle="--",
        color="black",
        label="Macro average",
    )
    axes[0, 0].set_title("(a) Task accuracy")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("(b) Inference compute")
    axes[0, 1].set_ylabel("Mean NFE (lower is better)")
    axes[1, 0].set_title("(c) Decode latency")
    axes[1, 0].set_ylabel("Seconds / item (lower is better)")
    axes[1, 1].plot(
        macro.calibration_size,
        macro.macro_nfe_speedup,
        marker="o",
        color="#CC79A7",
        label="NFE speedup",
    )
    axes[1, 1].plot(
        macro.calibration_size,
        macro.macro_wallclock_speedup,
        marker="s",
        color="#56B4E9",
        label="Wall-clock speedup",
    )
    axes[1, 1].set_title("(d) Macro speedup vs. Standard")
    axes[1, 1].set_ylabel("Speedup (×)")
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xticks(SIZES, labels=[str(x) for x in SIZES])
        ax.set_xlabel("Calibration trajectories")
        ax.grid(True)
    axes[0, 0].legend(ncol=2, frameon=False)
    axes[1, 1].legend(frameon=False)
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"figure_p01_calibration_size_tradeoff.{suffix}", bbox_inches="tight")
    plt.close(fig)


def save_support_figure(
    output_dir: Path,
    thresholds: pd.DataFrame,
    coverage: pd.DataFrame,
    tables: dict[int, dict[int, float]],
) -> None:
    plt = configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2), constrained_layout=True)
    positions = np.arange(len(SIZES))
    distributions = [np.clip(list(tables[size].values()), 0.01, 0.90) for size in SIZES]
    box = axes[0, 0].boxplot(distributions, positions=positions, patch_artist=True, showfliers=False)
    for patch in box["boxes"]:
        patch.set_facecolor("#56B4E9")
        patch.set_alpha(0.65)
    axes[0, 0].set_xticks(positions, [str(x) for x in SIZES])
    axes[0, 0].set_title("(a) Runtime-clipped threshold distribution")
    axes[0, 0].set_ylabel("Token threshold")
    axes[0, 0].set_xlabel("Calibration trajectories")
    axes[0, 0].grid(True, axis="y")

    axes[0, 1].bar(
        positions,
        thresholds.valid_token_thresholds,
        label="Valid threshold",
        color="#009E73",
    )
    axes[0, 1].bar(
        positions,
        thresholds.fallback_no_valid_threshold_tokens,
        bottom=thresholds.valid_token_thresholds,
        label="No valid threshold",
        color="#E69F00",
    )
    axes[0, 1].set_xticks(positions, [str(x) for x in SIZES])
    axes[0, 1].set_title("(b) Audited token support")
    axes[0, 1].set_ylabel("Token types")
    axes[0, 1].set_xlabel("Calibration trajectories")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(True, axis="y")

    for benchmark, color in zip(BENCHMARKS, ["#0072B2", "#D55E00", "#009E73"]):
        part = coverage[coverage.benchmark == benchmark].sort_values("calibration_size")
        axes[1, 0].plot(
            part.calibration_size,
            part.retokenized_output_token_coverage_pct,
            marker="o",
            color=color,
            label=DISPLAY[benchmark],
        )
    pooled = coverage.drop_duplicates("calibration_size").sort_values("calibration_size")
    axes[1, 0].plot(
        pooled.calibration_size,
        pooled.pooled_retokenized_output_token_coverage_pct,
        marker="D",
        linestyle="--",
        color="black",
        label="Pooled",
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xticks(SIZES, labels=[str(x) for x in SIZES])
    axes[1, 0].set_title("(c) Final-output token coverage")
    axes[1, 0].set_ylabel("Retokenized coverage (%)")
    axes[1, 0].set_xlabel("Calibration trajectories")
    axes[1, 0].legend(ncol=2, frameon=False)
    axes[1, 0].grid(True)

    axes[1, 1].plot(
        thresholds.calibration_size,
        thresholds["tokens_capped_at_runtime_max_0.90_pct"],
        marker="o",
        color="#CC79A7",
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xticks(SIZES, labels=[str(x) for x in SIZES])
    axes[1, 1].set_title("(d) Thresholds clipped by max=0.90")
    axes[1, 1].set_ylabel("Valid token types clipped (%)")
    axes[1, 1].set_xlabel("Calibration trajectories")
    axes[1, 1].grid(True)
    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"figure_p01_threshold_support.{suffix}", bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    output_dir: Path,
    evaluation: pd.DataFrame,
    macro: pd.DataFrame,
    coverage: pd.DataFrame,
    full_comparison: pd.DataFrame,
) -> None:
    dynamic = evaluation[evaluation.setting == "dynamic"]
    lines = [
        "# P0.1 Calibration-size ablation",
        "",
        "Single calibration-subset seed: 2027. Accuracy metrics are HumanEval pass@1, "
        "MATH500 math_verify, and GSM8K flexible-extract.",
        "",
        "Every speedup below is the ratio of macro-averaged values: NFE (or seconds) is "
        "averaged across the three benchmarks first, and the speedup is the ratio of "
        "those two averages, not the mean of three separate per-benchmark ratios. The "
        "unqualified `Speedup` columns use NFE as the basis.",
        "",
        "## Main summary",
        "",
        "| Calibration trajectories | Macro Acc. (%) | Mean NFE | s/item | NFE speedup | Wall-clock speedup | Valid thresholds | Output-token coverage (%) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in macro.itertuples():
        lines.append(
            f"| {int(row.calibration_size)} | {row.macro_accuracy_pct:.2f} | "
            f"{row.macro_mean_nfe:.2f} | {row.macro_seconds_per_item:.3f} | "
            f"{row.macro_nfe_speedup:.2f}× | {row.macro_wallclock_speedup:.2f}× | "
            f"{row.valid_token_thresholds:,} | "
            f"{row.pooled_retokenized_output_token_coverage_pct:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Full comparison",
            "",
            "| Setting | GSM8K Acc/NFE/s | MATH500 Acc/NFE/s | HumanEval Acc/NFE/s | Macro Acc/NFE/s | Speedup | Valid thresholds |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in full_comparison.iterrows():
        valid = "–" if pd.isna(row["Valid token thresholds"]) else f"{int(row['Valid token thresholds']):,}"
        lines.append(
            f"| {row['setting']} | {row['GSM8K Acc']:.2f}/{row['GSM8K NFE']:.2f}/{row['GSM8K s/it']:.3f} | "
            f"{row['MATH500 Acc']:.2f}/{row['MATH500 NFE']:.2f}/{row['MATH500 s/it']:.3f} | "
            f"{row['HumanEval Acc']:.2f}/{row['HumanEval NFE']:.2f}/{row['HumanEval s/it']:.3f} | "
            f"{row['Macro Acc']:.2f}/{row['Macro NFE']:.2f}/{row['Macro s/it']:.3f} | "
            f"{row['Macro Speedup']:.2f}× | {valid} |"
        )
    first = macro[macro.calibration_size == 100].iloc[0]
    full = macro[macro.calibration_size == 5000].iloc[0]
    standard_nfe = evaluation[evaluation.setting == "standard"].mean_nfe.mean()
    recovered = (standard_nfe - first.macro_mean_nfe) / (
        standard_nfe - full.macro_mean_nfe
    )
    lines.extend(
        [
            "",
            "## Rebuttal-ready observations",
            "",
            f"- 100 trajectories recover {100 * recovered:.1f}% of the macro NFE reduction obtained with 5,000 trajectories.",
            f"- Valid token-specific thresholds grow from {int(first.valid_token_thresholds):,} to {int(full.valid_token_thresholds):,}.",
            f"- Macro NFE changes from {first.macro_mean_nfe:.2f} at 100 trajectories to {full.macro_mean_nfe:.2f} at 5,000 trajectories.",
            f"- Macro accuracy ranges from {macro.macro_accuracy_pct.min():.2f}% to {macro.macro_accuracy_pct.max():.2f}% and is not monotonic in calibration size.",
            "- Accuracy uncertainty intervals in evaluation_metrics_long.csv use lm-eval standard errors and a normal 95% approximation.",
            "",
            "## Coverage interpretation",
            "",
            "`fallback_no_valid_threshold` is an exact calibration-audit status among token types with sufficient total count. "
            "Final-output coverage is an approximate deployment statistic computed by re-tokenizing saved final responses; "
            "candidate-level decoder traces were intentionally disabled for these jobs, so it must not be described as exact candidate-trace fallback.",
            "",
            "## Per-benchmark dynamic results",
            "",
            "| Size | Benchmark | Acc. (%) | 95% CI | NFE | s/item | Wall-clock speedup |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dynamic.sort_values(["calibration_size", "benchmark"]).itertuples():
        lines.append(
            f"| {int(row.calibration_size)} | {DISPLAY[row.benchmark]} | "
            f"{row.accuracy_pct:.2f} | [{row.accuracy_ci95_low_pct:.2f}, {row.accuracy_ci95_high_pct:.2f}] | "
            f"{row.mean_nfe:.2f} | {row.mean_seconds_per_item:.3f} | "
            f"{row.wallclock_speedup_vs_standard:.2f}× |"
        )
    (output_dir / "P01_CALIBRATION_SIZE_SUMMARY.md").write_text(
        "\n".join(lines) + "\n"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = load_evaluation_table(args.run_root)
    thresholds, tables = load_threshold_statistics(args.run_root)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    coverage = output_token_coverage(args.run_root, tokenizer, tables)
    macro = make_macro_table(evaluation, thresholds, coverage)
    full_comparison = make_full_comparison(evaluation, macro)

    evaluation.to_csv(args.output_dir / "evaluation_metrics_long.csv", index=False)
    thresholds.to_csv(args.output_dir / "threshold_statistics.csv", index=False)
    coverage.to_csv(args.output_dir / "output_token_coverage.csv", index=False)
    macro.to_csv(args.output_dir / "calibration_size_macro_summary.csv", index=False)
    full_comparison.to_csv(args.output_dir / "calibration_size_full_comparison.csv", index=False)
    write_markdown(args.output_dir, evaluation, macro, coverage, full_comparison)
    save_tradeoff_figure(args.output_dir, evaluation, macro)
    save_support_figure(args.output_dir, thresholds, coverage, tables)
    print(macro.to_string(index=False))
    print(f"Wrote P0.1 artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
