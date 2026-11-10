#!/usr/bin/env python3
"""Consolidate the P0.2 / P2.3 out-of-domain evaluations into tables.

The script reads the immutable lm-eval ``results_*.json`` payloads and the
decoder statistics printed into the per-run logs, then joins them on the
decoding setting parsed from both file names.  Nothing is recomputed from
generations, so every number here is traceable to one run directory.

Covered tasks: IFEval, ARC-Challenge (chat template), XSum-500,
WMT16 en-ro, and WritingPrompts-100 at generation length 1024.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUN_ROOT = Path("/leonardo_scratch/large/userexternal/mcao0000/prophet-runs")

SIZES = [100, 200, 500, 1000, 2000, 5000]

TASKS: dict[str, dict[str, Any]] = {
    "ifeval": {
        "label": "IFEval",
        "run_root": RUN_ROOT / "rebuttal-ifeval",
        "prefix": "ifeval",
        "gen_length": 256,
        "metrics": [
            ("prompt_level_strict_acc,none", "prompt_level_strict_acc"),
            ("inst_level_strict_acc,none", "inst_level_strict_acc"),
            ("prompt_level_loose_acc,none", "prompt_level_loose_acc"),
            ("inst_level_loose_acc,none", "inst_level_loose_acc"),
        ],
        "primary": "prompt_level_strict_acc",
        "primary_label": "Prompt-level strict acc. (%)",
        "primary_scale": 100.0,
        "higher_is_better": True,
    },
    "arc_challenge": {
        "label": "ARC-Challenge",
        "run_root": RUN_ROOT / "rebuttal-arc-challenge",
        "prefix": "arc_challenge",
        "gen_length": 256,
        "metrics": [("exact_match,remove_whitespace", "exact_match")],
        "primary": "exact_match",
        "primary_label": "Exact match (%)",
        "primary_scale": 100.0,
        "higher_is_better": True,
    },
    "xsum_500": {
        "label": "XSum-500",
        "run_root": RUN_ROOT / "rebuttal-xsum-500",
        "prefix": "xsum_500",
        "gen_length": 256,
        "metrics": [
            ("rouge1,none", "rouge1"),
            ("rouge2,none", "rouge2"),
            ("rougeL,none", "rougeL"),
        ],
        "primary": "rougeL",
        "primary_label": "ROUGE-L (%)",
        "primary_scale": 100.0,
        "higher_is_better": True,
    },
    "wmt16_en_ro": {
        "label": "WMT16 en-ro",
        "run_root": RUN_ROOT / "rebuttal-wmt16-en-ro",
        "prefix": "wmt16_en_ro",
        "gen_length": 256,
        "metrics": [
            ("bleu,none", "bleu"),
            ("chrf,none", "chrf"),
            ("ter,none", "ter"),
        ],
        "primary": "bleu",
        "primary_label": "BLEU",
        "primary_scale": 1.0,
        "higher_is_better": True,
    },
    "writingprompts_100": {
        "label": "WritingPrompts-100 (len 1024)",
        "run_root": RUN_ROOT / "rebuttal-writingprompts-100-len1024",
        "prefix": "writingprompts_100",
        "gen_length": 1024,
        "metrics": [
            ("word_count,none", "word_count"),
            ("distinct_4,none", "distinct_4"),
            ("repetition_4,none", "repetition_4"),
            ("repetition_8,none", "repetition_8"),
            ("sentence_repetition,none", "sentence_repetition"),
        ],
        "primary": "repetition_4",
        "primary_label": "4-gram repetition (lower is better)",
        "primary_scale": 1.0,
        "higher_is_better": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("iclr27_rebuttal_leonardo/analysis/ood_tasks"),
    )
    return parser.parse_args()


def result_key(prefix: str, name: str) -> tuple[str, int | None, int] | None:
    """Map an lm-eval output directory name to a decoding setting key."""
    patterns = [
        (rf"^{prefix}_standard_len(\d+)_block32(?:_chattemplate)?$", "standard", None),
        (
            rf"^{prefix}_fixed_apd(?:_thr0\.90)?_len(\d+)_block32(?:_chattemplate)?$",
            "fixed_0.90",
            None,
        ),
        (rf"^{prefix}_tc_apd_len(\d+)_block32(?:_chattemplate)?$", "tc_apd", 5000),
    ]
    for pattern, kind, size in patterns:
        match = re.match(pattern, name)
        if match:
            return kind, size, int(match.group(1))
    match = re.match(
        rf"^{prefix}_tc_apd_math_n(\d+)_seed2027_len(\d+)_block32(?:_chattemplate)?$",
        name,
    )
    if match:
        return "tc_apd", int(match.group(1)), int(match.group(2))
    return None


def log_key(prefix: str, name: str) -> tuple[str, int | None, int] | None:
    """Map a decoder log file name to the same decoding setting key."""
    patterns = [
        (
            rf"^{prefix}_standard_len(\d+)_block32(?:_chattemplate)?\.log$",
            "standard",
            None,
        ),
        (
            rf"^{prefix}_fixed_threshold_thr0\.90_len(\d+)_block32"
            r"(?:_chattemplate)?\.log$",
            "fixed_0.90",
            None,
        ),
    ]
    for pattern, kind, size in patterns:
        match = re.match(pattern, name)
        if match:
            return kind, size, int(match.group(1))
    match = re.match(
        rf"^{prefix}_dynamic_threshold_from_math_n(\d+)_seed2027_"
        r"c99\.5_mincount200_minaccepted100_len(\d+)_block32"
        r"(?:_chattemplate)?_maxthr0\.90_minthr0\.01\.log$",
        name,
    )
    if match:
        return "tc_apd", int(match.group(1)), int(match.group(2))
    return None


def latest_result_payload(directory: Path) -> dict[str, Any] | None:
    matches = sorted(directory.rglob("results_*.json"))
    if not matches:
        return None
    return json.loads(matches[-1].read_text())


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


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def setting_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    order = {"standard": 0, "fixed_0.90": 1, "tc_apd": 2}
    return (
        int(row["gen_length"]),
        order[row["setting"]],
        int(row["calibration_size"]) if row["calibration_size"] else 0,
    )


def collect_task(task: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    run_root = config["run_root"]
    prefix = config["prefix"]

    logs: dict[tuple[str, int | None, int], Path] = {}
    for path in sorted((run_root / "logs").glob("*.log")):
        key = log_key(prefix, path.name)
        if key is not None:
            logs[key] = path

    rows: list[dict[str, Any]] = []
    for directory in sorted((run_root / "results").iterdir()):
        if not directory.is_dir():
            continue
        key = result_key(prefix, directory.name)
        if key is None:
            raise ValueError(f"Unrecognized result directory: {directory}")
        payload = latest_result_payload(directory)
        if payload is None:
            print(f"[skip] no results_*.json below {directory}")
            continue
        if key not in logs:
            raise FileNotFoundError(f"No decoder log for {directory.name} ({key})")
        mean_nfe, mean_seconds, samples = parse_decode_log(logs[key])

        task_name, result = next(iter(payload["results"].items()))
        n_samples = payload.get("n-samples", {}).get(task_name, {})
        row: dict[str, Any] = {
            "task": task,
            "task_label": config["label"],
            "lm_eval_task": task_name,
            "setting": key[0],
            "calibration_size": key[1] if key[1] is not None else "",
            "gen_length": key[2],
            "evaluated_examples": int(n_samples.get("effective", samples)),
            "logged_generations": samples,
            "mean_nfe": mean_nfe,
            "mean_decode_seconds": mean_seconds,
            "result_dir": str(directory),
            "log_file": logs[key].name,
        }
        for json_key, column in config["metrics"]:
            row[column] = as_float(result.get(json_key))
            stderr_key = json_key.replace(",", "_stderr,", 1)
            row[f"{column}_stderr"] = as_float(result.get(stderr_key))
        rows.append(row)

    rows.sort(key=setting_sort_key)
    baselines = {
        row["gen_length"]: row for row in rows if row["setting"] == "standard"
    }
    for row in rows:
        baseline = baselines.get(row["gen_length"])
        if baseline is None:
            row["nfe_speedup"] = math.nan
            row["wallclock_speedup"] = math.nan
            row["primary_delta_vs_standard"] = math.nan
            continue
        row["nfe_speedup"] = baseline["mean_nfe"] / row["mean_nfe"]
        row["wallclock_speedup"] = (
            baseline["mean_decode_seconds"] / row["mean_decode_seconds"]
        )
        primary = config["primary"]
        scale = config["primary_scale"]
        row["primary_delta_vs_standard"] = scale * (row[primary] - baseline[primary])
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def setting_label(row: dict[str, Any]) -> str:
    if row["setting"] == "standard":
        return "Standard"
    if row["setting"] == "fixed_0.90":
        return "Fixed APD 0.90"
    return f"TC-APD Math-{row['calibration_size']}"


def fmt(value: float, digits: int = 2) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    for task, config in TASKS.items():
        all_rows[task] = collect_task(task, config)

    efficiency_fields = [
        "task",
        "task_label",
        "setting",
        "calibration_size",
        "gen_length",
        "evaluated_examples",
        "logged_generations",
        "mean_nfe",
        "mean_decode_seconds",
        "nfe_speedup",
        "wallclock_speedup",
        "primary_delta_vs_standard",
        "log_file",
        "result_dir",
    ]
    efficiency_rows = [row for task in TASKS for row in all_rows[task]]
    write_csv(
        args.output_dir / "ood_efficiency.csv", efficiency_rows, efficiency_fields
    )

    metric_rows: list[dict[str, Any]] = []
    for task, config in TASKS.items():
        for row in all_rows[task]:
            for _, column in config["metrics"]:
                metric_rows.append(
                    {
                        "task": task,
                        "task_label": config["label"],
                        "setting": row["setting"],
                        "calibration_size": row["calibration_size"],
                        "gen_length": row["gen_length"],
                        "metric": column,
                        "value": row[column],
                        "stderr": row[f"{column}_stderr"],
                        "is_primary": int(column == config["primary"]),
                    }
                )
    write_csv(
        args.output_dir / "ood_metrics_long.csv",
        metric_rows,
        ["task", "task_label", "setting", "calibration_size", "gen_length",
         "metric", "value", "stderr", "is_primary"],
    )

    main_rows: list[dict[str, Any]] = []
    for task, config in TASKS.items():
        default_length = config["gen_length"]
        wanted = [("standard", ""), ("fixed_0.90", ""), ("tc_apd", 500), ("tc_apd", 5000)]
        for setting, size in wanted:
            match = next(
                (
                    row
                    for row in all_rows[task]
                    if row["setting"] == setting
                    and row["gen_length"] == default_length
                    and (size == "" or row["calibration_size"] == size)
                ),
                None,
            )
            if match is None:
                continue
            main_rows.append(
                {
                    "task": task,
                    "task_label": config["label"],
                    "setting_label": setting_label(match),
                    "gen_length": match["gen_length"],
                    "evaluated_examples": match["evaluated_examples"],
                    "primary_metric": config["primary"],
                    "primary_value": config["primary_scale"] * match[config["primary"]],
                    "primary_delta_vs_standard": match["primary_delta_vs_standard"],
                    "mean_nfe": match["mean_nfe"],
                    "nfe_speedup": match["nfe_speedup"],
                    "mean_decode_seconds": match["mean_decode_seconds"],
                    "wallclock_speedup": match["wallclock_speedup"],
                }
            )
    write_csv(
        args.output_dir / "ood_main_table.csv",
        main_rows,
        list(main_rows[0]),
    )

    sweep_rows: list[dict[str, Any]] = []
    for task, config in TASKS.items():
        default_length = config["gen_length"]
        for row in all_rows[task]:
            if row["setting"] != "tc_apd" or row["gen_length"] != default_length:
                continue
            sweep_rows.append(
                {
                    "task": task,
                    "task_label": config["label"],
                    "calibration_size": row["calibration_size"],
                    "primary_metric": config["primary"],
                    "primary_value": config["primary_scale"] * row[config["primary"]],
                    "primary_delta_vs_standard": row["primary_delta_vs_standard"],
                    "mean_nfe": row["mean_nfe"],
                    "nfe_speedup": row["nfe_speedup"],
                    "wallclock_speedup": row["wallclock_speedup"],
                }
            )
    write_csv(
        args.output_dir / "ood_calibration_size_sweep.csv",
        sweep_rows,
        list(sweep_rows[0]),
    )

    probe_rows = [
        row
        for task, config in TASKS.items()
        for row in all_rows[task]
        if row["gen_length"] != config["gen_length"]
    ]
    if probe_rows:
        write_csv(
            args.output_dir / "ood_generation_length_probe.csv",
            probe_rows,
            efficiency_fields,
        )

    summary = {
        "tasks": {
            task: {
                "label": config["label"],
                "lm_eval_task": all_rows[task][0]["lm_eval_task"],
                "default_gen_length": config["gen_length"],
                "primary_metric": config["primary"],
                "higher_is_better": config["higher_is_better"],
                "settings_evaluated": len(all_rows[task]),
                "rows": all_rows[task],
            }
            for task, config in TASKS.items()
        }
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )

    lines = [
        "# P0.2 / P2.3 Out-of-domain evaluation",
        "",
        "All runs use the LLaDA-8B-Instruct Math calibration tables produced for the",
        "P0.1 size ablation. No task below was recalibrated, and none of these tasks",
        "participated in the original calibration-source selection.",
        "",
    ]
    for task, config in TASKS.items():
        rows = [
            row
            for row in all_rows[task]
            if row["gen_length"] == config["gen_length"]
        ]
        lines += [
            f"## {config['label']}",
            "",
            f"lm-eval task `{rows[0]['lm_eval_task']}`, "
            f"{rows[0]['evaluated_examples']:,} examples, "
            f"generation length {config['gen_length']}, block length 32.",
            "",
            f"| Setting | {config['primary_label']} | Δ vs Standard | Mean NFE | "
            "NFE speedup | Mean decode time (s) | Wall-clock speedup |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            delta = row["primary_delta_vs_standard"]
            delta_text = "---" if row["setting"] == "standard" else f"{delta:+.2f}"
            lines.append(
                f"| {setting_label(row)} "
                f"| {fmt(config['primary_scale'] * row[config['primary']])} "
                f"| {delta_text} "
                f"| {fmt(row['mean_nfe'], 1)} "
                f"| {fmt(row['nfe_speedup'])}x "
                f"| {fmt(row['mean_decode_seconds'])} "
                f"| {fmt(row['wallclock_speedup'])}x |"
            )
        secondary = [
            column for _, column in config["metrics"] if column != config["primary"]
        ]
        if secondary:
            lines += [
                "",
                "Secondary metrics: "
                + ", ".join(
                    f"`{column}` "
                    + "/".join(
                        fmt(row[column], 4)
                        for row in rows
                        if row["setting"] in {"standard", "tc_apd"}
                        and (row["setting"] == "standard" or row["calibration_size"] == 5000)
                    )
                    for column in secondary
                )
                + " (Standard / TC-APD Math-5000).",
            ]
        lines.append("")

    if probe_rows:
        lines += [
            "## Generation-length probe",
            "",
            "Fixed-threshold probes at a longer generation budget, used to check that",
            "the 256-token budget is not the binding constraint on these tasks.",
            "",
            "| Task | Setting | Gen length | Primary metric | Mean NFE | Mean decode time (s) |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in probe_rows:
            config = TASKS[row["task"]]
            lines.append(
                f"| {config['label']} | {setting_label(row)} | {row['gen_length']} "
                f"| {fmt(config['primary_scale'] * row[config['primary']])} "
                f"| {fmt(row['mean_nfe'], 1)} | {fmt(row['mean_decode_seconds'])} |"
            )
        lines.append("")

    lines += [
        "## Interpretation boundary",
        "",
        "Speedups are computed against the Standard run at the same generation length",
        "and are means over per-example decoder statistics, not paired confidence",
        "intervals. XSum ROUGE, WMT BLEU/chrF/TER, and the WritingPrompts repetition",
        "statistics are automatic metrics; the WritingPrompts numbers measure",
        "degeneration, not human-judged writing quality.",
        "",
        "## Files",
        "",
        "- `ood_main_table.csv`: compact Standard / Fixed 0.90 / Math-500 / Math-5000 table.",
        "- `ood_calibration_size_sweep.csv`: calibration-size sweep per task.",
        "- `ood_metrics_long.csv`: every reported metric with lm-eval standard errors.",
        "- `ood_efficiency.csv`: NFE, decode time, and speedups for every run.",
        "- `ood_generation_length_probe.csv`: longer-generation probe runs.",
        "- `summary.json`: machine-readable dump of all joined rows.",
    ]
    (args.output_dir / "OOD_TASKS_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(efficiency_rows)} joined runs to {args.output_dir}")


if __name__ == "__main__":
    main()
