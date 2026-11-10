#!/usr/bin/env python3
"""P2.2 decoding-configuration transfer and model-variant transfer tables.

Both experiment families reuse the single Math-5000 threshold table that was
calibrated once on LLaDA-8B-Instruct Standard traces at
``gen_length=256, block_length=32``.  Nothing is recalibrated here, so the
tables answer one question only: does an existing table still accelerate a
changed decoding configuration or a changed model, and what happens to task
accuracy?

Accuracy comes from the lm-eval ``results_*.json`` payloads; NFE and
seconds-per-item come from the decoder statistics in the matching inference
log, whose stem equals the result-directory name.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUN_ROOT = Path("/leonardo_scratch/large/userexternal/mcao0000/prophet-runs")
TRANSFER_ROOT = RUN_ROOT / "p22-transfer"
VARIANT_ROOT = RUN_ROOT / "p22-model-variants"

BENCHMARKS = ["gsm8k", "math500", "humaneval"]
DISPLAY = {"gsm8k": "GSM8K", "math500": "MATH500", "humaneval": "HumanEval"}
METHODS = {
    "standard": "Standard",
    "fixed_apd": "Fixed APD 0.90",
    "tc_apd": "TC-APD Math-5000",
}
VARIANTS = {
    "fine_tuned": "Fine-tuned (JustGRPO-GSM8K)",
    "quantized_int8": "INT8 GPTQ",
}
METRIC_PREFIX = {
    "gsm8k": "exact_match,flexible-extract",
    "math500": "math_verify,",
    "humaneval": "pass@1,",
}

# Runs the user explicitly cancelled and does not want resubmitted.
INTENTIONALLY_CANCELLED = {
    ("gsm8k", "standard", "len512_block32"),
}

TRANSFER_PATTERN = re.compile(
    r"^(?P<benchmark>gsm8k|math500|humaneval)_"
    r"(?P<method>standard|fixed_apd|tc_apd)_"
    r"len(?P<length>\d+)_block(?P<block>\d+)"
    r"(?P<chat>_chattemplate)?_math_n5000_transfer$"
)
VARIANT_PATTERN = re.compile(
    r"^(?P<benchmark>gsm8k|math500|humaneval)_"
    r"(?P<variant>fine_tuned|quantized_int8)_"
    r"(?P<method>standard|fixed_apd|tc_apd)_math_n5000_"
    r"len(?P<length>\d+)_block(?P<block>\d+)"
    r"(?:_limit(?P<limit>\d+))?(?P<smoke>_smoke)?$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "iclr27_rebuttal_leonardo/analysis/configuration_and_model_transfer"
        ),
    )
    return parser.parse_args()


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


def collect(root: Path, log_subdir_key: str, pattern: re.Pattern[str]):
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for directory in sorted((root / "results").iterdir()):
        if not directory.is_dir():
            continue
        match = pattern.match(directory.name)
        if match is None:
            raise ValueError(f"Unrecognized result directory: {directory}")
        fields = match.groupdict()
        if fields.get("smoke"):
            continue
        benchmark = fields["benchmark"]
        log_dir = fields[log_subdir_key] if log_subdir_key in fields else benchmark
        log_path = root / "logs" / log_dir / f"{directory.name}.log"
        payload_path = latest_result(directory)
        if payload_path is None or not log_path.is_file():
            missing.append(
                {
                    "result_dir": str(directory),
                    "expected_log": str(log_path),
                    "has_aggregate_result": int(payload_path is not None),
                    "has_log": int(log_path.is_file()),
                }
            )
            continue
        mean_nfe, mean_seconds, logged = parse_decode_log(log_path)
        accuracy, examples = extract_accuracy(
            benchmark, json.loads(payload_path.read_text())
        )
        row = {
            "benchmark": benchmark,
            "task": DISPLAY[benchmark],
            "method": fields["method"],
            "method_label": METHODS[fields["method"]],
            "gen_length": int(fields["length"]),
            "block_length": int(fields["block"]),
            "accuracy_pct": accuracy,
            "mean_nfe": mean_nfe,
            "seconds_per_item": mean_seconds,
            "evaluated_examples": examples,
            "logged_generations": logged,
            "result_file": str(payload_path),
            "log_file": str(log_path),
        }
        if "chat" in fields:
            row["chat_template"] = int(bool(fields["chat"]))
        if "variant" in fields:
            row["variant"] = fields["variant"]
            row["variant_label"] = VARIANTS[fields["variant"]]
        if fields.get("limit"):
            row["example_limit"] = int(fields["limit"])
        rows.append(row)
    return rows, missing


def configuration_key(row: dict[str, Any]) -> str:
    suffix = "_chattemplate" if row.get("chat_template") else ""
    return f"len{row['gen_length']}_block{row['block_length']}{suffix}"


def configuration_label(key: str) -> str:
    match = re.match(r"^len(\d+)_block(\d+)(_chattemplate)?$", key)
    assert match is not None
    chat = " + chat template" if match.group(3) else ""
    return f"length {match.group(1)}, block {match.group(2)}{chat}"


def add_speedups(rows: list[dict[str, Any]], group_fields: list[str]) -> None:
    def group_of(row: dict[str, Any]) -> tuple:
        return tuple(row[field] for field in group_fields)

    standards = {
        group_of(row): row for row in rows if row["method"] == "standard"
    }
    for row in rows:
        reference = standards.get(group_of(row))
        if reference is None:
            row["nfe_speedup_vs_standard"] = math.nan
            row["wallclock_speedup_vs_standard"] = math.nan
            row["accuracy_delta_vs_standard_pp"] = math.nan
            continue
        row["nfe_speedup_vs_standard"] = reference["mean_nfe"] / row["mean_nfe"]
        row["wallclock_speedup_vs_standard"] = (
            reference["seconds_per_item"] / row["seconds_per_item"]
        )
        row["accuracy_delta_vs_standard_pp"] = (
            row["accuracy_pct"] - reference["accuracy_pct"]
        )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 2) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def speedup(value: float) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.2f}x"


def delta(row: dict[str, Any]) -> str:
    if row["method"] == "standard":
        return "---"
    value = row["accuracy_delta_vs_standard_pp"]
    return "n/a" if math.isnan(value) else f"{value:+.2f}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    transfer, transfer_missing = collect(TRANSFER_ROOT, "benchmark", TRANSFER_PATTERN)
    for row in transfer:
        row["configuration"] = configuration_key(row)
    add_speedups(transfer, ["benchmark", "configuration"])

    variants, variant_missing = collect(VARIANT_ROOT, "variant", VARIANT_PATTERN)
    add_speedups(variants, ["benchmark", "variant"])

    transfer_fields = [
        "configuration",
        "benchmark",
        "task",
        "method",
        "method_label",
        "gen_length",
        "block_length",
        "chat_template",
        "accuracy_pct",
        "accuracy_delta_vs_standard_pp",
        "mean_nfe",
        "seconds_per_item",
        "nfe_speedup_vs_standard",
        "wallclock_speedup_vs_standard",
        "evaluated_examples",
        "logged_generations",
        "result_file",
        "log_file",
    ]
    order = {"standard": 0, "fixed_apd": 1, "tc_apd": 2}
    transfer.sort(
        key=lambda r: (r["configuration"], BENCHMARKS.index(r["benchmark"]), order[r["method"]])
    )
    write_csv(args.output_dir / "configuration_transfer_table.csv", transfer, transfer_fields)

    variant_fields = [
        "variant",
        "variant_label",
        "benchmark",
        "task",
        "method",
        "method_label",
        "gen_length",
        "block_length",
        "example_limit",
        "accuracy_pct",
        "accuracy_delta_vs_standard_pp",
        "mean_nfe",
        "seconds_per_item",
        "nfe_speedup_vs_standard",
        "wallclock_speedup_vs_standard",
        "evaluated_examples",
        "logged_generations",
        "result_file",
        "log_file",
    ]
    variants.sort(
        key=lambda r: (r["variant"], BENCHMARKS.index(r["benchmark"]), order[r["method"]])
    )
    write_csv(args.output_dir / "model_variant_table.csv", variants, variant_fields)

    missing = [
        {**entry, "family": "configuration_transfer"} for entry in transfer_missing
    ] + [{**entry, "family": "model_variant"} for entry in variant_missing]
    if missing:
        write_csv(
            args.output_dir / "missing_runs.csv",
            missing,
            ["family", "result_dir", "expected_log", "has_aggregate_result", "has_log"],
        )

    lines = [
        "# P2.2 Configuration and model-variant transfer of one fixed threshold table",
        "",
        "Every row below reuses the single Math-5000 threshold table calibrated once on",
        "LLaDA-8B-Instruct Standard traces at `gen_length=256, block_length=32`",
        "(`rebuttal-size-ablation/thresholds/math_n5000_seed2027_token_threshold_grid_"
        "p99.5_mincount200_minaccepted100.json`). No setting below was recalibrated.",
        "",
        "`SpeedUp` columns are ratios against the Standard run of the *same*",
        "configuration or model variant, so each block is internally matched.",
        "",
        "## Decoding-configuration transfer",
        "",
    ]
    for key in sorted({row["configuration"] for row in transfer}):
        lines += [
            f"### {configuration_label(key)}",
            "",
            "| Task | Setting | Acc (%) | Δ Acc vs Standard | Mean NFE | NFE speedup | "
            "s/it | Wall-clock speedup |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in [r for r in transfer if r["configuration"] == key]:
            lines.append(
                f"| {row['task']} | {row['method_label']} | {fmt(row['accuracy_pct'])} "
                f"| {delta(row)} | {fmt(row['mean_nfe'])} "
                f"| {speedup(row['nfe_speedup_vs_standard'])} | {fmt(row['seconds_per_item'], 3)} "
                f"| {speedup(row['wallclock_speedup_vs_standard'])} |"
            )
        absent = [
            f"{DISPLAY[benchmark]} {METHODS[method]}"
            for benchmark, method, config in INTENTIONALLY_CANCELLED
            if config == key
        ]
        if absent:
            lines += [
                "",
                "Not run in this configuration (cancelled by the authors, deliberately"
                " not resubmitted): " + ", ".join(absent) + ". The affected"
                " accelerated rows therefore have no same-configuration Standard"
                " reference and their speedup cells are left empty rather than"
                " compared against a different configuration.",
            ]
        lines.append("")

    lines += ["## Model-variant transfer", ""]
    for variant, label in VARIANTS.items():
        rows = [row for row in variants if row["variant"] == variant]
        if not rows:
            continue
        limits = sorted({row.get("example_limit") for row in rows if row.get("example_limit")})
        note = (
            f" Evaluated on reduced example budgets ({', '.join(str(l) for l in limits)})"
            " because this variant decodes very slowly."
            if limits
            else ""
        )
        lines += [
            f"### {label}",
            "",
            f"Threshold table unchanged; tokenizer unchanged.{note}",
            "",
            "| Task | Setting | Examples | Acc (%) | Δ Acc vs Standard | Mean NFE | "
            "NFE speedup | s/it | Wall-clock speedup |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['task']} | {row['method_label']} | {row['evaluated_examples']} "
                f"| {fmt(row['accuracy_pct'])} | {delta(row)} | {fmt(row['mean_nfe'])} "
                f"| {speedup(row['nfe_speedup_vs_standard'])} | {fmt(row['seconds_per_item'], 3)} "
                f"| {speedup(row['wallclock_speedup_vs_standard'])} |"
            )
        lines.append("")

    lines += [
        "## Settings that need qualification",
        "",
        "**HumanEval under the chat template.** All three decoders collapse to single-digit",
        "pass@1 in that configuration (Standard 8.54%), so the LLaDA chat template breaks",
        "the HumanEval completion format itself. That block measures the prompting format,",
        "not threshold transfer, and should not be read as a TC-APD failure or success.",
        "",
        "**Fine-tuned checkpoint.** Reusing the base-model table on the GRPO fine-tuned",
        "checkpoint still accelerates decoding, but accuracy drops relative to Standard on",
        "all three tasks for both accelerated decoders, and more for TC-APD than for the",
        "fixed threshold. Changing model weights therefore does shift the confidence",
        "distribution the table was calibrated on, and recalibration is advisable after",
        "fine-tuning.",
        "",
        "## Interpretation boundary",
        "",
        "These results show that an existing token-level table keeps producing a",
        "speedup under changed block sizes, a longer generation budget, a chat",
        "prompting template, GPTQ INT8 weights, and a fine-tuned checkpoint that",
        "shares the base tokenizer. They do not show that recalibration is never",
        "needed: accuracy does move under the model variants, and nothing here",
        "transfers across a different backbone or a different tokenizer.",
        "",
        "## Files",
        "",
        "- `configuration_transfer_table.csv`: per configuration, task, and decoder.",
        "- `model_variant_table.csv`: fine-tuned and INT8 variants (smoke runs excluded).",
        "- `missing_runs.csv`: result directories without a complete aggregate or log.",
        "- `summary.json`: machine-readable dump of both families.",
    ]
    (args.output_dir / "P22_TRANSFER_AND_VARIANTS.md").write_text("\n".join(lines) + "\n")

    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "configuration_transfer": transfer,
                "model_variants": variants,
                "missing_runs": missing,
                "intentionally_cancelled": sorted(
                    "_".join(entry) for entry in INTENTIONALLY_CANCELLED
                ),
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(
        f"configuration_transfer={len(transfer)} model_variants={len(variants)} "
        f"missing={len(missing)} -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
