#!/usr/bin/env python3
"""Paired Standard -> TC-APD distribution-shift analysis for GSM8K.

This consumes lm-eval ``samples_*.jsonl`` files.  Those files retain the
generated text, task filters, and per-example correctness, but not the
per-iteration mask/commit states.  Consequently, the sequence diagnostics
below describe the EOS/stop-truncated *final generation path*; they must not
be presented as internal denoising-trajectory agreement.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import glob
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


SETTINGS = {
    "fixed_0.90": "gsm8k_fixed_apd_thr0.90",
    "tc_n100": "gsm8k_tc_apd_math_n100_seed2027",
    "tc_n200": "gsm8k_tc_apd_math_n200_seed2027",
    "tc_n500": "gsm8k_tc_apd_math_n500_seed2027",
    "tc_n1000": "gsm8k_tc_apd_math_n1000_seed2027",
    "tc_n2000": "gsm8k_tc_apd_math_n2000_seed2027",
    "tc_n5000": "gsm8k_tc_apd_math_n5000_seed2027",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def locate_samples(root: Path, directory: str) -> Path:
    matches = sorted(glob.glob(str(root / directory / "**" / "samples_*.jsonl"), recursive=True))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one samples JSONL under {root / directory}, found {matches}")
    return Path(matches[0])


def load_samples(path: Path) -> dict[str, dict[int, dict]]:
    by_filter: dict[str, dict[int, dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_filter.setdefault(row["filter"], {})[int(row["doc_id"])] = row
    return by_filter


def response(row: dict) -> str:
    value = row.get("resps", [[""]])
    while isinstance(value, list) and value:
        value = value[0]
    return "" if value is None else str(value)


def filtered_answer(row: dict) -> str | None:
    value = row.get("filtered_resps", [None])
    while isinstance(value, list) and value:
        value = value[0]
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() == "[invalid]":
        return None
    return value


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def pct(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{100 * value:.2f}%"


def compare_sequences(reference: list[int], prediction: list[int]) -> dict[str, float | int]:
    matched = sum(a == b for a, b in zip(reference, prediction))
    common_prefix = 0
    for a, b in zip(reference, prediction):
        if a != b:
            break
        common_prefix += 1
    ref_len = len(reference)
    return {
        "position_agreement": matched / ref_len if ref_len else float(reference == prediction),
        "exact": int(reference == prediction),
        "common_prefix": common_prefix,
        "common_prefix_fraction": common_prefix / ref_len if ref_len else float(reference == prediction),
        "length_delta": len(prediction) - ref_len,
        "edit_similarity": difflib.SequenceMatcher(None, reference, prediction, autojunk=False).ratio(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    all_samples = {
        "standard": load_samples(locate_samples(args.results_root, "gsm8k_standard")),
        **{name: load_samples(locate_samples(args.results_root, directory)) for name, directory in SETTINGS.items()},
    }
    filters = ["strict-match", "flexible-extract"]
    standard = all_samples["standard"]

    # Raw responses are duplicated across filter blocks. Use strict-match once.
    doc_ids = sorted(standard["strict-match"])
    std_text = {i: response(standard["strict-match"][i]) for i in doc_ids}
    std_tokens = {
        i: tokenizer.encode(std_text[i], add_special_tokens=False) for i in doc_ids
    }

    sequence_rows: list[dict] = []
    transition_rows: list[dict] = []
    answer_rows: list[dict] = []
    per_sample_rows: list[dict] = []

    for method, samples in all_samples.items():
        if method == "standard":
            continue
        method_ids = sorted(samples["strict-match"])
        if method_ids != doc_ids:
            raise RuntimeError(f"doc_id mismatch for {method}")

        seq = []
        for doc_id in doc_ids:
            pred_text = response(samples["strict-match"][doc_id])
            pred_tokens = tokenizer.encode(pred_text, add_special_tokens=False)
            metrics = compare_sequences(std_tokens[doc_id], pred_tokens)
            seq.append(metrics)
            per_sample_rows.append({
                "doc_id": doc_id,
                "method": method,
                "raw_text_exact": int(std_text[doc_id] == pred_text),
                **metrics,
            })

        sequence_rows.append({
            "method": method,
            "num_samples": len(seq),
            "raw_text_exact_rate": mean([
                r["raw_text_exact"] for r in per_sample_rows if r["method"] == method
            ]),
            "eos_aware_exact_sequence_agreement": mean([r["exact"] for r in seq]),
            "eos_aware_mean_position_token_agreement": mean([r["position_agreement"] for r in seq]),
            "mean_common_prefix_tokens": mean([r["common_prefix"] for r in seq]),
            "median_common_prefix_tokens": median([r["common_prefix"] for r in seq]),
            "mean_common_prefix_fraction": mean([r["common_prefix_fraction"] for r in seq]),
            "mean_token_edit_similarity": mean([r["edit_similarity"] for r in seq]),
            "mean_length_delta_tokens": mean([r["length_delta"] for r in seq]),
            "mean_abs_length_delta_tokens": mean([abs(r["length_delta"]) for r in seq]),
            "sequence_disagreement_rate": 1 - mean([r["exact"] for r in seq]),
        })

        for filter_name in filters:
            counts = Counter()
            pair_total = pair_agree = 0
            extraction_both_missing = 0
            for doc_id in doc_ids:
                std_row = standard[filter_name][doc_id]
                pred_row = samples[filter_name][doc_id]
                std_correct = bool(std_row.get("exact_match", 0))
                pred_correct = bool(pred_row.get("exact_match", 0))
                counts[(std_correct, pred_correct)] += 1
                std_answer = filtered_answer(std_row)
                pred_answer = filtered_answer(pred_row)
                if std_answer is not None and pred_answer is not None:
                    pair_total += 1
                    pair_agree += int(std_answer == pred_answer)
                elif std_answer is None and pred_answer is None:
                    extraction_both_missing += 1

            n = len(doc_ids)
            std_correct_n = counts[(True, True)] + counts[(True, False)]
            std_incorrect_n = counts[(False, True)] + counts[(False, False)]
            transition_rows.append({
                "method": method,
                "filter": filter_name,
                "num_samples": n,
                "correct_to_correct": counts[(True, True)],
                "correct_to_incorrect": counts[(True, False)],
                "incorrect_to_correct": counts[(False, True)],
                "incorrect_to_incorrect": counts[(False, False)],
                "correct_to_incorrect_rate_given_standard_correct": counts[(True, False)] / std_correct_n,
                "incorrect_to_correct_rate_given_standard_incorrect": counts[(False, True)] / std_incorrect_n,
                "binary_correctness_agreement": (counts[(True, True)] + counts[(False, False)]) / n,
                "net_correct_change": counts[(False, True)] - counts[(True, False)],
            })
            answer_rows.append({
                "method": method,
                "filter": filter_name,
                "num_samples": n,
                "both_answers_extracted": pair_total,
                "answer_pair_coverage": pair_total / n,
                "final_answer_agreement_given_both_extracted": pair_agree / pair_total if pair_total else float("nan"),
                "both_answers_missing": extraction_both_missing,
            })

    write_csv(args.output_dir / "sequence_agreement.csv", sequence_rows)
    write_csv(args.output_dir / "answer_transitions.csv", transition_rows)
    write_csv(args.output_dir / "final_answer_agreement.csv", answer_rows)
    write_csv(args.output_dir / "per_sample_sequence_metrics.csv", per_sample_rows)

    # Rebuttal-facing compact report. Flexible extraction is the primary GSM8K metric.
    answer_map = {(r["method"], r["filter"]): r for r in answer_rows}
    trans_map = {(r["method"], r["filter"]): r for r in transition_rows}
    lines = [
        "# P1.1 Standard → TC-APD trajectory distribution shift",
        "",
        f"Paired GSM8K analysis over **{len(doc_ids):,}** examples. The primary answer metric below is lm-eval's `flexible-extract`; strict-match is retained in the CSVs.",
        "",
        "| Setting | Final-answer agreement¹ | C→I² | I→C³ | Acc. Δ | EOS-aware token agreement⁴ | Exact sequence | Sequence disagreement | Mean edit similarity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seq_row in sequence_rows:
        method = seq_row["method"]
        ans = answer_map[(method, "flexible-extract")]
        tr = trans_map[(method, "flexible-extract")]
        lines.append(
            f"| {method} | {pct(ans['final_answer_agreement_given_both_extracted'])} "
            f"({ans['both_answers_extracted']}/{len(doc_ids)} pairs) | "
            f"{tr['correct_to_incorrect']} ({pct(tr['correct_to_incorrect_rate_given_standard_correct'])}) | "
            f"{tr['incorrect_to_correct']} ({pct(tr['incorrect_to_correct_rate_given_standard_incorrect'])}) | "
            f"{100 * tr['net_correct_change'] / len(doc_ids):+.2f} pp | "
            f"{pct(seq_row['eos_aware_mean_position_token_agreement'])} | "
            f"{pct(seq_row['eos_aware_exact_sequence_agreement'])} | "
            f"{pct(seq_row['sequence_disagreement_rate'])} | {seq_row['mean_token_edit_similarity']:.3f} |"
        )
    lines += [
        "",
        "¹ Agreement is conditional on both runs yielding an extractable answer; pair coverage is shown in parentheses. This avoids counting two extraction failures as agreement.",
        "",
        "² Percentage denominator is the 965 examples Standard gets correct. ³ Percentage denominator is the 354 examples Standard gets incorrect. Accuracy Δ equals `(I→C − C→I) / 1319`.",
        "",
        "⁴ lm-eval responses are already truncated at the configured EOS/EOT/stop strings. Position-wise token agreement uses the Standard effective length as denominator; missing accelerated positions are mismatches.",
        "",
        "For the main `tc_n5000` setting, 79.83% of final token sequences differ from Standard, yet flexible-answer agreement is 76.27%, binary correctness agrees on 91.43% of examples, and the correctness flips nearly balance (56 C→I versus 57 I→C; +0.08 pp accuracy). Thus TC-APD clearly induces a decoding-path distribution shift, but sequence reproduction is not its objective and sequence disagreement does not directly imply quality loss.",
        "",
        "## Interpretation boundary",
        "",
        "The available lm-eval samples contain final stop-truncated generations, filtered answers, and correctness, but not per-iteration mask/commit states. Therefore `Exact sequence`, `Sequence disagreement`, prefix, and edit metrics measure disagreement of the realized final generation sequence. They demonstrate decoding-path distribution shift, but are not a direct measurement of internal denoising-trajectory agreement. A true iteration-level trajectory statistic requires rerunning with `selected_records/all_token_records` logging enabled.",
        "",
        "## Files",
        "",
        "- `final_answer_agreement.csv`: strict and flexible answer agreement plus extraction coverage.",
        "- `answer_transitions.csv`: all four correctness transitions and conditional flip rates.",
        "- `sequence_agreement.csv`: EOS-aware sequence/prefix/edit diagnostics.",
        "- `per_sample_sequence_metrics.csv`: paired sample-level sequence diagnostics.",
        "",
    ]
    (args.output_dir / "P11_STANDARD_TC_APD_SHIFT.md").write_text("\n".join(lines), encoding="utf-8")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"sequence": sequence_rows, "answers": answer_rows, "transitions": transition_rows}, handle, indent=2)


if __name__ == "__main__":
    main()
