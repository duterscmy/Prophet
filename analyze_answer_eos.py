#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Second-stage analysis for GSM8K TC-APD experiments.

Computes:
1) Final-answer agreement with Standard decoding.
2) Correctness transitions:
      Standard correct   -> Method correct
      Standard correct   -> Method incorrect
      Standard incorrect -> Method correct
      Standard incorrect -> Method incorrect
3) EOS/EOT-aware token agreement and exact-sequence agreement.

The script streams each large log and parses ONLY selected_records,
so it does not load all_token_records or the full log into memory.

Outputs are written to:
    analysis_results/
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# CONFIG
# ============================================================

LOGS = {
    "standard": Path(
        "logs/baseline-gsm8k_standard-len256-block32.log"
    ),
    "fixed_080": Path(
        "logs/gsm8k_adaptive_parallel_len256_block32_thr0.80.log"
    ),
    "fixed_090": Path(
        "logs/gsm8k_adaptive_parallel_len256_block32_thr0.90.log"
    ),
    "tc_080": Path(
        "logs/gsm8k_dynamic_from_math_c99.5_mincount200_"
        "minaccepted100_len256_block32_maxthr0.80_minthr0.05.log"
    ),
    "tc_090": Path(
        "logs/gsm8k_dynamic_from_math_c99.5_mincount200_"
        "minaccepted100_len256_block32_maxthr0.90_minthr0.05.log"
    ),
}

OUTPUT_DIR = Path("analysis_results")

# If your tokenizer/model is stored locally, run:
# export LLADA_TOKENIZER=/path/to/LLaDA-8B-Instruct
TOKENIZER_PATH = os.environ.get(
    "LLADA_TOKENIZER",
    "GSAI-ML/LLaDA-8B-Instruct",
)

EXPECTED_NUM_SAMPLES = 1319

# These two IDs are the EOS/EOT-like IDs used in your generation code.
KNOWN_STOP_TOKEN_IDS = {
    126081,
    126348,
}

# Sanity-check only; these are your existing GSM8K numbers.
EXPECTED_PAPER_ACCURACY = {
    "standard": 0.729,
    "fixed_080": 0.730,
    "fixed_090": 0.729,
    "tc_080": 0.723,
    "tc_090": 0.721,
}


# ============================================================
# OPTIONAL FAST JSON
# ============================================================

try:
    import orjson

    def json_loads(raw):
        return orjson.loads(raw)

    print("[INFO] Using orjson.")

except ImportError:

    def json_loads(raw):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return json.loads(raw)

    print("[INFO] orjson not installed; using Python json.")
    print("[INFO] Optional speedup: pip install orjson")


# ============================================================
# HELPERS
# ============================================================

def safe_div(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return a / b


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"[WARN] No rows for {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[WRITE] {path}")


# ============================================================
# FAST LARGE-LOG PARSER
# ============================================================

def extract_selected_records_from_line(
    raw_line: bytes,
) -> Optional[List[Dict[str, Any]]]:
    """
    Extract ONLY selected_records from one huge structured JSON line:

    {
      "selected_records": [...],
      "all_token_records": [...],
      "stats": {...}
    }

    This avoids parsing the much larger all_token_records array.
    """

    selected_marker = b'"selected_records"'
    all_marker = b'"all_token_records"'

    selected_pos = raw_line.find(selected_marker)
    if selected_pos < 0:
        return None

    all_pos = raw_line.find(all_marker, selected_pos)
    if all_pos < 0:
        return None

    colon_pos = raw_line.find(b":", selected_pos)
    if colon_pos < 0:
        return None

    array_start = raw_line.find(b"[", colon_pos)
    if array_start < 0:
        return None

    # selected_records closes before "all_token_records"
    array_end = raw_line.rfind(b"]", array_start, all_pos)
    if array_end < 0:
        return None

    selected_json = raw_line[array_start:array_end + 1]

    try:
        obj = json_loads(selected_json)
    except Exception as exc:
        print(f"[WARN] Failed parsing selected_records: {exc}")
        return None

    if not isinstance(obj, list):
        return None

    return obj


def iter_selected_records(path: Path, method_name: str):
    """
    Stream selected_records sample-by-sample.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing log: {path}")

    file_size = path.stat().st_size
    sample_idx = 0

    print()
    print("=" * 80)
    print(f"[READ] {method_name}")
    print(f"       {path}")
    print(f"       {file_size / (1024 ** 3):.2f} GB")
    print("=" * 80)

    with open(path, "rb", buffering=16 * 1024 * 1024) as f:
        for raw_line in f:
            if b'"selected_records"' not in raw_line:
                continue
            if b'"all_token_records"' not in raw_line:
                continue

            records = extract_selected_records_from_line(raw_line)
            if records is None:
                continue

            yield sample_idx, records
            sample_idx += 1

            if sample_idx % 100 == 0:
                try:
                    progress = f.tell() / file_size * 100.0
                    print(
                        f"[{method_name}] {sample_idx} samples "
                        f"({progress:.1f}%)"
                    )
                except Exception:
                    print(f"[{method_name}] {sample_idx} samples")

    print(f"[DONE] {method_name}: {sample_idx} samples")


# ============================================================
# RECONSTRUCT FINAL GENERATED TOKEN SEQUENCE
# ============================================================

def selected_records_to_tokens(
    records: List[Dict[str, Any]],
) -> Tuple[List[int], int]:
    """
    Reconstruct generated tokens by generation_relative_position.

    Returns:
        contiguous token list from position 0
        number of missing positions encountered in the full mapping
    """

    mapping: Dict[int, int] = {}

    for record in records:
        if "generation_relative_position" not in record:
            continue

        try:
            pos = int(record["generation_relative_position"])
            token_id = int(record["token_id"])
        except Exception:
            continue

        mapping[pos] = token_id

    if not mapping:
        return [], 0

    max_pos = max(mapping.keys())

    missing = 0
    for pos in range(max_pos + 1):
        if pos not in mapping:
            missing += 1

    # Only retain contiguous sequence from generation position 0.
    tokens = []
    for pos in range(max_pos + 1):
        if pos not in mapping:
            break
        tokens.append(mapping[pos])

    return tokens, missing


# ============================================================
# EOS / EOT HANDLING
# ============================================================

def truncate_at_stop_token(
    tokens: List[int],
    stop_ids: set,
) -> Tuple[List[int], Optional[int], Optional[int]]:
    """
    Truncate BEFORE the first EOS/EOT token.
    """

    for i, token_id in enumerate(tokens):
        if token_id in stop_ids:
            return tokens[:i], token_id, i

    return tokens, None, None


# ============================================================
# EOS-AWARE TOKEN AGREEMENT
# ============================================================

def compare_effective_tokens(
    reference: List[int],
    prediction: List[int],
) -> Dict[str, Any]:
    """
    Reference = Standard decoding effective sequence.

    Token agreement denominator is the Standard effective length.
    Missing method positions count as mismatches.

    Exact sequence agreement requires identical effective sequences.
    """

    ref_len = len(reference)
    pred_len = len(prediction)

    matched = 0

    for i in range(ref_len):
        if i < pred_len and prediction[i] == reference[i]:
            matched += 1

    agreement = safe_div(matched, ref_len)

    exact = int(reference == prediction)

    common_prefix = 0
    for a, b in zip(reference, prediction):
        if a != b:
            break
        common_prefix += 1

    return {
        "matched_tokens": matched,
        "reference_tokens": ref_len,
        "prediction_tokens": pred_len,
        "token_agreement": agreement,
        "exact_sequence_agreement": exact,
        "common_prefix_length": common_prefix,
    }


# ============================================================
# GSM8K ANSWER EXTRACTION
# ============================================================

# Strict: "The answer is 42."
STRICT_PATTERN = re.compile(
    r"The answer is\s+(-?[\d,]+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)

# Flexible: use the last numeric expression in the decoded output.
FLEX_PATTERN = re.compile(
    r"-?\$?\d[\d,]*(?:\.\d+)?"
)


def normalize_answer(
    answer: Optional[str],
) -> Optional[str]:
    if answer is None:
        return None

    answer = str(answer).strip()
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = answer.rstrip(".").strip()

    if not answer:
        return None

    try:
        value = float(answer)
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    except Exception:
        return answer


def extract_strict_answer(
    text: str,
) -> Optional[str]:
    match = STRICT_PATTERN.search(text)
    if match is None:
        return None

    return normalize_answer(match.group(1))


def extract_flexible_answer(
    text: str,
) -> Optional[str]:
    matches = list(FLEX_PATTERN.finditer(text))

    if not matches:
        return None

    return normalize_answer(matches[-1].group(0))


def extract_gold_answer(
    dataset_answer: str,
) -> Optional[str]:
    """
    GSM8K reference ends with:
        #### 42
    """

    if dataset_answer is None:
        return None

    text = str(dataset_answer)

    if "####" in text:
        text = text.rsplit("####", 1)[-1]

    return normalize_answer(text)


# ============================================================
# TOKENIZER + DATASET
# ============================================================

def load_tokenizer():
    from transformers import AutoTokenizer

    print()
    print(f"[TOKENIZER] {TOKENIZER_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True,
    )

    stop_ids = set(KNOWN_STOP_TOKEN_IDS)

    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))

    # Add terminal tokens if tokenizer knows them.
    for token_str in [
        "</s>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|eot|>",
    ]:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_str)

            if token_id is None:
                continue

            if tokenizer.unk_token_id is not None:
                if token_id == tokenizer.unk_token_id:
                    continue

            if int(token_id) >= 0:
                stop_ids.add(int(token_id))

        except Exception:
            pass

    print(f"[TOKENIZER] Stop token IDs: {sorted(stop_ids)}")

    return tokenizer, stop_ids


def load_gsm8k():
    from datasets import load_dataset

    print()
    print("[DATASET] Loading openai/gsm8k, main, test...")

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
    )

    print(f"[DATASET] Loaded {len(dataset)} samples.")

    return dataset


# ============================================================
# LOAD ALL FINAL SEQUENCES
# ============================================================

def load_final_sequences():
    sequences = {}
    missing_position_stats = {}

    for method, path in LOGS.items():
        method_sequences = []
        total_missing = 0

        for sample_idx, records in iter_selected_records(
            path,
            method,
        ):
            tokens, missing = selected_records_to_tokens(records)

            method_sequences.append(tokens)
            total_missing += missing

        sequences[method] = method_sequences
        missing_position_stats[method] = total_missing

        print(
            f"[SEQUENCE] {method}: "
            f"{len(method_sequences)} samples, "
            f"missing positions={total_missing}"
        )

    return sequences, missing_position_stats


# ============================================================
# DECODE ALL SEQUENCES
# ============================================================

def decode_sequences(
    sequences,
    tokenizer,
    stop_ids,
):
    decoded = {}
    effective_tokens = {}
    terminal_info = {}

    for method, method_sequences in sequences.items():
        decoded[method] = []
        effective_tokens[method] = []
        terminal_info[method] = []

        for tokens in method_sequences:
            (
                truncated_tokens,
                stop_token_id,
                stop_position,
            ) = truncate_at_stop_token(
                tokens,
                stop_ids,
            )

            text = tokenizer.decode(
                truncated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            decoded[method].append(text)
            effective_tokens[method].append(truncated_tokens)

            terminal_info[method].append({
                "stop_token_id": stop_token_id,
                "stop_position": stop_position,
                "raw_length": len(tokens),
                "effective_length": len(truncated_tokens),
            })

    return decoded, effective_tokens, terminal_info


# ============================================================
# CORRECTNESS TRANSITIONS
# ============================================================

def transition_label(
    standard_correct: bool,
    method_correct: bool,
) -> str:
    if standard_correct and method_correct:
        return "correct_to_correct"

    if standard_correct and not method_correct:
        return "correct_to_incorrect"

    if not standard_correct and method_correct:
        return "incorrect_to_correct"

    return "incorrect_to_incorrect"


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("#" * 80)
    print("TC-APD GSM8K FINAL ANSWER + EOS ANALYSIS")
    print("#" * 80)

    tokenizer, stop_ids = load_tokenizer()
    dataset = load_gsm8k()

    sequences, missing_position_stats = load_final_sequences()

    sample_counts = {
        method: len(seq)
        for method, seq in sequences.items()
    }

    print()
    print("Sample counts:")
    for method, count in sample_counts.items():
        print(f"  {method:12s}: {count}")

    common_n = min(
        len(dataset),
        *sample_counts.values(),
    )

    print()
    print(f"[INFO] Using first {common_n} aligned samples.")

    if common_n != EXPECTED_NUM_SAMPLES:
        print(
            f"[WARN] Expected {EXPECTED_NUM_SAMPLES} samples "
            f"but got {common_n}."
        )

    # --------------------------------------------------------
    # Decode outputs with EOS/EOT truncation
    # --------------------------------------------------------

    print()
    print("[DECODE] Decoding generated sequences...")

    (
        decoded,
        effective_tokens,
        terminal_info,
    ) = decode_sequences(
        sequences,
        tokenizer,
        stop_ids,
    )

    # --------------------------------------------------------
    # Gold answers
    # --------------------------------------------------------

    gold_answers = [
        extract_gold_answer(
            dataset[i]["answer"]
        )
        for i in range(common_n)
    ]

    methods = [
        "standard",
        "fixed_080",
        "fixed_090",
        "tc_080",
        "tc_090",
    ]

    accelerated_methods = [
        "fixed_080",
        "fixed_090",
        "tc_080",
        "tc_090",
    ]

    # --------------------------------------------------------
    # Extract predicted final answers
    # --------------------------------------------------------

    strict_answers = {}
    flexible_answers = {}

    for method in methods:
        strict_answers[method] = []
        flexible_answers[method] = []

        for i in range(common_n):
            text = decoded[method][i]

            strict_answers[method].append(
                extract_strict_answer(text)
            )

            flexible_answers[method].append(
                extract_flexible_answer(text)
            )

    # ========================================================
    # PER-SAMPLE TABLE
    # ========================================================

    per_sample_rows = []

    for i in range(common_n):
        gold = gold_answers[i]

        standard_strict = strict_answers["standard"][i]
        standard_flexible = flexible_answers["standard"][i]

        standard_strict_correct = (
            standard_strict is not None
            and standard_strict == gold
        )

        standard_flexible_correct = (
            standard_flexible is not None
            and standard_flexible == gold
        )

        for method in methods:
            strict_answer = strict_answers[method][i]
            flexible_answer = flexible_answers[method][i]

            strict_correct = (
                strict_answer is not None
                and strict_answer == gold
            )

            flexible_correct = (
                flexible_answer is not None
                and flexible_answer == gold
            )

            eos_result = compare_effective_tokens(
                effective_tokens["standard"][i],
                effective_tokens[method][i],
            )

            strict_answer_agreement = (
                strict_answer is not None
                and standard_strict is not None
                and strict_answer == standard_strict
            )

            flexible_answer_agreement = (
                flexible_answer is not None
                and standard_flexible is not None
                and flexible_answer == standard_flexible
            )

            row = {
                "sample_idx": i,
                "method": method,

                "gold_answer": gold,

                "strict_answer": strict_answer,
                "flexible_answer": flexible_answer,

                "strict_answer_extracted":
                    int(strict_answer is not None),

                "flexible_answer_extracted":
                    int(flexible_answer is not None),

                "strict_correct":
                    int(strict_correct),

                "flexible_correct":
                    int(flexible_correct),

                "strict_answer_agreement_with_standard":
                    int(strict_answer_agreement),

                "flexible_answer_agreement_with_standard":
                    int(flexible_answer_agreement),

                "eos_aware_token_agreement":
                    eos_result["token_agreement"],

                "eos_aware_exact_sequence_agreement":
                    eos_result["exact_sequence_agreement"],

                "eos_aware_common_prefix_length":
                    eos_result["common_prefix_length"],

                "standard_effective_length":
                    eos_result["reference_tokens"],

                "method_effective_length":
                    eos_result["prediction_tokens"],

                "stop_token_id":
                    terminal_info[method][i]["stop_token_id"],

                "stop_position":
                    terminal_info[method][i]["stop_position"],

                "raw_generation_length":
                    terminal_info[method][i]["raw_length"],
            }

            if method == "standard":
                row["strict_transition"] = None
                row["flexible_transition"] = None
            else:
                row["strict_transition"] = transition_label(
                    standard_strict_correct,
                    strict_correct,
                )

                row["flexible_transition"] = transition_label(
                    standard_flexible_correct,
                    flexible_correct,
                )

            per_sample_rows.append(row)

    write_csv(
        OUTPUT_DIR / "answer_eos_per_sample.csv",
        per_sample_rows,
    )

    # ========================================================
    # METHOD SUMMARY
    # ========================================================

    summary_rows = []

    for method in methods:
        rows = [
            r for r in per_sample_rows
            if r["method"] == method
        ]

        n = len(rows)

        strict_extracted = sum(
            r["strict_answer_extracted"]
            for r in rows
        )

        flexible_extracted = sum(
            r["flexible_answer_extracted"]
            for r in rows
        )

        strict_correct = sum(
            r["strict_correct"]
            for r in rows
        )

        flexible_correct = sum(
            r["flexible_correct"]
            for r in rows
        )

        eos_exact = sum(
            r["eos_aware_exact_sequence_agreement"]
            for r in rows
        )

        eos_token_agreements = [
            r["eos_aware_token_agreement"]
            for r in rows
            if (
                r["eos_aware_token_agreement"] is not None
                and not math.isnan(
                    r["eos_aware_token_agreement"]
                )
            )
        ]

        # ----------------------------------------------------
        # Final-answer agreement with Standard
        # Only count samples where both answers are extracted.
        # ----------------------------------------------------

        strict_pair_total = 0
        strict_pair_agree = 0

        flexible_pair_total = 0
        flexible_pair_agree = 0

        for i in range(common_n):
            std_s = strict_answers["standard"][i]
            pred_s = strict_answers[method][i]

            if std_s is not None and pred_s is not None:
                strict_pair_total += 1
                strict_pair_agree += int(std_s == pred_s)

            std_f = flexible_answers["standard"][i]
            pred_f = flexible_answers[method][i]

            if std_f is not None and pred_f is not None:
                flexible_pair_total += 1
                flexible_pair_agree += int(std_f == pred_f)

        terminated = sum(
            terminal_info[method][i]["stop_token_id"] is not None
            for i in range(common_n)
        )

        mean_effective_len = float(
            np.mean([
                terminal_info[method][i]["effective_length"]
                for i in range(common_n)
            ])
        )

        expected = EXPECTED_PAPER_ACCURACY.get(method)

        strict_accuracy = safe_div(
            strict_correct,
            n,
        )

        flexible_accuracy = safe_div(
            flexible_correct,
            n,
        )

        summary_rows.append({
            "method":
                method,

            "num_samples":
                n,

            "strict_extract_rate":
                safe_div(
                    strict_extracted,
                    n,
                ),

            "flexible_extract_rate":
                safe_div(
                    flexible_extracted,
                    n,
                ),

            "strict_accuracy":
                strict_accuracy,

            "flexible_accuracy":
                flexible_accuracy,

            "expected_paper_accuracy":
                expected,

            "strict_minus_expected":
                (
                    strict_accuracy - expected
                    if expected is not None
                    else None
                ),

            "flexible_minus_expected":
                (
                    flexible_accuracy - expected
                    if expected is not None
                    else None
                ),

            "strict_answer_agreement_with_standard":
                safe_div(
                    strict_pair_agree,
                    strict_pair_total,
                ),

            "strict_answer_pair_coverage":
                safe_div(
                    strict_pair_total,
                    n,
                ),

            "flexible_answer_agreement_with_standard":
                safe_div(
                    flexible_pair_agree,
                    flexible_pair_total,
                ),

            "flexible_answer_pair_coverage":
                safe_div(
                    flexible_pair_total,
                    n,
                ),

            "eos_aware_mean_token_agreement":
                (
                    float(
                        np.mean(
                            eos_token_agreements
                        )
                    )
                    if eos_token_agreements
                    else None
                ),

            "eos_aware_exact_sequence_agreement":
                safe_div(
                    eos_exact,
                    n,
                ),

            "terminal_token_found_rate":
                safe_div(
                    terminated,
                    n,
                ),

            "mean_effective_generation_length":
                mean_effective_len,

            "missing_generation_positions":
                missing_position_stats[
                    method
                ],
        })

    write_csv(
        OUTPUT_DIR / "answer_eos_summary.csv",
        summary_rows,
    )

    # ========================================================
    # CORRECTNESS TRANSITIONS
    # ========================================================

    transition_rows = []

    for method in accelerated_methods:
        rows = [
            r for r in per_sample_rows
            if r["method"] == method
        ]

        for metric_name in [
            "strict",
            "flexible",
        ]:
            key = f"{metric_name}_transition"

            counts = {
                "correct_to_correct": 0,
                "correct_to_incorrect": 0,
                "incorrect_to_correct": 0,
                "incorrect_to_incorrect": 0,
            }

            for row in rows:
                label = row[key]

                if label in counts:
                    counts[label] += 1

            n = len(rows)

            standard_correct_total = (
                counts["correct_to_correct"]
                + counts["correct_to_incorrect"]
            )

            standard_incorrect_total = (
                counts["incorrect_to_correct"]
                + counts["incorrect_to_incorrect"]
            )

            transition_rows.append({
                "method":
                    method,

                "metric":
                    metric_name,

                "num_samples":
                    n,

                "correct_to_correct":
                    counts["correct_to_correct"],

                "correct_to_incorrect":
                    counts["correct_to_incorrect"],

                "incorrect_to_correct":
                    counts["incorrect_to_correct"],

                "incorrect_to_incorrect":
                    counts["incorrect_to_incorrect"],

                "correct_to_incorrect_rate_given_standard_correct":
                    safe_div(
                        counts["correct_to_incorrect"],
                        standard_correct_total,
                    ),

                "incorrect_to_correct_rate_given_standard_incorrect":
                    safe_div(
                        counts["incorrect_to_correct"],
                        standard_incorrect_total,
                    ),

                "net_correct_change":
                    (
                        counts["incorrect_to_correct"]
                        - counts["correct_to_incorrect"]
                    ),
            })

    write_csv(
        OUTPUT_DIR / "answer_transitions.csv",
        transition_rows,
    )

    # ========================================================
    # COMPACT JSON
    # ========================================================

    output_json = {
        "summary": summary_rows,
        "transitions": transition_rows,
        "sample_count": common_n,
        "stop_token_ids": sorted(stop_ids),
        "tokenizer": TOKENIZER_PATH,
        "expected_paper_accuracy":
            EXPECTED_PAPER_ACCURACY,
    }

    with open(
        OUTPUT_DIR / "answer_eos_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output_json,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "[WRITE] "
        "analysis_results/answer_eos_summary.json"
    )

    # ========================================================
    # HUMAN-READABLE REPORT
    # ========================================================

    report_path = (
        OUTPUT_DIR
        / "answer_eos_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "TC-APD GSM8K FINAL ANSWER + EOS ANALYSIS\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            f"Samples: {common_n}\n"
        )

        f.write(
            "Stop token IDs: "
            f"{sorted(stop_ids)}\n\n"
        )

        f.write(
            "METHOD SUMMARY\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        for row in summary_rows:

            f.write(
                f"\n[{row['method']}]\n"
            )

            f.write(
                "  strict accuracy: "
                f"{row['strict_accuracy']:.6f}\n"
            )

            f.write(
                "  flexible accuracy: "
                f"{row['flexible_accuracy']:.6f}\n"
            )

            f.write(
                "  expected paper accuracy: "
                f"{row['expected_paper_accuracy']:.6f}\n"
            )

            f.write(
                "  strict extract rate: "
                f"{row['strict_extract_rate']:.6f}\n"
            )

            f.write(
                "  flexible extract rate: "
                f"{row['flexible_extract_rate']:.6f}\n"
            )

            f.write(
                "  strict final-answer agreement "
                "with Standard: "
                f"{row['strict_answer_agreement_with_standard']:.6f}\n"
            )

            f.write(
                "  flexible final-answer agreement "
                "with Standard: "
                f"{row['flexible_answer_agreement_with_standard']:.6f}\n"
            )

            f.write(
                "  EOS-aware mean token agreement: "
                f"{row['eos_aware_mean_token_agreement']:.6f}\n"
            )

            f.write(
                "  EOS-aware exact sequence agreement: "
                f"{row['eos_aware_exact_sequence_agreement']:.6f}\n"
            )

            f.write(
                "  terminal token found rate: "
                f"{row['terminal_token_found_rate']:.6f}\n"
            )

            f.write(
                "  mean effective generation length: "
                f"{row['mean_effective_generation_length']:.3f}\n"
            )

        f.write(
            "\n\nCORRECTNESS TRANSITIONS\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        for row in transition_rows:

            f.write(
                f"\n[{row['method']} / "
                f"{row['metric']}]\n"
            )

            f.write(
                "  correct -> correct: "
                f"{row['correct_to_correct']}\n"
            )

            f.write(
                "  correct -> incorrect: "
                f"{row['correct_to_incorrect']}\n"
            )

            f.write(
                "  incorrect -> correct: "
                f"{row['incorrect_to_correct']}\n"
            )

            f.write(
                "  incorrect -> incorrect: "
                f"{row['incorrect_to_incorrect']}\n"
            )

            f.write(
                "  C->I rate given Standard correct: "
                f"{row['correct_to_incorrect_rate_given_standard_correct']:.6f}\n"
            )

            f.write(
                "  I->C rate given Standard incorrect: "
                f"{row['incorrect_to_correct_rate_given_standard_incorrect']:.6f}\n"
            )

            f.write(
                "  net correct change: "
                f"{row['net_correct_change']}\n"
            )

    print(f"[WRITE] {report_path}")

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print()
    print("#" * 80)
    print("ANALYSIS COMPLETE")
    print("#" * 80)

    print()
    print("Generated files:")

    for filename in [
        "answer_eos_report.txt",
        "answer_eos_summary.csv",
        "answer_transitions.csv",
        "answer_eos_per_sample.csv",
        "answer_eos_summary.json",
    ]:
        print(
            f"  analysis_results/{filename}"
        )

    print()
    print(
        "Send me these four files:"
    )

    print(
        "  analysis_results/answer_eos_report.txt"
    )

    print(
        "  analysis_results/answer_eos_summary.csv"
    )

    print(
        "  analysis_results/answer_transitions.csv"
    )

    print(
        "  analysis_results/answer_eos_summary.json"
    )


if __name__ == "__main__":
    main()
