#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exact lm-eval-harness GSM8K rescoring for the five existing logs.

Purpose
-------
Reconstruct the final generated text from selected_records, then use the
INSTALLED lm-eval-harness implementation itself for:

1. gsm8k_cot_zeroshot strict-match extraction
2. gsm8k_cot_zeroshot flexible-extract extraction
3. lm-eval exact_match scoring with the task's official normalization
4. final-answer agreement with Standard
5. correctness transitions:
      Standard correct   -> Method correct
      Standard correct   -> Method incorrect
      Standard incorrect -> Method correct
      Standard incorrect -> Method incorrect

This script intentionally imports:
    lm_eval.filters.extraction.RegexFilter
    lm_eval.api.metrics.exact_match_hf_evaluate

so the scoring behavior comes from the lm-eval-harness installed on the
server rather than from a hand-written approximation.

Large-log behavior
------------------
Only selected_records is parsed from each large JSON line.
all_token_records is NOT deserialized, so memory use stays small.

Outputs
-------
All outputs are written to the existing:
    analysis_results/

Files:
    official_gsm8k_summary.csv
    official_gsm8k_transitions.csv
    official_gsm8k_per_sample.csv
    official_gsm8k_report.txt
    official_gsm8k_summary.json
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
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

TOKENIZER_PATH = os.environ.get(
    "LLADA_TOKENIZER",
    "GSAI-ML/LLaDA-8B-Instruct",
)

EXPECTED_NUM_SAMPLES = 1319

# Only a sanity check against your paper table.
EXPECTED_PAPER_ACCURACY = {
    "standard": 0.729,
    "fixed_080": 0.730,
    "fixed_090": 0.729,
    "tc_080": 0.723,
    "tc_090": 0.721,
}

# These IDs appear explicitly in your LLaDA generation code.
KNOWN_STOP_TOKEN_IDS = {
    126081,
    126348,
}

# gsm8k_cot_zeroshot generation_kwargs.until
TEXT_UNTIL = [
    "Q:",
    "</s>",
    "<|im_end|>",
]


# ============================================================
# IMPORT THE INSTALLED LM-EVAL IMPLEMENTATION
# ============================================================

try:
    import lm_eval
    from lm_eval.filters.extraction import RegexFilter
    from lm_eval.api.metrics import exact_match_hf_evaluate
except Exception as exc:
    print(
        "\n[ERROR] Could not import the installed lm-eval-harness scoring code.\n"
        "Make sure this script is run in the same environment used for evaluation.\n"
        f"Original import error: {exc}\n",
        file=sys.stderr,
    )
    raise


def get_lm_eval_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("lm_eval")
    except Exception:
        try:
            import importlib.metadata
            return importlib.metadata.version("lm-eval")
        except Exception:
            return "unknown"


LM_EVAL_VERSION = get_lm_eval_version()


# ============================================================
# OFFICIAL GSM8K_COT_ZEROSHOT SETTINGS
# ============================================================

# From gsm8k-cot-zeroshot.yaml
STRICT_REGEX = r"The answer is (\-?[0-9\.\,]+)."

FLEXIBLE_REGEX = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"

EXACT_MATCH_KWARGS = {
    "ignore_case": True,
    "ignore_punctuation": False,
    "regexes_to_ignore": [
        ",",
        r"\$",
        r"(?s).*#### ",
        r"\.$",
    ],
}

STRICT_FILTER = RegexFilter(
    regex_pattern=STRICT_REGEX,
    group_select=0,
)

FLEXIBLE_FILTER = RegexFilter(
    regex_pattern=FLEXIBLE_REGEX,
    group_select=-1,
)


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
# GENERIC HELPERS
# ============================================================

def safe_div(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return a / b


def safe_json_value(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"[WARN] No rows for {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
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
# LARGE LOG PARSER: SELECTED_RECORDS ONLY
# ============================================================

def extract_selected_records_from_line(
    raw_line: bytes,
) -> Optional[List[Dict[str, Any]]]:
    """
    Extract only selected_records from:

      {
        "selected_records": [...],
        "all_token_records": [...],
        "stats": {...}
      }

    The huge all_token_records array is never parsed.
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

    array_end = raw_line.rfind(
        b"]",
        array_start,
        all_pos,
    )
    if array_end < 0:
        return None

    selected_json = raw_line[
        array_start:array_end + 1
    ]

    try:
        obj = json_loads(selected_json)
    except Exception as exc:
        print(
            f"[WARN] Failed to parse selected_records: {exc}"
        )
        return None

    if not isinstance(obj, list):
        return None

    return obj


def iter_selected_records(
    path: Path,
    method_name: str,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing log: {path}"
        )

    file_size = path.stat().st_size
    sample_idx = 0

    print()
    print("=" * 80)
    print(f"[READ] {method_name}")
    print(f"       {path}")
    print(
        f"       {file_size / (1024 ** 3):.2f} GB"
    )
    print("=" * 80)

    with open(
        path,
        "rb",
        buffering=16 * 1024 * 1024,
    ) as f:
        for raw_line in f:

            if b'"selected_records"' not in raw_line:
                continue

            if b'"all_token_records"' not in raw_line:
                continue

            records = extract_selected_records_from_line(
                raw_line
            )

            if records is None:
                continue

            yield sample_idx, records
            sample_idx += 1

            if sample_idx % 100 == 0:
                try:
                    progress = (
                        f.tell() / file_size * 100.0
                    )
                    print(
                        f"[{method_name}] "
                        f"{sample_idx} samples "
                        f"({progress:.1f}%)"
                    )
                except Exception:
                    print(
                        f"[{method_name}] "
                        f"{sample_idx} samples"
                    )

    print(
        f"[DONE] {method_name}: "
        f"{sample_idx} samples"
    )


# ============================================================
# RECONSTRUCT FINAL TOKEN SEQUENCES
# ============================================================

def selected_records_to_tokens(
    records: List[Dict[str, Any]],
) -> Tuple[List[int], int]:

    mapping: Dict[int, int] = {}

    for record in records:
        try:
            pos = int(
                record[
                    "generation_relative_position"
                ]
            )
            token_id = int(
                record["token_id"]
            )
        except Exception:
            continue

        mapping[pos] = token_id

    if not mapping:
        return [], 0

    max_pos = max(mapping)

    missing = sum(
        int(pos not in mapping)
        for pos in range(max_pos + 1)
    )

    # Keep only the contiguous generated sequence.
    tokens: List[int] = []

    for pos in range(max_pos + 1):
        if pos not in mapping:
            break

        tokens.append(
            mapping[pos]
        )

    return tokens, missing


# ============================================================
# TERMINATION + DECODING
# ============================================================

def load_tokenizer():
    from transformers import AutoTokenizer

    print()
    print(
        f"[TOKENIZER] {TOKENIZER_PATH}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True,
    )

    stop_ids = set(
        KNOWN_STOP_TOKEN_IDS
    )

    if tokenizer.eos_token_id is not None:
        stop_ids.add(
            int(tokenizer.eos_token_id)
        )

    for token_str in [
        "</s>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|eot|>",
    ]:
        try:
            token_id = (
                tokenizer.convert_tokens_to_ids(
                    token_str
                )
            )

            if token_id is None:
                continue

            if (
                tokenizer.unk_token_id is not None
                and
                token_id == tokenizer.unk_token_id
            ):
                continue

            if int(token_id) >= 0:
                stop_ids.add(
                    int(token_id)
                )

        except Exception:
            pass

    print(
        f"[TOKENIZER] stop IDs = "
        f"{sorted(stop_ids)}"
    )

    return tokenizer, stop_ids


def truncate_at_stop_id(
    tokens: List[int],
    stop_ids: set,
) -> List[int]:

    for i, token_id in enumerate(tokens):
        if token_id in stop_ids:
            return tokens[:i]

    return tokens


def trim_at_until_strings(
    text: str,
) -> str:

    positions = []

    for stop in TEXT_UNTIL:
        pos = text.find(stop)

        if pos >= 0:
            positions.append(pos)

    if positions:
        text = text[:min(positions)]

    return text


def decode_output(
    tokens: List[int],
    tokenizer,
    stop_ids: set,
) -> str:

    tokens = truncate_at_stop_id(
        tokens,
        stop_ids,
    )

    text = tokenizer.decode(
        tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    text = trim_at_until_strings(
        text
    )

    return text


# ============================================================
# USE LM-EVAL'S FILTER IMPLEMENTATION ITSELF
# ============================================================

def apply_regex_filter(
    filter_obj: RegexFilter,
    text: str,
) -> str:
    """
    lm-eval's RegexFilter expects:
        resps = Iterable[Sequence[str]]

    For one sample with one generation:
        [[text]]

    It returns:
        [[filtered_answer]]

    gsm8k then applies take_first, which is simply
    the first element of the inner list here.
    """

    result = filter_obj.apply(
        [[text]],
        [{}],
    )

    result = list(result)

    if not result:
        return "[invalid]"

    if not result[0]:
        return "[invalid]"

    return str(result[0][0])


def official_extract(
    text: str,
    filter_name: str,
) -> str:

    if filter_name == "strict-match":
        return apply_regex_filter(
            STRICT_FILTER,
            text,
        )

    if filter_name == "flexible-extract":
        return apply_regex_filter(
            FLEXIBLE_FILTER,
            text,
        )

    raise ValueError(
        f"Unknown filter: {filter_name}"
    )


# ============================================================
# USE LM-EVAL'S EXACT-MATCH IMPLEMENTATION ITSELF
# ============================================================

def official_exact_match(
    prediction: str,
    reference: str,
) -> bool:

    result = exact_match_hf_evaluate(
        predictions=[prediction],
        references=[reference],
        **EXACT_MATCH_KWARGS,
    )

    return bool(
        float(result["exact_match"]) > 0.5
    )


def meaningful_answer_agreement(
    standard_answer: str,
    method_answer: str,
) -> Optional[bool]:
    """
    Compare extracted answers using the same exact-match normalizer.

    If either side is [invalid], return None rather than treating
    two extraction failures as agreement.
    """

    if (
        standard_answer == "[invalid]"
        or
        method_answer == "[invalid]"
    ):
        return None

    return official_exact_match(
        prediction=method_answer,
        reference=standard_answer,
    )


# ============================================================
# DATASET
# ============================================================

def load_gsm8k():
    from datasets import load_dataset

    print()
    print(
        "[DATASET] Loading "
        "openai/gsm8k / main / test"
    )

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
    )

    print(
        f"[DATASET] {len(dataset)} samples"
    )

    return dataset


# ============================================================
# LOAD / DECODE ALL FIVE METHODS
# ============================================================

def load_decoded_outputs(
    tokenizer,
    stop_ids,
):

    decoded_outputs: Dict[str, List[str]] = {}
    missing_stats: Dict[str, int] = {}

    for method, path in LOGS.items():

        outputs: List[str] = []
        missing_total = 0

        for _, records in iter_selected_records(
            path,
            method,
        ):

            tokens, missing = (
                selected_records_to_tokens(
                    records
                )
            )

            missing_total += missing

            outputs.append(
                decode_output(
                    tokens,
                    tokenizer,
                    stop_ids,
                )
            )

        decoded_outputs[
            method
        ] = outputs

        missing_stats[
            method
        ] = missing_total

        print(
            f"[OUTPUT] {method}: "
            f"{len(outputs)} samples; "
            f"missing positions={missing_total}"
        )

    return decoded_outputs, missing_stats


# ============================================================
# CORRECTNESS TRANSITIONS
# ============================================================

def transition_label(
    standard_correct: bool,
    method_correct: bool,
) -> str:

    if (
        standard_correct
        and
        method_correct
    ):
        return "correct_to_correct"

    if (
        standard_correct
        and
        not method_correct
    ):
        return "correct_to_incorrect"

    if (
        not standard_correct
        and
        method_correct
    ):
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
    print(
        "OFFICIAL LM-EVAL GSM8K RESCORING"
    )
    print("#" * 80)
    print(
        f"lm-eval version: "
        f"{LM_EVAL_VERSION}"
    )

    tokenizer, stop_ids = (
        load_tokenizer()
    )

    dataset = load_gsm8k()

    (
        decoded_outputs,
        missing_stats,
    ) = load_decoded_outputs(
        tokenizer,
        stop_ids,
    )

    counts = {
        method: len(outputs)
        for method, outputs
        in decoded_outputs.items()
    }

    print()
    print("Sample counts:")

    for method, count in counts.items():
        print(
            f"  {method:12s}: {count}"
        )

    common_n = min(
        len(dataset),
        *counts.values(),
    )

    print()
    print(
        f"[INFO] Aligned sample count: "
        f"{common_n}"
    )

    if common_n != EXPECTED_NUM_SAMPLES:
        print(
            f"[WARN] Expected "
            f"{EXPECTED_NUM_SAMPLES}, "
            f"but using {common_n}."
        )

    methods = list(
        LOGS.keys()
    )

    accelerated_methods = [
        "fixed_080",
        "fixed_090",
        "tc_080",
        "tc_090",
    ]

    filters = [
        "strict-match",
        "flexible-extract",
    ]

    # --------------------------------------------------------
    # Extract all answers using lm-eval RegexFilter
    # --------------------------------------------------------

    extracted: Dict[
        str,
        Dict[str, List[str]]
    ] = {}

    for filter_name in filters:

        extracted[
            filter_name
        ] = {}

        for method in methods:

            answers: List[str] = []

            for i in range(common_n):

                answer = official_extract(
                    decoded_outputs[
                        method
                    ][i],
                    filter_name,
                )

                answers.append(
                    answer
                )

            extracted[
                filter_name
            ][method] = answers

    # --------------------------------------------------------
    # Score correctness using lm-eval exact_match
    # --------------------------------------------------------

    correctness: Dict[
        str,
        Dict[str, List[bool]]
    ] = {}

    for filter_name in filters:

        correctness[
            filter_name
        ] = {}

        for method in methods:

            scores: List[bool] = []

            for i in range(common_n):

                prediction = extracted[
                    filter_name
                ][method][i]

                # IMPORTANT:
                # gsm8k_cot_zeroshot doc_to_target is "{{answer}}",
                # i.e. the full GSM8K reference string.
                reference = str(
                    dataset[i]["answer"]
                )

                score = official_exact_match(
                    prediction=prediction,
                    reference=reference,
                )

                scores.append(
                    score
                )

            correctness[
                filter_name
            ][method] = scores

    # ========================================================
    # PER-SAMPLE TABLE
    # ========================================================

    per_sample_rows: List[
        Dict[str, Any]
    ] = []

    for i in range(common_n):

        for method in methods:

            row: Dict[str, Any] = {
                "sample_idx": i,
                "method": method,
            }

            for filter_name in filters:

                short = (
                    "strict"
                    if filter_name
                    == "strict-match"
                    else "flexible"
                )

                method_answer = extracted[
                    filter_name
                ][method][i]

                standard_answer = extracted[
                    filter_name
                ]["standard"][i]

                method_correct = correctness[
                    filter_name
                ][method][i]

                standard_correct = correctness[
                    filter_name
                ]["standard"][i]

                agreement = (
                    meaningful_answer_agreement(
                        standard_answer,
                        method_answer,
                    )
                )

                row[
                    f"{short}_answer"
                ] = method_answer

                row[
                    f"{short}_correct"
                ] = int(
                    method_correct
                )

                row[
                    f"{short}_answer_valid"
                ] = int(
                    method_answer
                    != "[invalid]"
                )

                row[
                    f"{short}_answer_agreement_with_standard"
                ] = (
                    None
                    if agreement is None
                    else int(agreement)
                )

                if method == "standard":

                    row[
                        f"{short}_transition"
                    ] = None

                else:

                    row[
                        f"{short}_transition"
                    ] = transition_label(
                        standard_correct,
                        method_correct,
                    )

            per_sample_rows.append(
                row
            )

    write_csv(
        OUTPUT_DIR
        / "official_gsm8k_per_sample.csv",
        per_sample_rows,
    )

    # ========================================================
    # METHOD SUMMARY
    # ========================================================

    summary_rows: List[
        Dict[str, Any]
    ] = []

    for method in methods:

        row: Dict[str, Any] = {
            "method": method,
            "num_samples": common_n,
            "expected_paper_accuracy":
                EXPECTED_PAPER_ACCURACY.get(
                    method
                ),
            "missing_generation_positions":
                missing_stats.get(
                    method,
                    0
                ),
        }

        for filter_name in filters:

            short = (
                "strict"
                if filter_name
                == "strict-match"
                else "flexible"
            )

            scores = correctness[
                filter_name
            ][method]

            answers = extracted[
                filter_name
            ][method]

            accuracy = float(
                np.mean(scores)
            )

            valid_count = sum(
                a != "[invalid]"
                for a in answers
            )

            # Final-answer agreement with Standard.
            agreement_values = []

            if method == "standard":

                # trivially 1.0 on valid answers
                for answer in answers:
                    if answer != "[invalid]":
                        agreement_values.append(
                            True
                        )

            else:

                standard_answers = extracted[
                    filter_name
                ]["standard"]

                for standard_answer, method_answer in zip(
                    standard_answers,
                    answers,
                ):

                    agreement = (
                        meaningful_answer_agreement(
                            standard_answer,
                            method_answer,
                        )
                    )

                    if agreement is not None:
                        agreement_values.append(
                            agreement
                        )

            agreement_rate = (
                float(
                    np.mean(
                        agreement_values
                    )
                )
                if agreement_values
                else float("nan")
            )

            pair_coverage = safe_div(
                len(agreement_values),
                common_n,
            )

            expected = (
                EXPECTED_PAPER_ACCURACY.get(
                    method
                )
            )

            row[
                f"{short}_accuracy"
            ] = accuracy

            row[
                f"{short}_minus_expected"
            ] = (
                accuracy - expected
                if expected is not None
                else None
            )

            row[
                f"{short}_valid_extract_rate"
            ] = safe_div(
                valid_count,
                common_n,
            )

            row[
                f"{short}_final_answer_agreement_with_standard"
            ] = agreement_rate

            row[
                f"{short}_answer_pair_coverage"
            ] = pair_coverage

        summary_rows.append(
            row
        )

    write_csv(
        OUTPUT_DIR
        / "official_gsm8k_summary.csv",
        summary_rows,
    )

    # ========================================================
    # CORRECTNESS TRANSITIONS
    # ========================================================

    transition_rows: List[
        Dict[str, Any]
    ] = []

    for method in accelerated_methods:

        for filter_name in filters:

            short = (
                "strict"
                if filter_name
                == "strict-match"
                else "flexible"
            )

            std_scores = correctness[
                filter_name
            ]["standard"]

            method_scores = correctness[
                filter_name
            ][method]

            counts_dict = {
                "correct_to_correct": 0,
                "correct_to_incorrect": 0,
                "incorrect_to_correct": 0,
                "incorrect_to_incorrect": 0,
            }

            for standard_correct, method_correct in zip(
                std_scores,
                method_scores,
            ):

                label = transition_label(
                    standard_correct,
                    method_correct,
                )

                counts_dict[
                    label
                ] += 1

            std_correct_n = (
                counts_dict[
                    "correct_to_correct"
                ]
                +
                counts_dict[
                    "correct_to_incorrect"
                ]
            )

            std_incorrect_n = (
                counts_dict[
                    "incorrect_to_correct"
                ]
                +
                counts_dict[
                    "incorrect_to_incorrect"
                ]
            )

            transition_rows.append({
                "method":
                    method,

                "filter":
                    filter_name,

                "num_samples":
                    common_n,

                "correct_to_correct":
                    counts_dict[
                        "correct_to_correct"
                    ],

                "correct_to_incorrect":
                    counts_dict[
                        "correct_to_incorrect"
                    ],

                "incorrect_to_correct":
                    counts_dict[
                        "incorrect_to_correct"
                    ],

                "incorrect_to_incorrect":
                    counts_dict[
                        "incorrect_to_incorrect"
                    ],

                "correct_to_incorrect_rate_given_standard_correct":
                    safe_div(
                        counts_dict[
                            "correct_to_incorrect"
                        ],
                        std_correct_n,
                    ),

                "incorrect_to_correct_rate_given_standard_incorrect":
                    safe_div(
                        counts_dict[
                            "incorrect_to_correct"
                        ],
                        std_incorrect_n,
                    ),

                "net_correct_change":
                    (
                        counts_dict[
                            "incorrect_to_correct"
                        ]
                        -
                        counts_dict[
                            "correct_to_incorrect"
                        ]
                    ),

                "standard_accuracy":
                    float(
                        np.mean(
                            std_scores
                        )
                    ),

                "method_accuracy":
                    float(
                        np.mean(
                            method_scores
                        )
                    ),
            })

    write_csv(
        OUTPUT_DIR
        / "official_gsm8k_transitions.csv",
        transition_rows,
    )

    # ========================================================
    # WHICH FILTER BEST MATCHES THE PAPER?
    # ========================================================

    paper_match_rows = []

    for filter_name in filters:

        short = (
            "strict"
            if filter_name
            == "strict-match"
            else "flexible"
        )

        abs_diffs = []

        for row in summary_rows:

            expected = row[
                "expected_paper_accuracy"
            ]

            observed = row[
                f"{short}_accuracy"
            ]

            if expected is not None:
                abs_diffs.append(
                    abs(
                        observed
                        - expected
                    )
                )

        paper_match_rows.append({
            "filter": filter_name,
            "mean_absolute_difference_from_paper":
                float(
                    np.mean(abs_diffs)
                )
                if abs_diffs
                else None,
            "max_absolute_difference_from_paper":
                float(
                    np.max(abs_diffs)
                )
                if abs_diffs
                else None,
        })

    paper_match_rows.sort(
        key=lambda x: (
            float("inf")
            if x[
                "mean_absolute_difference_from_paper"
            ] is None
            else x[
                "mean_absolute_difference_from_paper"
            ]
        )
    )

    best_filter = (
        paper_match_rows[0]["filter"]
        if paper_match_rows
        else None
    )

    # ========================================================
    # JSON
    # ========================================================

    json_output = {
        "lm_eval_version":
            LM_EVAL_VERSION,

        "tokenizer":
            TOKENIZER_PATH,

        "sample_count":
            common_n,

        "official_task_settings": {
            "strict_regex":
                STRICT_REGEX,

            "flexible_regex":
                FLEXIBLE_REGEX,

            "exact_match_kwargs":
                EXACT_MATCH_KWARGS,

            "until":
                TEXT_UNTIL,
        },

        "best_matching_filter_to_paper":
            best_filter,

        "paper_match_diagnostics":
            paper_match_rows,

        "summary":
            summary_rows,

        "transitions":
            transition_rows,
    }

    with open(
        OUTPUT_DIR
        / "official_gsm8k_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_output,
            f,
            ensure_ascii=False,
            indent=2,
            default=safe_json_value,
        )

    print(
        "[WRITE] "
        "analysis_results/"
        "official_gsm8k_summary.json"
    )

    # ========================================================
    # HUMAN-READABLE REPORT
    # ========================================================

    report_path = (
        OUTPUT_DIR
        / "official_gsm8k_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "OFFICIAL LM-EVAL GSM8K RESCORING\n"
        )
        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            f"lm-eval version: "
            f"{LM_EVAL_VERSION}\n"
        )

        f.write(
            f"samples: {common_n}\n"
        )

        f.write(
            f"tokenizer: "
            f"{TOKENIZER_PATH}\n\n"
        )

        f.write(
            "FILTER VS PAPER SANITY CHECK\n"
        )
        f.write(
            "-" * 80
            + "\n"
        )

        for row in paper_match_rows:

            f.write(
                f"{row['filter']}: "
                f"mean |delta| = "
                f"{row['mean_absolute_difference_from_paper']:.6f}, "
                f"max |delta| = "
                f"{row['max_absolute_difference_from_paper']:.6f}\n"
            )

        f.write(
            f"\nBest matching filter: "
            f"{best_filter}\n\n"
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
                "  expected paper accuracy: "
                f"{row['expected_paper_accuracy']:.6f}\n"
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
                "  strict valid extraction: "
                f"{row['strict_valid_extract_rate']:.6f}\n"
            )

            f.write(
                "  flexible valid extraction: "
                f"{row['flexible_valid_extract_rate']:.6f}\n"
            )

            f.write(
                "  strict answer agreement with Standard: "
                f"{row['strict_final_answer_agreement_with_standard']:.6f}\n"
            )

            f.write(
                "  flexible answer agreement with Standard: "
                f"{row['flexible_final_answer_agreement_with_standard']:.6f}\n"
            )

            f.write(
                "  flexible answer-pair coverage: "
                f"{row['flexible_answer_pair_coverage']:.6f}\n"
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
                f"{row['filter']}]\n"
            )

            f.write(
                "  Standard accuracy: "
                f"{row['standard_accuracy']:.6f}\n"
            )

            f.write(
                "  Method accuracy: "
                f"{row['method_accuracy']:.6f}\n"
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
                "  C->I / Standard-correct: "
                f"{row['correct_to_incorrect_rate_given_standard_correct']:.6f}\n"
            )

            f.write(
                "  I->C / Standard-incorrect: "
                f"{row['incorrect_to_correct_rate_given_standard_incorrect']:.6f}\n"
            )

            f.write(
                "  net correct change: "
                f"{row['net_correct_change']}\n"
            )

    print(
        f"[WRITE] {report_path}"
    )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print()
    print("#" * 80)
    print(
        "OFFICIAL RESCORING COMPLETE"
    )
    print("#" * 80)

    print(
        f"\nBest filter matching your paper: "
        f"{best_filter}"
    )

    print(
        "\nGenerated files:"
    )

    for filename in [
        "official_gsm8k_report.txt",
        "official_gsm8k_summary.csv",
        "official_gsm8k_transitions.csv",
        "official_gsm8k_per_sample.csv",
        "official_gsm8k_summary.json",
    ]:
        print(
            f"  analysis_results/{filename}"
        )

    print(
        "\nSend me these four small files:"
    )

    print(
        "  analysis_results/"
        "official_gsm8k_report.txt"
    )

    print(
        "  analysis_results/"
        "official_gsm8k_summary.csv"
    )

    print(
        "  analysis_results/"
        "official_gsm8k_transitions.csv"
    )

    print(
        "  analysis_results/"
        "official_gsm8k_summary.json"
    )


if __name__ == "__main__":
    main()
