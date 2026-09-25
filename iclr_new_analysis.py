#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory-efficient analysis for TC-APD / Fixed APD / Standard GSM8K logs.

Expected methods:
    Standard
    Fixed APD 0.8
    Fixed APD 0.9
    TC-APD 0.8
    TC-APD 0.9

The script streams large log files sample-by-sample and does NOT load
the full logs into memory.

Main analyses:
1. Final token agreement with Standard decoding.
2. Exact sequence agreement with Standard decoding.
3. Parallel-commit agreement with Standard final tokens.
4. Single-fallback agreement.
5. Candidate / eligible / selected empirical consistency.
6. TC-APD default-threshold / clipping / threshold-use diagnostics.
7. Fixed-vs-TC paired comparisons.
8. Paired bootstrap CI for NFE and agreement differences.
9. Vocabulary-wide token-conditioned heterogeneity on Standard trajectories.

Important:
- Samples are aligned by their order in the logs.
- Therefore, all five logs MUST evaluate the same GSM8K examples
  in exactly the same order.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Any, Optional, Tuple

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
        "logs/"
        "gsm8k_dynamic_from_math_c99.5_mincount200_"
        "minaccepted100_len256_block32_maxthr0.80_minthr0.05.log"
    ),

    "tc_090": Path(
        "logs/"
        "gsm8k_dynamic_from_math_c99.5_mincount200_"
        "minaccepted100_len256_block32_maxthr0.90_minthr0.05.log"
    ),
}


OUTPUT_DIR = Path("analysis_results")

EXPECTED_NUM_SAMPLES = 1319

BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 42

# Confidence-bin width for vocabulary-wide heterogeneity analysis.
CONF_BIN_WIDTH = 0.05

# A token must have at least this many observations inside one
# confidence bin before it contributes to token-level dispersion stats.
MIN_TOKEN_BIN_SUPPORT = 20


# ============================================================
# Optional fast JSON parser
# ============================================================

try:
    import orjson

    def loads_json(raw):
        return orjson.loads(raw)

    print("[INFO] Using orjson for fast JSON parsing.")

except ImportError:

    def loads_json(raw):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return json.loads(raw)

    print("[INFO] orjson not found; using Python json.")
    print("       Optional speedup: pip install orjson")


# ============================================================
# Utility classes
# ============================================================

class AgreementCounter:
    """
    Counts:
        total comparable observations
        observations agreeing with Standard final token
    """

    def __init__(self):
        self.total = 0
        self.agree = 0

    def add(self, is_agree: bool):
        self.total += 1
        self.agree += int(bool(is_agree))

    @property
    def rate(self) -> float:
        if self.total == 0:
            return float("nan")
        return self.agree / self.total

    def to_dict(self):
        return {
            "agree": int(self.agree),
            "total": int(self.total),
            "rate": safe_float(self.rate),
        }


# ============================================================
# General helpers
# ============================================================

def safe_float(x):
    try:
        x = float(x)
    except Exception:
        return None

    if math.isnan(x) or math.isinf(x):
        return None

    return x


def safe_mean(values):
    values = [
        float(x)
        for x in values
        if x is not None and not math.isnan(float(x))
    ]

    if not values:
        return float("nan")

    return float(np.mean(values))


def safe_median(values):
    values = [
        float(x)
        for x in values
        if x is not None and not math.isnan(float(x))
    ]

    if not values:
        return float("nan")

    return float(np.median(values))


def ratio(num, den):
    if den == 0:
        return float("nan")
    return num / den


def get_generation_position(record: Dict[str, Any]) -> Optional[int]:
    """
    Prefer generation_relative_position because absolute position
    depends on prompt length.
    """

    if "generation_relative_position" in record:
        try:
            return int(record["generation_relative_position"])
        except Exception:
            pass

    # Do not silently use absolute "position", because prompt lengths
    # vary across GSM8K examples.
    return None


def reconstruct_final_tokens(
    selected_records: List[Dict[str, Any]]
) -> Dict[int, int]:
    """
    Every generated position should eventually be committed exactly once.
    Reconstruct:
        generation_relative_position -> final token_id
    """

    result = {}

    for r in selected_records:
        pos = get_generation_position(r)

        if pos is None:
            continue

        try:
            token_id = int(r["token_id"])
        except Exception:
            continue

        result[pos] = token_id

    return result


def compare_final_sequences(
    reference: Dict[int, int],
    prediction: Dict[int, int],
) -> Tuple[int, int, float, bool]:
    """
    Reference is Standard decoding.

    Missing prediction positions count as mismatches.
    Extra prediction positions also make exact match False.
    """

    ref_positions = set(reference)
    pred_positions = set(prediction)

    comparable_total = len(ref_positions)

    matched = 0

    for pos in ref_positions:
        if prediction.get(pos, None) == reference[pos]:
            matched += 1

    agreement = (
        matched / comparable_total
        if comparable_total > 0
        else float("nan")
    )

    exact = (
        ref_positions == pred_positions
        and all(
            reference[p] == prediction[p]
            for p in ref_positions
        )
    )

    return matched, comparable_total, agreement, exact


# ============================================================
# Large-log parser
# ============================================================

def iter_structured_samples(
    path: Path,
    method_name: str,
):
    """
    Streams structured JSON objects from the log.

    Full-confidence generators print one object like:

        {
            "selected_records": [...],
            "all_token_records": [...],
            "stats": {...}
        }

    They may additionally print selected_records alone as a JSON array.
    This parser ignores the latter to avoid double counting.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Log not found: {path}"
        )

    file_size = path.stat().st_size
    sample_idx = 0

    marker1 = b'"selected_records"'
    marker2 = b'"all_token_records"'

    print()
    print("=" * 80)
    print(f"[READ] {method_name}")
    print(f"       {path}")
    print(
        f"       Size: "
        f"{file_size / (1024 ** 3):.2f} GB"
    )
    print("=" * 80)

    with open(
        path,
        "rb",
        buffering=1024 * 1024 * 16
    ) as f:

        for raw_line in f:

            # Fast rejection before expensive JSON parsing.
            if marker1 not in raw_line:
                continue

            if marker2 not in raw_line:
                continue

            start = raw_line.find(b"{")

            if start < 0:
                continue

            raw_json = raw_line[start:].strip()

            try:
                obj = loads_json(raw_json)

            except Exception as exc:
                print(
                    f"[WARN] JSON parse failed in {method_name}, "
                    f"candidate sample {sample_idx}: {exc}"
                )
                continue

            if not isinstance(obj, dict):
                continue

            if (
                "selected_records" not in obj
                or "all_token_records" not in obj
            ):
                continue

            yield sample_idx, obj

            sample_idx += 1

            if sample_idx % 50 == 0:
                try:
                    progress = (
                        f.tell() / file_size * 100.0
                    )
                    print(
                        f"[{method_name}] "
                        f"{sample_idx} samples "
                        f"({progress:.1f}% file)"
                    )
                except Exception:
                    print(
                        f"[{method_name}] "
                        f"{sample_idx} samples"
                    )

    print(
        f"[DONE] {method_name}: "
        f"{sample_idx} structured samples."
    )


# ============================================================
# Standard trajectory analysis
# ============================================================

def confidence_bin(conf: float) -> int:
    """
    Example width=0.05:
        0.00-0.05 -> 0
        ...
        0.95-1.00 -> 19
    """

    conf = min(
        max(float(conf), 0.0),
        1.0
    )

    idx = int(
        conf / CONF_BIN_WIDTH
    )

    max_idx = int(
        math.ceil(1.0 / CONF_BIN_WIDTH)
    ) - 1

    return min(idx, max_idx)


def analyze_standard_log(path: Path):
    """
    Parse Standard decoding once.

    Stores only:
        - final token sequence per sample
        - compact sample statistics

    Candidate traces are aggregated online into confidence/token bins.
    They are NOT retained in memory.
    """

    standard_final_tokens = []
    standard_sample_metrics = []

    # (bin_idx, token_id) -> [total, consistent]
    token_bin_stats = defaultdict(
        lambda: [0, 0]
    )

    # bin_idx -> [total, consistent]
    global_bin_stats = defaultdict(
        lambda: [0, 0]
    )

    for sample_idx, obj in iter_structured_samples(
        path,
        "standard"
    ):

        selected_records = obj.get(
            "selected_records",
            []
        )

        all_records = obj.get(
            "all_token_records",
            []
        )

        stats = obj.get(
            "stats",
            {}
        )

        final_tokens = reconstruct_final_tokens(
            selected_records
        )

        standard_final_tokens.append(
            final_tokens
        )

        nfe = stats.get(
            "model_forward_calls",
            stats.get("steps", None)
        )

        decoding_steps = stats.get(
            "steps",
            None
        )

        standard_sample_metrics.append({
            "sample_idx": sample_idx,
            "nfe": nfe,
            "steps": decoding_steps,
            "num_final_tokens":
                len(final_tokens),
        })

        # --------------------------------------------
        # Standard trajectory:
        # confidence -> final consistency
        # --------------------------------------------

        for r in all_records:

            pos = get_generation_position(r)

            if pos is None:
                continue

            if pos not in final_tokens:
                continue

            try:
                token_id = int(r["token_id"])
                conf = float(r["confidence"])
            except Exception:
                continue

            is_consistent = (
                token_id
                == final_tokens[pos]
            )

            b = confidence_bin(conf)

            tb = token_bin_stats[
                (b, token_id)
            ]

            tb[0] += 1
            tb[1] += int(is_consistent)

            gb = global_bin_stats[b]

            gb[0] += 1
            gb[1] += int(is_consistent)

    return {
        "final_tokens": standard_final_tokens,
        "sample_metrics": standard_sample_metrics,
        "token_bin_stats": token_bin_stats,
        "global_bin_stats": global_bin_stats,
    }


# ============================================================
# Accelerated trajectory analysis
# ============================================================

def analyze_accelerated_log(
    path: Path,
    method_name: str,
    standard_final_tokens: List[Dict[int, int]],
):
    """
    Analyze Fixed APD or TC-APD against Standard final outputs.
    """

    sample_metrics = []

    selected_groups = defaultdict(
        AgreementCounter
    )

    candidate_groups = defaultdict(
        AgreementCounter
    )

    counters = defaultdict(int)

    stats_first_sample = None

    for sample_idx, obj in iter_structured_samples(
        path,
        method_name
    ):

        if sample_idx >= len(
            standard_final_tokens
        ):
            print(
                f"[WARN] {method_name} contains more samples "
                f"than Standard; stopping at sample "
                f"{sample_idx}."
            )
            break

        reference = (
            standard_final_tokens[
                sample_idx
            ]
        )

        selected_records = obj.get(
            "selected_records",
            []
        )

        all_records = obj.get(
            "all_token_records",
            []
        )

        stats = obj.get(
            "stats",
            {}
        )

        if stats_first_sample is None:
            stats_first_sample = dict(stats)

        # --------------------------------------------
        # Final sequence agreement
        # --------------------------------------------

        prediction = reconstruct_final_tokens(
            selected_records
        )

        (
            final_match_count,
            final_total,
            final_agreement,
            exact_sequence,
        ) = compare_final_sequences(
            reference,
            prediction,
        )

        final_mismatch_count = (
            final_total
            - final_match_count
        )

        # --------------------------------------------
        # Selected / committed token agreement
        # --------------------------------------------

        parallel_total = 0
        parallel_agree = 0

        single_total = 0
        single_agree = 0

        first_parallel_mismatch_step = None

        for r in selected_records:

            pos = get_generation_position(r)

            if (
                pos is None
                or pos not in reference
            ):
                continue

            try:
                token_id = int(
                    r["token_id"]
                )
            except Exception:
                continue

            agree = (
                token_id
                == reference[pos]
            )

            strategy = r.get(
                "strategy",
                "unknown"
            )

            selected_groups[
                "all_selected"
            ].add(agree)

            if strategy == "parallel":

                selected_groups[
                    "parallel"
                ].add(agree)

                parallel_total += 1
                parallel_agree += int(
                    agree
                )

                if (
                    not agree
                    and
                    first_parallel_mismatch_step
                    is None
                ):
                    first_parallel_mismatch_step = (
                        r.get(
                            "global_step",
                            r.get("step")
                        )
                    )

            elif strategy == "single":

                selected_groups[
                    "single"
                ].add(agree)

                single_total += 1
                single_agree += int(
                    agree
                )

            # TC diagnostics if fields exist
            if (
                "used_default_threshold"
                in r
            ):

                if r[
                    "used_default_threshold"
                ]:
                    selected_groups[
                        "selected_default_threshold"
                    ].add(agree)

                    counters[
                        "selected_default"
                    ] += 1

                else:
                    selected_groups[
                        "selected_token_specific"
                    ].add(agree)

                    counters[
                        "selected_token_specific"
                    ] += 1

            if (
                "threshold_clipped"
                in r
            ):

                if r[
                    "threshold_clipped"
                ]:
                    selected_groups[
                        "selected_clipped"
                    ].add(agree)

                    counters[
                        "selected_clipped"
                    ] += 1

                else:
                    selected_groups[
                        "selected_not_clipped"
                    ].add(agree)

                    counters[
                        "selected_not_clipped"
                    ] += 1

        # --------------------------------------------
        # All candidate trajectory records
        # --------------------------------------------

        for r in all_records:

            pos = get_generation_position(r)

            if (
                pos is None
                or pos not in reference
            ):
                continue

            try:
                token_id = int(
                    r["token_id"]
                )
            except Exception:
                continue

            agree = (
                token_id
                == reference[pos]
            )

            candidate_groups[
                "all_candidates"
            ].add(agree)

            counters[
                "candidate_total"
            ] += 1

            # ------------------------------------
            # Eligibility
            # ------------------------------------

            if "eligible" in r:

                if r["eligible"]:

                    candidate_groups[
                        "eligible"
                    ].add(agree)

                    counters[
                        "eligible"
                    ] += 1

                else:

                    candidate_groups[
                        "not_eligible"
                    ].add(agree)

                    counters[
                        "not_eligible"
                    ] += 1

            # ------------------------------------
            # Selected candidate
            # ------------------------------------

            if r.get(
                "selected",
                False
            ):

                candidate_groups[
                    "selected_candidates"
                ].add(agree)

                counters[
                    "candidate_selected"
                ] += 1

            # ------------------------------------
            # Default vs token-specific threshold
            # ------------------------------------

            if (
                "used_default_threshold"
                in r
            ):

                if r[
                    "used_default_threshold"
                ]:

                    candidate_groups[
                        "default_threshold_candidates"
                    ].add(agree)

                    counters[
                        "candidate_default"
                    ] += 1

                else:

                    candidate_groups[
                        "token_specific_candidates"
                    ].add(agree)

                    counters[
                        "candidate_token_specific"
                    ] += 1

            # ------------------------------------
            # Clipping diagnostics
            # ------------------------------------

            if (
                "threshold_clipped"
                in r
            ):

                if r[
                    "threshold_clipped"
                ]:

                    candidate_groups[
                        "clipped_candidates"
                    ].add(agree)

                    counters[
                        "candidate_clipped"
                    ] += 1

                else:

                    candidate_groups[
                        "not_clipped_candidates"
                    ].add(agree)

                    counters[
                        "candidate_not_clipped"
                    ] += 1

        nfe = stats.get(
            "model_forward_calls",
            stats.get("steps", None)
        )

        decoding_steps = stats.get(
            "steps",
            None
        )

        avg_tokens_step = stats.get(
            "tokens_per_decoding_step",
            None
        )

        sample_metrics.append({
            "sample_idx":
                sample_idx,

            "nfe":
                nfe,

            "steps":
                decoding_steps,

            "avg_tokens_per_decoding_step":
                avg_tokens_step,

            "final_token_agreement":
                final_agreement,

            "final_match_count":
                final_match_count,

            "final_total_tokens":
                final_total,

            "final_mismatch_count":
                final_mismatch_count,

            "exact_sequence_agreement":
                int(exact_sequence),

            "parallel_commit_count":
                parallel_total,

            "parallel_commit_agree":
                parallel_agree,

            "parallel_commit_agreement":
                ratio(
                    parallel_agree,
                    parallel_total
                ),

            "single_commit_count":
                single_total,

            "single_commit_agree":
                single_agree,

            "single_commit_agreement":
                ratio(
                    single_agree,
                    single_total
                ),

            "first_parallel_mismatch_step":
                first_parallel_mismatch_step,
        })

    return {
        "sample_metrics":
            sample_metrics,

        "selected_groups":
            selected_groups,

        "candidate_groups":
            candidate_groups,

        "counters":
            counters,

        "stats_first_sample":
            stats_first_sample,
    }


# ============================================================
# Vocabulary-wide heterogeneity
# ============================================================

def build_heterogeneity_table(
    token_bin_stats,
    global_bin_stats,
):
    """
    For each confidence bin, measure dispersion in token-specific
    final-consistency rates.

    This directly supports the claim that identical confidence
    levels correspond to different stability across token IDs.
    """

    max_bin = int(
        math.ceil(
            1.0 / CONF_BIN_WIDTH
        )
    )

    rows = []

    for b in range(max_bin):

        left = b * CONF_BIN_WIDTH

        right = min(
            1.0,
            (b + 1) * CONF_BIN_WIDTH
        )

        token_rates = []
        token_counts = []

        for (
            bin_idx,
            token_id
        ), (
            total,
            consistent
        ) in token_bin_stats.items():

            if bin_idx != b:
                continue

            if total < MIN_TOKEN_BIN_SUPPORT:
                continue

            rate = (
                consistent / total
            )

            token_rates.append(
                rate
            )

            token_counts.append(
                total
            )

        global_total, global_consistent = (
            global_bin_stats.get(
                b,
                [0, 0]
            )
        )

        global_rate = ratio(
            global_consistent,
            global_total
        )

        if token_rates:

            arr = np.asarray(
                token_rates,
                dtype=float
            )

            rows.append({
                "confidence_left":
                    left,

                "confidence_right":
                    right,

                "global_observations":
                    global_total,

                "global_consistency":
                    global_rate,

                "eligible_tokens":
                    len(arr),

                "token_observations_after_support_filter":
                    int(
                        sum(token_counts)
                    ),

                "token_mean_consistency":
                    float(
                        np.mean(arr)
                    ),

                "token_std_consistency":
                    float(
                        np.std(
                            arr,
                            ddof=0
                        )
                    ),

                "p10":
                    float(
                        np.quantile(
                            arr,
                            0.10
                        )
                    ),

                "p25":
                    float(
                        np.quantile(
                            arr,
                            0.25
                        )
                    ),

                "median":
                    float(
                        np.quantile(
                            arr,
                            0.50
                        )
                    ),

                "p75":
                    float(
                        np.quantile(
                            arr,
                            0.75
                        )
                    ),

                "p90":
                    float(
                        np.quantile(
                            arr,
                            0.90
                        )
                    ),

                "iqr":
                    float(
                        np.quantile(
                            arr,
                            0.75
                        )
                        -
                        np.quantile(
                            arr,
                            0.25
                        )
                    ),
            })

        else:

            rows.append({
                "confidence_left":
                    left,

                "confidence_right":
                    right,

                "global_observations":
                    global_total,

                "global_consistency":
                    global_rate,

                "eligible_tokens":
                    0,

                "token_observations_after_support_filter":
                    0,

                "token_mean_consistency":
                    None,

                "token_std_consistency":
                    None,

                "p10":
                    None,

                "p25":
                    None,

                "median":
                    None,

                "p75":
                    None,

                "p90":
                    None,

                "iqr":
                    None,
            })

    return rows


# ============================================================
# Method summaries
# ============================================================

def summarize_method(
    method_name: str,
    analysis: Dict[str, Any],
):
    samples = analysis[
        "sample_metrics"
    ]

    selected_groups = analysis[
        "selected_groups"
    ]

    candidate_groups = analysis[
        "candidate_groups"
    ]

    counters = analysis[
        "counters"
    ]

    nfes = [
        x["nfe"]
        for x in samples
        if x.get("nfe") is not None
    ]

    final_agreements = [
        x["final_token_agreement"]
        for x in samples
        if x.get(
            "final_token_agreement"
        ) is not None
    ]

    exacts = [
        x["exact_sequence_agreement"]
        for x in samples
    ]

    avg_tokens_steps = [
        x[
            "avg_tokens_per_decoding_step"
        ]
        for x in samples
        if x.get(
            "avg_tokens_per_decoding_step"
        ) is not None
    ]

    candidate_total = counters.get(
        "candidate_total",
        0
    )

    row = {
        "method":
            method_name,

        "num_samples":
            len(samples),

        "mean_nfe":
            safe_mean(nfes),

        "median_nfe":
            safe_median(nfes),

        "mean_final_token_agreement":
            safe_mean(
                final_agreements
            ),

        "exact_sequence_agreement_rate":
            safe_mean(exacts),

        "parallel_commit_agreement":
            safe_float(
                selected_groups[
                    "parallel"
                ].rate
            ),

        "parallel_commit_total":
            selected_groups[
                "parallel"
            ].total,

        "single_commit_agreement":
            safe_float(
                selected_groups[
                    "single"
                ].rate
            ),

        "single_commit_total":
            selected_groups[
                "single"
            ].total,

        "all_candidate_consistency":
            safe_float(
                candidate_groups[
                    "all_candidates"
                ].rate
            ),

        "eligible_candidate_consistency":
            safe_float(
                candidate_groups[
                    "eligible"
                ].rate
            ),

        "eligible_candidate_count":
            candidate_groups[
                "eligible"
            ].total,

        "selected_candidate_consistency":
            safe_float(
                candidate_groups[
                    "selected_candidates"
                ].rate
            ),

        "mean_tokens_per_decoding_step":
            safe_mean(
                avg_tokens_steps
            ),

        "token_specific_candidate_ratio":
            safe_float(
                ratio(
                    counters.get(
                        "candidate_token_specific",
                        0
                    ),
                    candidate_total
                )
            ),

        "default_candidate_ratio":
            safe_float(
                ratio(
                    counters.get(
                        "candidate_default",
                        0
                    ),
                    candidate_total
                )
            ),

        "clipped_candidate_ratio":
            safe_float(
                ratio(
                    counters.get(
                        "candidate_clipped",
                        0
                    ),
                    candidate_total
                )
            ),

        "selected_default_threshold_agreement":
            safe_float(
                selected_groups[
                    "selected_default_threshold"
                ].rate
            ),

        "selected_token_specific_agreement":
            safe_float(
                selected_groups[
                    "selected_token_specific"
                ].rate
            ),

        "selected_clipped_agreement":
            safe_float(
                selected_groups[
                    "selected_clipped"
                ].rate
            ),

        "selected_not_clipped_agreement":
            safe_float(
                selected_groups[
                    "selected_not_clipped"
                ].rate
            ),
    }

    return row


# ============================================================
# Bootstrap
# ============================================================

def paired_bootstrap_mean_diff(
    a,
    b,
    repeats=BOOTSTRAP_REPEATS,
    seed=BOOTSTRAP_SEED,
):
    """
    Returns mean(b - a) and paired bootstrap 95% CI.

    a and b must correspond to the same examples.
    """

    pairs = []

    for x, y in zip(a, b):

        if x is None or y is None:
            continue

        x = float(x)
        y = float(y)

        if (
            math.isnan(x)
            or math.isnan(y)
        ):
            continue

        pairs.append(
            (x, y)
        )

    if not pairs:
        return {
            "n": 0,
            "mean_difference": None,
            "ci_low": None,
            "ci_high": None,
        }

    arr = np.asarray(
        pairs,
        dtype=float
    )

    differences = (
        arr[:, 1]
        - arr[:, 0]
    )

    observed = float(
        np.mean(differences)
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(differences)

    boot_means = np.empty(
        repeats,
        dtype=float
    )

    for i in range(repeats):

        idx = rng.integers(
            0,
            n,
            size=n
        )

        boot_means[i] = np.mean(
            differences[idx]
        )

    low, high = np.quantile(
        boot_means,
        [0.025, 0.975]
    )

    return {
        "n":
            n,

        "mean_difference":
            observed,

        "ci_low":
            float(low),

        "ci_high":
            float(high),
    }


def compare_fixed_tc(
    fixed_analysis,
    tc_analysis,
    threshold_label,
):
    fixed_samples = fixed_analysis[
        "sample_metrics"
    ]

    tc_samples = tc_analysis[
        "sample_metrics"
    ]

    n = min(
        len(fixed_samples),
        len(tc_samples)
    )

    fixed_samples = fixed_samples[:n]
    tc_samples = tc_samples[:n]

    fixed_nfe = [
        x["nfe"]
        for x in fixed_samples
    ]

    tc_nfe = [
        x["nfe"]
        for x in tc_samples
    ]

    fixed_final = [
        x["final_token_agreement"]
        for x in fixed_samples
    ]

    tc_final = [
        x["final_token_agreement"]
        for x in tc_samples
    ]

    fixed_exact = [
        x["exact_sequence_agreement"]
        for x in fixed_samples
    ]

    tc_exact = [
        x["exact_sequence_agreement"]
        for x in tc_samples
    ]

    fixed_parallel = [
        x["parallel_commit_agreement"]
        for x in fixed_samples
    ]

    tc_parallel = [
        x["parallel_commit_agreement"]
        for x in tc_samples
    ]

    results = []

    comparisons = [
        (
            "NFE",
            fixed_nfe,
            tc_nfe,
        ),

        (
            "final_token_agreement",
            fixed_final,
            tc_final,
        ),

        (
            "exact_sequence_agreement",
            fixed_exact,
            tc_exact,
        ),

        (
            "parallel_commit_agreement",
            fixed_parallel,
            tc_parallel,
        ),
    ]

    for metric, fixed_values, tc_values in comparisons:

        result = paired_bootstrap_mean_diff(
            fixed_values,
            tc_values,
        )

        results.append({
            "threshold":
                threshold_label,

            "metric":
                metric,

            "difference_definition":
                "TC - Fixed",

            **result,
        })

    return results


# ============================================================
# CSV / JSON writers
# ============================================================

def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
):
    if not rows:
        print(
            f"[WARN] No rows for {path}"
        )
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(
        f"[WRITE] {path}"
    )


def make_json_serializable(obj):

    if isinstance(obj, AgreementCounter):
        return obj.to_dict()

    if isinstance(obj, defaultdict):
        obj = dict(obj)

    if isinstance(obj, dict):
        return {
            str(k):
                make_json_serializable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            make_json_serializable(v)
            for v in obj
        ]

    if isinstance(
        obj,
        (np.integer,)
    ):
        return int(obj)

    if isinstance(
        obj,
        (np.floating,)
    ):
        value = float(obj)

        if (
            math.isnan(value)
            or math.isinf(value)
        ):
            return None

        return value

    if isinstance(obj, float):

        if (
            math.isnan(obj)
            or math.isinf(obj)
        ):
            return None

        return obj

    return obj


# ============================================================
# Human-readable report
# ============================================================

def write_text_report(
    path: Path,
    method_summaries,
    bootstrap_rows,
):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "TC-APD GSM8K LOG ANALYSIS\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        for row in method_summaries:

            f.write(
                f"[{row['method']}]\n"
            )

            for key, value in row.items():

                if key == "method":
                    continue

                if isinstance(
                    value,
                    float
                ):
                    f.write(
                        f"  {key}: "
                        f"{value:.6f}\n"
                    )

                else:
                    f.write(
                        f"  {key}: "
                        f"{value}\n"
                    )

            f.write("\n")

        f.write(
            "\nPAIRED BOOTSTRAP "
            "(difference = TC - Fixed)\n"
        )

        f.write(
            "=" * 80
            + "\n"
        )

        for row in bootstrap_rows:

            f.write(
                f"\nThreshold "
                f"{row['threshold']} | "
                f"{row['metric']}\n"
            )

            f.write(
                f"  n = {row['n']}\n"
            )

            f.write(
                f"  mean difference = "
                f"{row['mean_difference']}\n"
            )

            f.write(
                f"  95% CI = "
                f"[{row['ci_low']}, "
                f"{row['ci_high']}]\n"
            )

    print(
        f"[WRITE] {path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("#" * 80)
    print("TC-APD GSM8K ANALYSIS")
    print("#" * 80)

    print("\nInput logs:")

    for name, path in LOGS.items():
        print(
            f"  {name:12s}: {path}"
        )

    # --------------------------------------------------------
    # 1. Standard
    # --------------------------------------------------------

    print(
        "\n[1/6] Parsing Standard trajectories..."
    )

    standard = analyze_standard_log(
        LOGS["standard"]
    )

    n_standard = len(
        standard["final_tokens"]
    )

    print(
        f"\nStandard samples: "
        f"{n_standard}"
    )

    if (
        EXPECTED_NUM_SAMPLES is not None
        and n_standard
        != EXPECTED_NUM_SAMPLES
    ):
        print(
            f"[WARN] Expected "
            f"{EXPECTED_NUM_SAMPLES} "
            f"GSM8K samples but parsed "
            f"{n_standard}."
        )

    # --------------------------------------------------------
    # 2. Fixed + TC
    # --------------------------------------------------------

    analyses = {}

    order = [
        "fixed_080",
        "fixed_090",
        "tc_080",
        "tc_090",
    ]

    for i, method in enumerate(
        order,
        start=2
    ):

        print(
            f"\n[{i}/6] "
            f"Parsing {method}..."
        )

        analyses[method] = (
            analyze_accelerated_log(
                LOGS[method],
                method,
                standard["final_tokens"],
            )
        )

    # --------------------------------------------------------
    # Validate sample counts
    # --------------------------------------------------------

    print(
        "\nSample counts:"
    )

    print(
        f"  standard   : "
        f"{n_standard}"
    )

    for method in order:

        n = len(
            analyses[
                method
            ][
                "sample_metrics"
            ]
        )

        print(
            f"  {method:10s}: "
            f"{n}"
        )

        if n != n_standard:

            print(
                f"[WARN] {method} has "
                f"{n} samples while Standard "
                f"has {n_standard}."
            )

            print(
                "       Pairing assumes identical "
                "sample order."
            )

    # --------------------------------------------------------
    # 3. Method summary
    # --------------------------------------------------------

    method_summaries = []

    for method in order:

        method_summaries.append(
            summarize_method(
                method,
                analyses[method],
            )
        )

    write_csv(
        OUTPUT_DIR
        / "method_summary.csv",
        method_summaries,
    )

    # --------------------------------------------------------
    # 4. Per-sample compact metrics
    # --------------------------------------------------------

    sample_rows = []

    for method in order:

        for row in analyses[
            method
        ][
            "sample_metrics"
        ]:

            sample_rows.append({
                "method":
                    method,

                **row,
            })

    write_csv(
        OUTPUT_DIR
        / "sample_metrics.csv",
        sample_rows,
    )

    # --------------------------------------------------------
    # 5. Standard vocabulary-wide heterogeneity
    # --------------------------------------------------------

    heterogeneity_rows = (
        build_heterogeneity_table(
            standard[
                "token_bin_stats"
            ],
            standard[
                "global_bin_stats"
            ],
        )
    )

    write_csv(
        OUTPUT_DIR
        / "standard_token_heterogeneity.csv",
        heterogeneity_rows,
    )

    # --------------------------------------------------------
    # 6. TC / Fixed detailed group reliability
    # --------------------------------------------------------

    group_rows = []

    for method in order:

        analysis = analyses[
            method
        ]

        for (
            group_name,
            counter
        ) in analysis[
            "selected_groups"
        ].items():

            group_rows.append({
                "method":
                    method,

                "source":
                    "selected_records",

                "group":
                    group_name,

                "agree":
                    counter.agree,

                "total":
                    counter.total,

                "agreement":
                    safe_float(
                        counter.rate
                    ),
            })

        for (
            group_name,
            counter
        ) in analysis[
            "candidate_groups"
        ].items():

            group_rows.append({
                "method":
                    method,

                "source":
                    "all_token_records",

                "group":
                    group_name,

                "agree":
                    counter.agree,

                "total":
                    counter.total,

                "agreement":
                    safe_float(
                        counter.rate
                    ),
            })

    write_csv(
        OUTPUT_DIR
        / "trajectory_group_agreement.csv",
        group_rows,
    )

    # --------------------------------------------------------
    # 7. Paired Fixed-vs-TC bootstrap
    # --------------------------------------------------------

    bootstrap_rows = []

    bootstrap_rows.extend(
        compare_fixed_tc(
            analyses["fixed_080"],
            analyses["tc_080"],
            "0.80",
        )
    )

    bootstrap_rows.extend(
        compare_fixed_tc(
            analyses["fixed_090"],
            analyses["tc_090"],
            "0.90",
        )
    )

    write_csv(
        OUTPUT_DIR
        / "fixed_vs_tc_bootstrap.csv",
        bootstrap_rows,
    )

    # --------------------------------------------------------
    # 8. Compact diagnostics JSON
    # --------------------------------------------------------

    compact_diagnostics = {}

    for method in order:

        compact_diagnostics[
            method
        ] = {
            "counters":
                dict(
                    analyses[
                        method
                    ][
                        "counters"
                    ]
                ),

            "first_sample_stats":
                analyses[
                    method
                ][
                    "stats_first_sample"
                ],
        }

    combined_json = {
        "method_summary":
            method_summaries,

        "fixed_vs_tc_bootstrap":
            bootstrap_rows,

        "diagnostics":
            compact_diagnostics,

        "standard_num_samples":
            n_standard,

        "configuration": {
            "confidence_bin_width":
                CONF_BIN_WIDTH,

            "min_token_bin_support":
                MIN_TOKEN_BIN_SUPPORT,

            "bootstrap_repeats":
                BOOTSTRAP_REPEATS,

            "bootstrap_seed":
                BOOTSTRAP_SEED,
        },
    }

    with open(
        OUTPUT_DIR
        / "analysis_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            make_json_serializable(
                combined_json
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "[WRITE] "
        f"{OUTPUT_DIR / 'analysis_summary.json'}"
    )

    # --------------------------------------------------------
    # 9. Human-readable text report
    # --------------------------------------------------------

    write_text_report(
        OUTPUT_DIR
        / "report.txt",
        method_summaries,
        bootstrap_rows,
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print()
    print("#" * 80)
    print("ANALYSIS COMPLETE")
    print("#" * 80)

    print(
        "\nGenerated files:"
    )

    for name in [
        "method_summary.csv",
        "sample_metrics.csv",
        "trajectory_group_agreement.csv",
        "standard_token_heterogeneity.csv",
        "fixed_vs_tc_bootstrap.csv",
        "analysis_summary.json",
        "report.txt",
    ]:
        print(
            f"  {OUTPUT_DIR / name}"
        )

    print(
        "\nThe most useful files to send back to me are:"
    )

    print(
        "  1. analysis_results/report.txt"
    )

    print(
        "  2. analysis_results/method_summary.csv"
    )

    print(
        "  3. analysis_results/"
        "trajectory_group_agreement.csv"
    )

    print(
        "  4. analysis_results/"
        "standard_token_heterogeneity.csv"
    )

    print(
        "  5. analysis_results/"
        "fixed_vs_tc_bootstrap.csv"
    )

    print(
        "\nYou do NOT need to send the original 5GB logs."
    )


if __name__ == "__main__":
    main()