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