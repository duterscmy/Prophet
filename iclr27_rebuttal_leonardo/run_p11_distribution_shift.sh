#!/bin/bash
set -euo pipefail

REPO_ROOT="/leonardo/home/userexternal/mcao0000/prophet"
PYTHON="/leonardo/home/userexternal/mcao0000/miniconda3/envs/tc-apd/bin/python"
RESULTS_ROOT="/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-size-ablation/results"
TOKENIZER="/leonardo_scratch/large/userexternal/mcao0000/models/LLaDA-8B-Instruct"
OUTPUT_DIR="$REPO_ROOT/iclr27_rebuttal_leonardo/analysis/standard_tc_apd_shift"

cd "$REPO_ROOT"
"$PYTHON" iclr27_rebuttal_leonardo/analyze_standard_tc_apd_shift.py \
  --results-root "$RESULTS_ROOT" \
  --tokenizer "$TOKENIZER" \
  --output-dir "$OUTPUT_DIR"
