#!/bin/bash
set -euo pipefail

REPO_ROOT="/leonardo/home/userexternal/mcao0000/prophet"
PYTHON="/leonardo/home/userexternal/mcao0000/miniconda3/envs/tc-apd/bin/python"

cd "$REPO_ROOT"
"$PYTHON" iclr27_rebuttal_leonardo/analyze_calibration_amortization.py \
  --metrics-csv iclr27_rebuttal_leonardo/analysis/calibration_size/evaluation_metrics_long.csv \
  --output-dir iclr27_rebuttal_leonardo/analysis/calibration_amortization \
  --threshold-cpu-seconds 124 \
  --threshold-cpus 8 \
  --source-run-gpu-hours 15.622
