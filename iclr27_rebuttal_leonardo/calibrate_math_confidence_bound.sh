#!/bin/bash
#SBATCH --job-name=iclr27_p13_wilson_cal
#SBATCH --account=EUHPC_D36_236
#SBATCH --partition=lrd_all_serial
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-confidence-bound
LOG=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/calibration-logs/complete/math500_train_standard-len256-block32.log
EMPIRICAL=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-size-ablation/thresholds/math_n5000_seed2027_token_threshold_grid_p99.5_mincount200_minaccepted100.csv

source /leonardo/home/userexternal/mcao0000/miniconda3/etc/profile.d/conda.sh
conda activate tc-apd
mkdir -p "${RUN_ROOT}/thresholds" "${RUN_ROOT}/scheduler"
cd "${PROJECT_ROOT}"

python iclr27_rebuttal_leonardo/calibrate_math_confidence_bound.py \
  "${LOG}" \
  --output-dir "${RUN_ROOT}/thresholds" \
  --empirical-csv "${EMPIRICAL}" \
  --traces 5000 \
  --rho 0.995 \
  --confidence-level 0.95 \
  --bound wilson \
  --min-count 200 \
  --min-accepted 100

ANALYSIS_DIR="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/analysis/confidence_bound"
mkdir -p "${ANALYSIS_DIR}"
cp "${RUN_ROOT}/thresholds/math_n5000_wilson95lower_p99.5_mincount200_minaccepted100.json" "${ANALYSIS_DIR}/"
cp "${RUN_ROOT}/thresholds/math_n5000_wilson95lower_p99.5_mincount200_minaccepted100.csv" "${ANALYSIS_DIR}/"
cp "${RUN_ROOT}/thresholds/summary.json" "${ANALYSIS_DIR}/"
cp "${RUN_ROOT}/thresholds/P13_CONFIDENCE_BOUND_CALIBRATION.md" "${ANALYSIS_DIR}/"
