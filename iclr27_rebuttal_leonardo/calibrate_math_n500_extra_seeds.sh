#!/bin/bash
#SBATCH --job-name=iclr27_cal_n500_seeds
#SBATCH --account=EUHPC_D36_236
#SBATCH --partition=lrd_all_serial
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --output=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-seed-sensitivity-n500/scheduler/calibration-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-seed-sensitivity-n500/scheduler/calibration-%j.err

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-seed-sensitivity-n500
TABLE_ROOT="${RUN_ROOT}/thresholds"
MATH_LOG=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/calibration-logs/complete/math500_train_standard-len256-block32.log
MODEL_PATH=/leonardo_scratch/large/userexternal/mcao0000/models/LLaDA-8B-Instruct

source /leonardo/home/userexternal/mcao0000/miniconda3/etc/profile.d/conda.sh
conda activate tc-apd
cd "${PROJECT_ROOT}"

for seed in 2028 2029; do
  python iclr27_rebuttal_leonardo/calibrate_math_size_ablation.py \
    "${MATH_LOG}" \
    --output-dir "${TABLE_ROOT}/seed${seed}" \
    --sizes 500 \
    --pool-size 5000 \
    --seed "${seed}" \
    --rho 0.995 \
    --min-count 200 \
    --min-accepted 100 \
    --threshold-start 1.0 \
    --threshold-end 0.01 \
    --grid-step 0.005
done

mkdir -p "${TABLE_ROOT}/seed2027"
cp /leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-size-ablation/thresholds/math_n500_seed2027_token_threshold_grid_p99.5_mincount200_minaccepted100.json "${TABLE_ROOT}/seed2027/"
cp /leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-size-ablation/thresholds/math_n500_seed2027_token_threshold_grid_p99.5_mincount200_minaccepted100.csv "${TABLE_ROOT}/seed2027/"

python iclr27_rebuttal_leonardo/analyze_threshold_seed_sensitivity.py \
  --threshold-root "${TABLE_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --size 500 \
  --output-dir "${PROJECT_ROOT}/iclr27_rebuttal_leonardo/analysis/threshold_seed_sensitivity_n500"
