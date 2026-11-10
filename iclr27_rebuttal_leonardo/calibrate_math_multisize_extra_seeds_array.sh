#!/bin/bash
#SBATCH --job-name=iclr27_cal_multiseed
#SBATCH --account=EUHPC_D36_236
#SBATCH --partition=lrd_all_serial
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-seed-sensitivity-multisize
MATH_LOG=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/calibration-logs/complete/math500_train_standard-len256-block32.log

source /leonardo/home/userexternal/mcao0000/miniconda3/etc/profile.d/conda.sh
conda activate tc-apd
cd "${PROJECT_ROOT}"

for SEED in 2028 2029; do
  TABLE_DIR="${RUN_ROOT}/thresholds/seed${SEED}"
  mkdir -p "${TABLE_DIR}"
  python iclr27_rebuttal_leonardo/calibrate_math_size_ablation.py \
    "${MATH_LOG}" \
    --output-dir "${TABLE_DIR}" \
    --sizes 200,1000,2000,5000 \
    --pool-size 5000 \
    --seed "${SEED}" \
    --rho 0.995 \
    --min-count 200 \
    --min-accepted 100 \
    --threshold-start 1.0 \
    --threshold-end 0.01 \
    --grid-step 0.005
done
