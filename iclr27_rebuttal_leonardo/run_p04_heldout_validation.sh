#!/bin/bash
#SBATCH --job-name=iclr27_p04_heldout
#SBATCH --account=EUHPC_D36_236
#SBATCH --partition=lrd_all_serial
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --output=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-p04-heldout/scheduler/p04-heldout-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-p04-heldout/scheduler/p04-heldout-%j.err

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
LOG=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/calibration-logs/complete/math500_train_standard-len256-block32.log
OUTPUT_DIR="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/analysis/heldout_consistency"

source /leonardo/home/userexternal/mcao0000/miniconda3/etc/profile.d/conda.sh
conda activate tc-apd
cd "${PROJECT_ROOT}"

python iclr27_rebuttal_leonardo/validate_heldout_consistency.py \
  "${LOG}" \
  --output-dir "${OUTPUT_DIR}" \
  --total-traces 5000 \
  --estimation-traces 4000 \
  --seed 2027 \
  --rho 0.995 \
  --min-count 200 \
  --min-accepted 100 \
  --threshold-start 1.0 \
  --threshold-end 0.01 \
  --grid-step 0.005 \
  --runtime-min-threshold 0.01 \
  --runtime-max-threshold 0.90 \
  --default-threshold 0.90
