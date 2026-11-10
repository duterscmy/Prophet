#!/bin/bash
#SBATCH --job-name=iclr27_p14_analysis
#SBATCH --account=EUHPC_D36_236
#SBATCH --partition=lrd_all_serial
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
FIXED_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-matched-compute
TC_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-size-ablation
OUTPUT_DIR="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/analysis/matched_compute"

source /leonardo/home/userexternal/mcao0000/miniconda3/etc/profile.d/conda.sh
conda activate tc-apd
cd "${PROJECT_ROOT}"

python iclr27_rebuttal_leonardo/analyze_p14_matched_compute.py \
  --fixed-root "${FIXED_ROOT}" \
  --tc-root "${TC_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --bootstrap-repeats 20000 \
  --seed 42
