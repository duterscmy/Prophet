#!/bin/bash
#SBATCH --job-name=iclr27_p11_traj_analysis
#SBATCH --account=EUHPC_D36_236
#SBATCH --partition=lrd_all_serial
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --output=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-p11-iteration-trajectories/scheduler/analysis-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-p11-iteration-trajectories/scheduler/analysis-%j.err

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-p11-iteration-trajectories
THRESHOLD_PATH=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-size-ablation/thresholds/math_n5000_seed2027_token_threshold_grid_p99.5_mincount200_minaccepted100.json

source /leonardo/home/userexternal/mcao0000/miniconda3/etc/profile.d/conda.sh
conda activate tc-apd
cd "${PROJECT_ROOT}"

python iclr27_rebuttal_leonardo/analyze_p11_iteration_trajectory.py \
  --standard-log "${RUN_ROOT}/logs/gsm8k_standard_limit500_len256_block32_iteration_traces.log" \
  --tc-log "${RUN_ROOT}/logs/gsm8k_tc_apd_limit500_len256_block32_iteration_traces.log" \
  --threshold-table "${THRESHOLD_PATH}" \
  --num-traces 500 \
  --output-dir iclr27_rebuttal_leonardo/analysis/iteration_trajectory_shift
