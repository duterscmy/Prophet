#!/bin/bash

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-p11-iteration-trajectories
EVAL_SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/eval_p11_iteration_trajectory_array.sh"
ANALYSIS_SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/run_p11_iteration_trajectory_analysis.sh"

mkdir -p "${RUN_ROOT}/scheduler" "${RUN_ROOT}/logs" "${RUN_ROOT}/results"
cd "${PROJECT_ROOT}"

EVAL_JOB=$(sbatch --parsable \
  --job-name=iclr27_p11_iter_traj \
  --account=EUHPC_D36_236 \
  --partition=boost_usr_prod \
  --qos=boost_qos_lprod \
  --time=04:00:00 \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=80G \
  --array=0-1 \
  --output="${RUN_ROOT}/scheduler/eval-%A_%a.out" \
  --error="${RUN_ROOT}/scheduler/eval-%A_%a.err" \
  "${EVAL_SCRIPT}")

ANALYSIS_JOB=$(sbatch --parsable \
  --dependency="afterok:${EVAL_JOB}" \
  "${ANALYSIS_SCRIPT}")

echo "p11_iteration_trajectory_eval=${EVAL_JOB}"
echo "p11_iteration_trajectory_analysis=${ANALYSIS_JOB}"
