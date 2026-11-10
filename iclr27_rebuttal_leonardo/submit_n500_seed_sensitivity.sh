#!/bin/bash

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-seed-sensitivity-n500
CAL_SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/calibrate_math_n500_extra_seeds.sh"
EVAL_SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/eval_n500_seed_sensitivity_array.sh"
mkdir -p "${RUN_ROOT}/scheduler" "${RUN_ROOT}/logs" "${RUN_ROOT}/results" "${RUN_ROOT}/thresholds"
cd "${PROJECT_ROOT}"

CAL_JOB=$(sbatch --parsable "${CAL_SCRIPT}")
echo "calibration_and_table_analysis=${CAL_JOB}"

submit_eval() {
  local benchmark=$1
  local time_limit=$2
  sbatch --parsable \
    --job-name="iclr27_${benchmark}_n500_seeds" \
    --account=EUHPC_D36_236 \
    --partition=boost_usr_prod \
    --qos=boost_qos_lprod \
    --time="${time_limit}" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=80G \
    --array=0-1 \
    --dependency="afterok:${CAL_JOB}" \
    --output="${RUN_ROOT}/scheduler/${benchmark}-%A_%a.out" \
    --error="${RUN_ROOT}/scheduler/${benchmark}-%A_%a.err" \
    --export="ALL,BENCHMARK=${benchmark}" \
    "${EVAL_SCRIPT}"
}

echo "humaneval=$(submit_eval humaneval 00:30:00)"
echo "math500=$(submit_eval math500 02:00:00)"
echo "gsm8k=$(submit_eval gsm8k 04:00:00)"
