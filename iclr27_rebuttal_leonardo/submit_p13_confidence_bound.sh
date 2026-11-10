#!/bin/bash

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-confidence-bound
CAL_SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/calibrate_math_confidence_bound.sh"
EVAL_SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/eval_p13_confidence_bound.sh"
mkdir -p "${RUN_ROOT}/scheduler" "${RUN_ROOT}/logs" "${RUN_ROOT}/results" "${RUN_ROOT}/thresholds"
cd "${PROJECT_ROOT}"

CAL_JOB=$(sbatch --parsable \
  --output="${RUN_ROOT}/scheduler/calibration-%j.out" \
  --error="${RUN_ROOT}/scheduler/calibration-%j.err" \
  "${CAL_SCRIPT}")
echo "p13_wilson_calibration=${CAL_JOB}"

submit_eval() {
  local benchmark=$1
  local time_limit=$2
  sbatch --parsable \
    --job-name="iclr27_p13_${benchmark}" \
    --account=EUHPC_D36_236 \
    --partition=boost_usr_prod \
    --qos=boost_qos_lprod \
    --time="${time_limit}" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=80G \
    --dependency="afterok:${CAL_JOB}" \
    --output="${RUN_ROOT}/scheduler/${benchmark}-%j.out" \
    --error="${RUN_ROOT}/scheduler/${benchmark}-%j.err" \
    --export="ALL,BENCHMARK=${benchmark}" \
    "${EVAL_SCRIPT}"
}

echo "p13_humaneval=$(submit_eval humaneval 00:45:00)"
echo "p13_math500=$(submit_eval math500 02:00:00)"
echo "p13_gsm8k=$(submit_eval gsm8k 04:00:00)"
