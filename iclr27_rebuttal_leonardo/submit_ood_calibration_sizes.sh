#!/bin/bash

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/eval_ood_calibration_sizes_array.sh"
cd "${PROJECT_ROOT}"

submit_array() {
  local benchmark=$1
  local run_root=$2
  local time_limit=$3
  mkdir -p "${run_root}/scheduler" "${run_root}/logs" "${run_root}/results"
  sbatch --parsable \
    --job-name="iclr27_${benchmark}_cal_sizes" \
    --account=EUHPC_D36_236 \
    --partition=boost_usr_prod \
    --qos=boost_qos_lprod \
    --time="${time_limit}" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=80G \
    --array=0-4 \
    --output="${run_root}/scheduler/cal-sizes-%A_%a.out" \
    --error="${run_root}/scheduler/cal-sizes-%A_%a.err" \
    --export="ALL,BENCHMARK=${benchmark}" \
    "${SCRIPT}"
}

ARC_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-arc-challenge
XSUM_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-xsum-500

echo "arc_challenge_calibration_sizes=$(submit_array arc_challenge "${ARC_ROOT}" 01:00:00)"
echo "xsum500_calibration_sizes=$(submit_array xsum500 "${XSUM_ROOT}" 01:30:00)"
