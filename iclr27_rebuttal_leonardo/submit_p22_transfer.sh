#!/bin/bash

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/p22-transfer
SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/eval_p22_transfer_array.sh"

mkdir -p "${RUN_ROOT}/scheduler" "${RUN_ROOT}/logs" "${RUN_ROOT}/results"
cd "${PROJECT_ROOT}"

submit_benchmark() {
  local benchmark=$1
  local time_limit=$2
  sbatch --parsable \
    --job-name="iclr27_p22_${benchmark}" \
    --account=EUHPC_D36_236 \
    --partition=boost_usr_prod \
    --qos=boost_qos_lprod \
    --time="${time_limit}" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=80G \
    --array=0-11 \
    --output="${RUN_ROOT}/scheduler/${benchmark}-%A_%a.out" \
    --error="${RUN_ROOT}/scheduler/${benchmark}-%A_%a.err" \
    --export="ALL,BENCHMARK=${benchmark}" \
    "${SCRIPT}"
}

echo "p22_humaneval=$(submit_benchmark humaneval 02:00:00)"
echo "p22_math500=$(submit_benchmark math500 05:00:00)"
echo "p22_gsm8k=$(submit_benchmark gsm8k 10:00:00)"
