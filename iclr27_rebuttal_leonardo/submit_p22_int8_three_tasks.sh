#!/bin/bash

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/p22-model-variants
SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/eval_p22_model_variant_array.sh"
mkdir -p "${RUN_ROOT}/scheduler" "${RUN_ROOT}/logs" "${RUN_ROOT}/results"
cd "${PROJECT_ROOT}"

submit_task() {
  local benchmark=$1
  local eval_limit=$2
  local time_limit=$3
  sbatch --parsable \
    --job-name="iclr27_p22_int8_${benchmark}" \
    --account=EUHPC_D36_236 --partition=boost_usr_prod --qos=boost_qos_lprod \
    --time="${time_limit}" --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G \
    --array=3-5 \
    --output="${RUN_ROOT}/scheduler/int8-${benchmark}-%A_%a.out" \
    --error="${RUN_ROOT}/scheduler/int8-${benchmark}-%A_%a.err" \
    --export="ALL,SMOKE=0,BENCHMARK=${benchmark},EVAL_LIMIT=${eval_limit}" \
    "${SCRIPT}"
}

echo "int8_gsm8k_limit300=$(submit_task gsm8k 300 07:00:00)"
echo "int8_math500_limit300=$(submit_task math500 300 08:00:00)"
# HumanEval has only 164 examples, so run the complete benchmark.
echo "int8_humaneval_full=$(submit_task humaneval 164 05:00:00)"
