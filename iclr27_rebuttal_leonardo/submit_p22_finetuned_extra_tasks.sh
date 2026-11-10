#!/bin/bash

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/p22-model-variants
SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/eval_p22_model_variant_array.sh"
mkdir -p "${RUN_ROOT}/scheduler" "${RUN_ROOT}/logs" "${RUN_ROOT}/results"
cd "${PROJECT_ROOT}"

submit_task() {
  local benchmark=$1
  local time_limit=$2
  sbatch --parsable \
    --job-name="iclr27_p22_ft_${benchmark}" \
    --account=EUHPC_D36_236 --partition=boost_usr_prod --qos=boost_qos_lprod \
    --time="${time_limit}" --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G \
    --array=0-2 \
    --output="${RUN_ROOT}/scheduler/finetuned-${benchmark}-%A_%a.out" \
    --error="${RUN_ROOT}/scheduler/finetuned-${benchmark}-%A_%a.err" \
    --export="ALL,SMOKE=0,BENCHMARK=${benchmark}" \
    "${SCRIPT}"
}

echo "finetuned_math500=$(submit_task math500 03:00:00)"
echo "finetuned_humaneval=$(submit_task humaneval 01:30:00)"
