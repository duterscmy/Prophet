#!/bin/bash

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/p22-model-variants
SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/eval_p22_model_variant_array.sh"
mkdir -p "${RUN_ROOT}/scheduler" "${RUN_ROOT}/logs" "${RUN_ROOT}/results"
cd "${PROJECT_ROOT}"

FINE_JOB=$(sbatch --parsable \
  --job-name=iclr27_p22_finetuned_full \
  --account=EUHPC_D36_236 --partition=boost_usr_prod --qos=boost_qos_lprod \
  --time=06:00:00 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G \
  --array=0-2 \
  --output="${RUN_ROOT}/scheduler/finetuned-full-%A_%a.out" \
  --error="${RUN_ROOT}/scheduler/finetuned-full-%A_%a.err" \
  --export=ALL,SMOKE=0 \
  "${SCRIPT}")

QUANT_JOB=$(sbatch --parsable \
  --job-name=iclr27_p22_int8_limit500 \
  --account=EUHPC_D36_236 --partition=boost_usr_prod --qos=boost_qos_lprod \
  --time=12:00:00 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G \
  --array=3-5 \
  --output="${RUN_ROOT}/scheduler/int8-limit500-%A_%a.out" \
  --error="${RUN_ROOT}/scheduler/int8-limit500-%A_%a.err" \
  --export=ALL,SMOKE=0,EVAL_LIMIT=500 \
  "${SCRIPT}")

echo "finetuned_full=${FINE_JOB}"
echo "quantized_int8_limit500=${QUANT_JOB}"
