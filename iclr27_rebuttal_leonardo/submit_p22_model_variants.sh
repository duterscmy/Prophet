#!/bin/bash

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/p22-model-variants
SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/eval_p22_model_variant_array.sh"
mkdir -p "${RUN_ROOT}/scheduler" "${RUN_ROOT}/logs" "${RUN_ROOT}/results"
cd "${PROJECT_ROOT}"

SMOKE_JOB=$(sbatch --parsable \
  --job-name=iclr27_p22_variant_smoke \
  --account=EUHPC_D36_236 \
  --partition=boost_usr_prod \
  --qos=boost_qos_lprod \
  --time=01:00:00 \
  --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G \
  --array=0-5 \
  --output="${RUN_ROOT}/scheduler/smoke-%A_%a.out" \
  --error="${RUN_ROOT}/scheduler/smoke-%A_%a.err" \
  --export=ALL,SMOKE=1 \
  "${SCRIPT}")

FULL_JOB=$(sbatch --parsable \
  --job-name=iclr27_p22_model_variants \
  --account=EUHPC_D36_236 \
  --partition=boost_usr_prod \
  --qos=boost_qos_lprod \
  --time=06:00:00 \
  --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=80G \
  --array=0-5 \
  --dependency="afterok:${SMOKE_JOB}" \
  --output="${RUN_ROOT}/scheduler/full-%A_%a.out" \
  --error="${RUN_ROOT}/scheduler/full-%A_%a.err" \
  --export=ALL,SMOKE=0 \
  "${SCRIPT}")

echo "model_variant_smoke=${SMOKE_JOB}"
echo "model_variant_full=${FULL_JOB}"
