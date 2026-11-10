#!/bin/bash

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-ifeval
SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/eval_ifeval_setting_array.sh"

mkdir -p "${RUN_ROOT}/scheduler" "${RUN_ROOT}/logs" "${RUN_ROOT}/results"
cd "${PROJECT_ROOT}"

STANDARD_JOB=$(sbatch --parsable \
  --job-name=iclr27_ifeval_standard_l256 \
  --account=EUHPC_D36_236 \
  --partition=boost_usr_prod \
  --qos=boost_qos_lprod \
  --time=03:00:00 \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=80G \
  --output="${RUN_ROOT}/scheduler/ifeval-standard-len256-%j.out" \
  --error="${RUN_ROOT}/scheduler/ifeval-standard-len256-%j.err" \
  --export=ALL,EXPERIMENT=standard \
  "${SCRIPT}")

SIZE_JOB=$(sbatch --parsable \
  --job-name=iclr27_ifeval_sizes_l256 \
  --account=EUHPC_D36_236 \
  --partition=boost_usr_prod \
  --qos=boost_qos_lprod \
  --time=02:00:00 \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=80G \
  --array=0-5 \
  --output="${RUN_ROOT}/scheduler/ifeval-sizes-len256-%A_%a.out" \
  --error="${RUN_ROOT}/scheduler/ifeval-sizes-len256-%A_%a.err" \
  --export=ALL,EXPERIMENT=calibration_size \
  "${SCRIPT}")

echo "ifeval_standard_len256=${STANDARD_JOB}"
echo "ifeval_math_sizes_len256=${SIZE_JOB}"
echo "ifeval_fixed09_len256=63853656 (completed)"
