#!/bin/bash

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-writingprompts-100-len1024
SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/eval_writingprompts_100_len1024.sh"

mkdir -p "${RUN_ROOT}/scheduler" "${RUN_ROOT}/logs" "${RUN_ROOT}/results"
cd "${PROJECT_ROOT}"

JOB=$(sbatch --parsable \
  --job-name=iclr27_wp100_l1024 \
  --account=EUHPC_D36_236 \
  --partition=boost_usr_prod \
  --qos=boost_qos_lprod \
  --time=24:00:00 \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=80G \
  --array=0-2 \
  --output="${RUN_ROOT}/scheduler/writingprompts100-%A_%a.out" \
  --error="${RUN_ROOT}/scheduler/writingprompts100-%A_%a.err" \
  "${SCRIPT}")

echo "writingprompts100_len1024_standard_fixed09_tc_apd=${JOB}"
