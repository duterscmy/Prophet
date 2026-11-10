#!/bin/bash

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-ifeval
SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/eval_ifeval_fixed_length.sh"

mkdir -p "${RUN_ROOT}/scheduler" "${RUN_ROOT}/logs" "${RUN_ROOT}/results"
cd "${PROJECT_ROOT}"

submit_one() {
  local length=$1
  local time_limit=$2
  sbatch --parsable \
    --job-name="iclr27_ifeval_fix09_l${length}" \
    --account=EUHPC_D36_236 \
    --partition=boost_usr_prod \
    --qos=boost_qos_lprod \
    --time="${time_limit}" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=80G \
    --output="${RUN_ROOT}/scheduler/ifeval-fixed09-len${length}-%j.out" \
    --error="${RUN_ROOT}/scheduler/ifeval-fixed09-len${length}-%j.err" \
    --export="ALL,GEN_LENGTH=${length}" \
    "${SCRIPT}"
}

echo "ifeval_fixed09_len256=$(submit_one 256 02:00:00)"
echo "ifeval_fixed09_len512=$(submit_one 512 04:00:00)"
