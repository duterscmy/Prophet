#!/bin/bash

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-size-ablation
SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/eval_size_ablation_array.sh"

mkdir -p "${RUN_ROOT}/scheduler" "${RUN_ROOT}/logs" "${RUN_ROOT}/results" "${RUN_ROOT}/thresholds"
cd "${PROJECT_ROOT}"

CALIBRATION_JOB=$(sbatch --parsable iclr27_rebuttal_leonardo/calibrate_math_size_ablation.sh)
echo "calibration=${CALIBRATION_JOB}"

submit_eval() {
  local benchmark=$1
  local experiment=$2
  local array=$3
  local time_limit=$4
  local qos=$5
  local dependency=${6:-}
  local dependency_args=()
  if [[ -n "${dependency}" ]]; then
    dependency_args=(--dependency="afterok:${dependency}")
  fi
  sbatch --parsable \
    --job-name="iclr27_${benchmark}_${experiment}" \
    --account=EUHPC_D36_236 \
    --partition=boost_usr_prod \
    --qos="${qos}" \
    --time="${time_limit}" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=80G \
    --array="${array}" \
    --output="${RUN_ROOT}/scheduler/${benchmark}-${experiment}-%A_%a.out" \
    --error="${RUN_ROOT}/scheduler/${benchmark}-${experiment}-%A_%a.err" \
    --export="ALL,BENCHMARK=${benchmark},EXPERIMENT=${experiment}" \
    "${dependency_args[@]}" \
    "${SCRIPT}"
}

echo "humaneval_baselines=$(submit_eval humaneval baselines 0-1 00:30:00 boost_qos_lprod)"
echo "math500_baselines=$(submit_eval math500 baselines 0-1 02:00:00 boost_qos_lprod)"
echo "gsm8k_baselines=$(submit_eval gsm8k baselines 0-1 04:00:00 boost_qos_lprod)"

echo "humaneval_sizes=$(submit_eval humaneval calibration_size 0-5 00:30:00 boost_qos_lprod "${CALIBRATION_JOB}")"
echo "math500_sizes=$(submit_eval math500 calibration_size 0-5 02:00:00 boost_qos_lprod "${CALIBRATION_JOB}")"
echo "gsm8k_sizes=$(submit_eval gsm8k calibration_size 0-5 04:00:00 boost_qos_lprod "${CALIBRATION_JOB}")"
