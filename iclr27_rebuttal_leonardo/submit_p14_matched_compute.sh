#!/bin/bash

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-matched-compute
EVAL_SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/eval_p14_fixed077.sh"
ANALYSIS_SCRIPT="${PROJECT_ROOT}/iclr27_rebuttal_leonardo/run_p14_matched_compute_analysis.sh"
mkdir -p "${RUN_ROOT}/scheduler" "${RUN_ROOT}/logs" "${RUN_ROOT}/results"
cd "${PROJECT_ROOT}"

submit_eval() {
  local benchmark=$1
  local time_limit=$2
  sbatch --parsable \
    --job-name="iclr27_p14_${benchmark}_fix077" \
    --account=EUHPC_D36_236 \
    --partition=boost_usr_prod \
    --qos=boost_qos_lprod \
    --time="${time_limit}" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=80G \
    --output="${RUN_ROOT}/scheduler/${benchmark}-%j.out" \
    --error="${RUN_ROOT}/scheduler/${benchmark}-%j.err" \
    --export="ALL,BENCHMARK=${benchmark}" \
    "${EVAL_SCRIPT}"
}

HUMANEVAL_JOB=$(submit_eval humaneval 00:30:00)
MATH500_JOB=$(submit_eval math500 02:00:00)
GSM8K_JOB=$(submit_eval gsm8k 04:00:00)
echo "p14_humaneval_fixed077=${HUMANEVAL_JOB}"
echo "p14_math500_fixed077=${MATH500_JOB}"
echo "p14_gsm8k_fixed077=${GSM8K_JOB}"

ANALYSIS_JOB=$(sbatch --parsable \
  --dependency="afterok:${HUMANEVAL_JOB}:${MATH500_JOB}:${GSM8K_JOB}" \
  --output="${RUN_ROOT}/scheduler/analysis-%j.out" \
  --error="${RUN_ROOT}/scheduler/analysis-%j.err" \
  "${ANALYSIS_SCRIPT}")
echo "p14_paired_bootstrap_analysis=${ANALYSIS_JOB}"
