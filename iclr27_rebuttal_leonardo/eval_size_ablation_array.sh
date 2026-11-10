#!/bin/bash

set -euo pipefail

: "${BENCHMARK:?BENCHMARK is required}"
: "${EXPERIMENT:?EXPERIMENT is required}"
: "${SLURM_ARRAY_TASK_ID:?Run this script as a Slurm array}"

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-size-ablation
MODEL_PATH=/leonardo_scratch/large/userexternal/mcao0000/models/LLaDA-8B-Instruct
THRESHOLD_ROOT="${RUN_ROOT}/thresholds"
SIZES=(100 200 500 1000 2000 5000)

source /leonardo/home/userexternal/mcao0000/miniconda3/etc/profile.d/conda.sh
conda activate tc-apd

export HF_HOME=/leonardo_scratch/large/userexternal/mcao0000/huggingface
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

case "${BENCHMARK}" in
  gsm8k)
    TASK_NAME=gsm8k_cot_zeroshot
    INCLUDE_ARGS=()
    UNSAFE_ARGS=()
    ;;
  math500)
    TASK_NAME=minerva_math500_leonardo
    INCLUDE_ARGS=(--include_path iclr27_rebuttal_leonardo/tasks)
    UNSAFE_ARGS=()
    ;;
  humaneval)
    TASK_NAME=humaneval_leonardo
    INCLUDE_ARGS=(--include_path iclr27_rebuttal_leonardo/tasks)
    UNSAFE_ARGS=(--confirm_run_unsafe_code)
    ;;
  *)
    echo "Unsupported BENCHMARK=${BENCHMARK}" >&2
    exit 2
    ;;
esac

case "${EXPERIMENT}" in
  calibration_size)
    SIZE="${SIZES[SLURM_ARRAY_TASK_ID]}"
    TABLE_PATH="${THRESHOLD_ROOT}/math_n${SIZE}_seed2027_token_threshold_grid_p99.5_mincount200_minaccepted100.json"
    MODEL_ARGS="model_path=${MODEL_PATH},gen_length=256,steps=256,block_length=32,decoder=tc_apd,threshold_path=${TABLE_PATH},max_threshold=0.90,min_threshold=0.01,default_threshold=0.90,min_parallel_tokens=1,print_detail_log=false"
    LOG_NAME="${BENCHMARK}_dynamic_threshold_from_math_n${SIZE}_seed2027_c99.5_mincount200_minaccepted100_len256_block32_maxthr0.90_minthr0.01.log"
    RESULT_NAME="${BENCHMARK}_tc_apd_math_n${SIZE}_seed2027"
    ;;
  baselines)
    METHODS=(standard fixed_apd)
    METHOD="${METHODS[SLURM_ARRAY_TASK_ID]}"
    if [[ "${METHOD}" == standard ]]; then
      MODEL_ARGS="model_path=${MODEL_PATH},gen_length=256,steps=256,block_length=32,decoder=standard,print_detail_log=false"
      LOG_NAME="${BENCHMARK}_standard_len256_block32.log"
      RESULT_NAME="${BENCHMARK}_standard"
    else
      MODEL_ARGS="model_path=${MODEL_PATH},gen_length=256,steps=256,block_length=32,decoder=fixed_apd,confidence_threshold=0.90,min_parallel_tokens=1,print_detail_log=false"
      LOG_NAME="${BENCHMARK}_fixed_threshold_thr0.90_len256_block32.log"
      RESULT_NAME="${BENCHMARK}_fixed_apd_thr0.90"
    fi
    ;;
  *)
    echo "Unsupported EXPERIMENT=${EXPERIMENT}" >&2
    exit 2
    ;;
esac

LOG_DIR="${RUN_ROOT}/logs/${BENCHMARK}"
RESULT_DIR="${RUN_ROOT}/results/${RESULT_NAME}"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"
cd "${PROJECT_ROOT}"

python eval_llada.auto_thresh.py \
  --tasks "${TASK_NAME}" \
  "${INCLUDE_ARGS[@]}" \
  "${UNSAFE_ARGS[@]}" \
  --model llada_dist \
  --num_fewshot 0 \
  --batch_size 1 \
  --output_path "${RESULT_DIR}" \
  --log_samples \
  --model_args "${MODEL_ARGS}" \
  &> "${LOG_DIR}/${LOG_NAME}"
