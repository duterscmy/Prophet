#!/bin/bash

set -euo pipefail

: "${BENCHMARK:?BENCHMARK is required}"
: "${SLURM_ARRAY_TASK_ID:?Run this script as a Slurm array}"

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/p22-transfer
MODEL_PATH=/leonardo_scratch/large/userexternal/mcao0000/models/LLaDA-8B-Instruct
THRESHOLD_PATH=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-size-ablation/thresholds/math_n5000_seed2027_token_threshold_grid_p99.5_mincount200_minaccepted100.json

CONFIGS=(len256_block16 len256_block8 len512_block32 len256_block32_chattemplate)
METHODS=(standard fixed_apd tc_apd)
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
CONFIG="${CONFIGS[CONFIG_INDEX]}"
METHOD="${METHODS[METHOD_INDEX]}"

GEN_LENGTH=256
BLOCK_LENGTH=32
CHAT_ARGS=()
case "${CONFIG}" in
  len256_block16)
    BLOCK_LENGTH=16
    ;;
  len256_block8)
    BLOCK_LENGTH=8
    ;;
  len512_block32)
    GEN_LENGTH=512
    ;;
  len256_block32_chattemplate)
    CHAT_ARGS=(--apply_chat_template)
    ;;
  *)
    echo "Unsupported CONFIG=${CONFIG}" >&2
    exit 2
    ;;
esac

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

case "${METHOD}" in
  standard)
    MODEL_ARGS="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${GEN_LENGTH},block_length=${BLOCK_LENGTH},decoder=standard,print_detail_log=false"
    ;;
  fixed_apd)
    MODEL_ARGS="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${GEN_LENGTH},block_length=${BLOCK_LENGTH},decoder=fixed_apd,confidence_threshold=0.90,min_parallel_tokens=1,print_detail_log=false"
    ;;
  tc_apd)
    MODEL_ARGS="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${GEN_LENGTH},block_length=${BLOCK_LENGTH},decoder=tc_apd,threshold_path=${THRESHOLD_PATH},max_threshold=0.90,min_threshold=0.01,default_threshold=0.90,min_parallel_tokens=1,print_detail_log=false"
    ;;
esac

source /leonardo/home/userexternal/mcao0000/miniconda3/etc/profile.d/conda.sh
conda activate tc-apd

export HF_HOME=/leonardo_scratch/large/userexternal/mcao0000/huggingface
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

LOG_DIR="${RUN_ROOT}/logs/${BENCHMARK}"
RESULT_NAME="${BENCHMARK}_${METHOD}_${CONFIG}_math_n5000_transfer"
RESULT_DIR="${RUN_ROOT}/results/${RESULT_NAME}"
LOG_PATH="${LOG_DIR}/${RESULT_NAME}.log"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"
cd "${PROJECT_ROOT}"

python eval_llada.auto_thresh.py \
  --tasks "${TASK_NAME}" \
  "${INCLUDE_ARGS[@]}" \
  "${UNSAFE_ARGS[@]}" \
  "${CHAT_ARGS[@]}" \
  --model llada_dist \
  --num_fewshot 0 \
  --batch_size 1 \
  --output_path "${RESULT_DIR}" \
  --log_samples \
  --model_args "${MODEL_ARGS}" \
  &> "${LOG_PATH}"
