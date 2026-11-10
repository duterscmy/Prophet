#!/bin/bash

set -euo pipefail

: "${MODEL_VARIANT:?MODEL_VARIANT is required}"
: "${METHOD:?METHOD is required}"

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/p22-model-variants
BASE_MODEL=/leonardo_scratch/large/userexternal/mcao0000/models/LLaDA-8B-Instruct
THRESHOLD_PATH=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-size-ablation/thresholds/math_n5000_seed2027_token_threshold_grid_p99.5_mincount200_minaccepted100.json
SMOKE="${SMOKE:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
BENCHMARK="${BENCHMARK:-gsm8k}"

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

case "${MODEL_VARIANT}" in
  fine_tuned)
    MODEL_PATH=/leonardo_scratch/large/userexternal/mcao0000/models/LLaDA-Instruct-JustGRPO-GSM8K
    PYTHON=/leonardo/home/userexternal/mcao0000/miniconda3/envs/tc-apd/bin/python
    VARIANT_MODEL_ARGS="tokenizer_path=${BASE_MODEL}"
    ;;
  quantized_int8)
    MODEL_PATH=/leonardo_scratch/large/userexternal/mcao0000/models/llada-8b-instruct-8bit-gptq
    PYTHON=/leonardo_scratch/large/userexternal/mcao0000/envs/tc-apd-gptq/bin/python
    VARIANT_MODEL_ARGS="tokenizer_path=${BASE_MODEL},quantized=true"
    ;;
  *)
    echo "Unsupported MODEL_VARIANT=${MODEL_VARIANT}" >&2
    exit 2
    ;;
esac

case "${METHOD}" in
  standard)
    DECODER_ARGS="decoder=standard"
    ;;
  fixed_apd)
    DECODER_ARGS="decoder=fixed_apd,confidence_threshold=0.90,min_parallel_tokens=1"
    ;;
  tc_apd)
    DECODER_ARGS="decoder=tc_apd,threshold_path=${THRESHOLD_PATH},max_threshold=0.90,min_threshold=0.01,default_threshold=0.90,min_parallel_tokens=1"
    ;;
  *)
    echo "Unsupported METHOD=${METHOD}" >&2
    exit 2
    ;;
esac

export HF_HOME=/leonardo_scratch/large/userexternal/mcao0000/huggingface
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

SUFFIX=""
LIMIT_ARGS=()
if [[ "${SMOKE}" == "1" ]]; then
  SUFFIX="_smoke"
  LIMIT_ARGS=(--limit 2)
elif [[ -n "${EVAL_LIMIT}" ]]; then
  SUFFIX="_limit${EVAL_LIMIT}"
  LIMIT_ARGS=(--limit "${EVAL_LIMIT}")
fi

LOG_DIR="${RUN_ROOT}/logs/${MODEL_VARIANT}"
RESULT_DIR="${RUN_ROOT}/results/${BENCHMARK}_${MODEL_VARIANT}_${METHOD}_math_n5000_len256_block32${SUFFIX}"
LOG_PATH="${LOG_DIR}/${BENCHMARK}_${MODEL_VARIANT}_${METHOD}_math_n5000_len256_block32${SUFFIX}.log"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"
cd "${PROJECT_ROOT}"

"${PYTHON}" eval_llada.auto_thresh.py \
  --tasks "${TASK_NAME}" \
  "${INCLUDE_ARGS[@]}" \
  "${UNSAFE_ARGS[@]}" \
  --model llada_dist \
  --num_fewshot 0 \
  --batch_size 1 \
  --output_path "${RESULT_DIR}" \
  --log_samples \
  "${LIMIT_ARGS[@]}" \
  --model_args "model_path=${MODEL_PATH},${VARIANT_MODEL_ARGS},gen_length=256,steps=256,block_length=32,${DECODER_ARGS},print_detail_log=false" \
  &> "${LOG_PATH}"
