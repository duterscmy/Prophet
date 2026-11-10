#!/bin/bash

set -euo pipefail

: "${BENCHMARK:?BENCHMARK is required}"
: "${SLURM_ARRAY_TASK_ID:?Run this script as a Slurm array}"

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
MODEL_PATH=/leonardo_scratch/large/userexternal/mcao0000/models/LLaDA-8B-Instruct
THRESHOLD_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-size-ablation/thresholds
SIZES=(100 200 500 1000 2000)
SIZE="${SIZES[SLURM_ARRAY_TASK_ID]}"
THRESHOLD_PATH="${THRESHOLD_ROOT}/math_n${SIZE}_seed2027_token_threshold_grid_p99.5_mincount200_minaccepted100.json"

case "${BENCHMARK}" in
  arc_challenge)
    RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-arc-challenge
    TASK_NAME=arc_challenge_chat
    INCLUDE_ARGS=()
    LOG_STEM="arc_challenge_dynamic_threshold_from_math_n${SIZE}_seed2027_c99.5_mincount200_minaccepted100_len256_block32_chattemplate_maxthr0.90_minthr0.01"
    RESULT_NAME="arc_challenge_tc_apd_math_n${SIZE}_seed2027_len256_block32_chattemplate"
    ;;
  xsum500)
    RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-xsum-500
    TASK_NAME=xsum_500_leonardo
    INCLUDE_ARGS=(--include_path iclr27_rebuttal_leonardo/tasks)
    LOG_STEM="xsum_500_dynamic_threshold_from_math_n${SIZE}_seed2027_c99.5_mincount200_minaccepted100_len256_block32_chattemplate_maxthr0.90_minthr0.01"
    RESULT_NAME="xsum_500_tc_apd_math_n${SIZE}_seed2027_len256_block32_chattemplate"
    ;;
  *)
    echo "Unsupported BENCHMARK=${BENCHMARK}" >&2
    exit 2
    ;;
esac

source /leonardo/home/userexternal/mcao0000/miniconda3/etc/profile.d/conda.sh
conda activate tc-apd
export HF_HOME=/leonardo_scratch/large/userexternal/mcao0000/huggingface
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

LOG_DIR="${RUN_ROOT}/logs"
RESULT_DIR="${RUN_ROOT}/results/${RESULT_NAME}"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"
cd "${PROJECT_ROOT}"

python eval_llada.auto_thresh.py \
  --tasks "${TASK_NAME}" \
  "${INCLUDE_ARGS[@]}" \
  --model llada_dist \
  --num_fewshot 0 \
  --batch_size 1 \
  --apply_chat_template \
  --output_path "${RESULT_DIR}" \
  --log_samples \
  --model_args "model_path=${MODEL_PATH},gen_length=256,steps=256,block_length=32,decoder=tc_apd,threshold_path=${THRESHOLD_PATH},max_threshold=0.90,min_threshold=0.01,default_threshold=0.90,min_parallel_tokens=1,print_detail_log=false" \
  &> "${LOG_DIR}/${LOG_STEM}.log"
