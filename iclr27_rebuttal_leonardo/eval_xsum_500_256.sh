#!/bin/bash

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?Run this script as a Slurm array}"

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-xsum-500
MODEL_PATH=/leonardo_scratch/large/userexternal/mcao0000/models/LLaDA-8B-Instruct
THRESHOLD_PATH=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-size-ablation/thresholds/math_n5000_seed2027_token_threshold_grid_p99.5_mincount200_minaccepted100.json
METHODS=(standard fixed_apd tc_apd)
METHOD="${METHODS[SLURM_ARRAY_TASK_ID]}"

case "${METHOD}" in
  standard)
    MODEL_ARGS="model_path=${MODEL_PATH},gen_length=256,steps=256,block_length=32,decoder=standard,print_detail_log=false"
    LOG_NAME=xsum_500_standard_len256_block32_chattemplate.log
    ;;
  fixed_apd)
    MODEL_ARGS="model_path=${MODEL_PATH},gen_length=256,steps=256,block_length=32,decoder=fixed_apd,confidence_threshold=0.90,min_parallel_tokens=1,print_detail_log=false"
    LOG_NAME=xsum_500_fixed_threshold_thr0.90_len256_block32_chattemplate.log
    ;;
  tc_apd)
    MODEL_ARGS="model_path=${MODEL_PATH},gen_length=256,steps=256,block_length=32,decoder=tc_apd,threshold_path=${THRESHOLD_PATH},max_threshold=0.90,min_threshold=0.01,default_threshold=0.90,min_parallel_tokens=1,print_detail_log=false"
    LOG_NAME=xsum_500_dynamic_threshold_from_math_n5000_seed2027_c99.5_mincount200_minaccepted100_len256_block32_chattemplate_maxthr0.90_minthr0.01.log
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
RESULT_DIR="${RUN_ROOT}/results/xsum_500_${METHOD}_len256_block32_chattemplate"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"
cd "${PROJECT_ROOT}"

python eval_llada.auto_thresh.py \
  --tasks xsum_500_leonardo \
  --include_path iclr27_rebuttal_leonardo/tasks \
  --model llada_dist \
  --num_fewshot 0 \
  --batch_size 1 \
  --apply_chat_template \
  --output_path "${RESULT_DIR}" \
  --log_samples \
  --model_args "${MODEL_ARGS}" \
  &> "${LOG_DIR}/${LOG_NAME}"
