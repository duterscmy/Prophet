#!/bin/bash

set -euo pipefail

: "${EXPERIMENT:?EXPERIMENT is required (standard or calibration_size)}"

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-wmt16-en-ro
MODEL_PATH=/leonardo_scratch/large/userexternal/mcao0000/models/LLaDA-8B-Instruct
THRESHOLD_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-size-ablation/thresholds
GEN_LENGTH=256
BLOCK_LENGTH=32
SIZES=(100 200 500 1000 2000 5000)

case "${EXPERIMENT}" in
  standard)
    MODEL_ARGS="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${GEN_LENGTH},block_length=${BLOCK_LENGTH},decoder=standard,print_detail_log=false"
    LOG_NAME="wmt16_en_ro_standard_len256_block32.log"
    RESULT_NAME=wmt16_en_ro_standard_len256_block32
    ;;
  calibration_size)
    : "${SLURM_ARRAY_TASK_ID:?calibration_size requires a Slurm array}"
    SIZE="${SIZES[SLURM_ARRAY_TASK_ID]}"
    TABLE_PATH="${THRESHOLD_ROOT}/math_n${SIZE}_seed2027_token_threshold_grid_p99.5_mincount200_minaccepted100.json"
    MODEL_ARGS="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${GEN_LENGTH},block_length=${BLOCK_LENGTH},decoder=tc_apd,threshold_path=${TABLE_PATH},max_threshold=0.90,min_threshold=0.01,default_threshold=0.90,min_parallel_tokens=1,print_detail_log=false"
    LOG_NAME="wmt16_en_ro_dynamic_threshold_from_math_n${SIZE}_seed2027_c99.5_mincount200_minaccepted100_len256_block32_maxthr0.90_minthr0.01.log"
    RESULT_NAME="wmt16_en_ro_tc_apd_math_n${SIZE}_seed2027_len256_block32"
    ;;
  *)
    echo "Unsupported EXPERIMENT=${EXPERIMENT}" >&2
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
  --tasks wmt16_en_ro_leonardo \
  --include_path iclr27_rebuttal_leonardo/tasks \
  --model llada_dist \
  --num_fewshot 0 \
  --batch_size 1 \
  --output_path "${RESULT_DIR}" \
  --log_samples \
  --model_args "${MODEL_ARGS}" \
  &> "${LOG_DIR}/${LOG_NAME}"
