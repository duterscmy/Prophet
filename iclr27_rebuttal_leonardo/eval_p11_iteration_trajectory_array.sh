#!/bin/bash

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?Run this script as a Slurm array}"

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-p11-iteration-trajectories
MODEL_PATH=/leonardo_scratch/large/userexternal/mcao0000/models/LLaDA-8B-Instruct
THRESHOLD_PATH=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-size-ablation/thresholds/math_n5000_seed2027_token_threshold_grid_p99.5_mincount200_minaccepted100.json
METHODS=(standard tc_apd)
METHOD="${METHODS[SLURM_ARRAY_TASK_ID]}"

case "${METHOD}" in
  standard)
    MODEL_ARGS="model_path=${MODEL_PATH},gen_length=256,steps=256,block_length=32,decoder=standard,print_detail_log=true"
    ;;
  tc_apd)
    MODEL_ARGS="model_path=${MODEL_PATH},gen_length=256,steps=256,block_length=32,decoder=tc_apd,threshold_path=${THRESHOLD_PATH},max_threshold=0.90,min_threshold=0.01,default_threshold=0.90,min_parallel_tokens=1,print_detail_log=true"
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
RESULT_DIR="${RUN_ROOT}/results/gsm8k_${METHOD}_limit500_len256_block32_traces"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"
cd "${PROJECT_ROOT}"

python eval_llada.auto_thresh.py \
  --tasks gsm8k_cot_zeroshot \
  --model llada_dist \
  --num_fewshot 0 \
  --batch_size 1 \
  --limit 500 \
  --output_path "${RESULT_DIR}" \
  --log_samples \
  --model_args "${MODEL_ARGS}" \
  &> "${LOG_DIR}/gsm8k_${METHOD}_limit500_len256_block32_iteration_traces.log"
