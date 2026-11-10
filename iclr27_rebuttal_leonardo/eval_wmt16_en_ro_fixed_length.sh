#!/bin/bash

set -euo pipefail

: "${GEN_LENGTH:?GEN_LENGTH is required (256 or 512)}"
if [[ "${GEN_LENGTH}" != "256" && "${GEN_LENGTH}" != "512" ]]; then
  echo "Unsupported GEN_LENGTH=${GEN_LENGTH}; expected 256 or 512" >&2
  exit 2
fi

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-wmt16-en-ro
MODEL_PATH=/leonardo_scratch/large/userexternal/mcao0000/models/LLaDA-8B-Instruct
BLOCK_LENGTH=32
LOG_DIR="${RUN_ROOT}/logs"
RESULT_DIR="${RUN_ROOT}/results/wmt16_en_ro_fixed_apd_thr0.90_len${GEN_LENGTH}_block${BLOCK_LENGTH}"
LOG_PATH="${LOG_DIR}/wmt16_en_ro_fixed_threshold_thr0.90_len${GEN_LENGTH}_block${BLOCK_LENGTH}.log"

source /leonardo/home/userexternal/mcao0000/miniconda3/etc/profile.d/conda.sh
conda activate tc-apd

export HF_HOME=/leonardo_scratch/large/userexternal/mcao0000/huggingface
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

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
  --model_args "model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${GEN_LENGTH},block_length=${BLOCK_LENGTH},decoder=fixed_apd,confidence_threshold=0.90,min_parallel_tokens=1,print_detail_log=false" \
  &> "${LOG_PATH}"
