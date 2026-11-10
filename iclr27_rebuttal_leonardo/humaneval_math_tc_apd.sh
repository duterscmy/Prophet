#!/bin/bash
#SBATCH --job-name=iclr27_he_math
#SBATCH --account=EUHPC_D36_236
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/humaneval-math-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/humaneval-math-%j.err

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs
MODEL_PATH=/leonardo_scratch/large/userexternal/mcao0000/models/LLaDA-8B-Instruct
TABLE_PATH="${PROJECT_ROOT}/token_threshold_on_trainset/math_train_token_threshold_grid_p99.5_mincount200_minaccepted100.json"
RUN_DIR="${RUN_ROOT}/humaneval-math-${SLURM_JOB_ID}"

source /leonardo/home/userexternal/mcao0000/miniconda3/etc/profile.d/conda.sh
conda activate tc-apd

export HF_HOME=/leonardo_scratch/large/userexternal/mcao0000/huggingface
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${RUN_DIR}"
cd "${PROJECT_ROOT}"

python eval_llada.auto_thresh.py \
  --tasks humaneval_leonardo \
  --include_path iclr27_rebuttal_leonardo/tasks \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --num_fewshot 0 \
  --batch_size 1 \
  --output_path "${RUN_DIR}/results" \
  --log_samples \
  --model_args "model_path=${MODEL_PATH},gen_length=256,steps=256,block_length=32,decoder=tc_apd,threshold_path=${TABLE_PATH},max_threshold=0.90,min_threshold=0.01,default_threshold=0.90,min_parallel_tokens=1,print_detail_log=false"
