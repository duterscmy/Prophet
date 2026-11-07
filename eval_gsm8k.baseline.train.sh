#!/bin/bash
#SBATCH --job-name=eval_gsm8k_base
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=a100

source ~/.bashrc
conda activate ttrl_env
cd "${TC_APD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
mkdir -p logs
mkdir -p evals_results/baseline

length=256
block=32

accelerate launch --num_processes 1 eval_llada.auto_thresh.py \
  --tasks gsm8k_cot_zeroshot_train \
  --model llada_dist \
  --output_path evals_results/baseline/gsm8k_train_standard_len${length}_block${block}_4gpu \
  --log_samples \
  --model_args model_path=${MODEL_PATH:-/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct},gen_length=${length},steps=${length},block_length=${block},decoder=standard,print_detail_log=true \
  &> logs/baseline-gsm8k_train_standard-len${length}-block${block}.log