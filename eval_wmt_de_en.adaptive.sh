#!/bin/bash
#SBATCH --job-name=eval_wmt_de_en_adaptive
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=a100

source ~/.bashrc
conda activate ttrl_env
cd "${TC_APD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0

mkdir -p logs
mkdir -p evals_results/adaptive_parallel

length=64
block=32
threshold=0.80

accelerate launch --num_processes 1 eval_llada.auto_thresh.py \
  --tasks wmt16-de-en \
  --model llada_dist \
  --output_path evals_results/adaptive_parallel/wmt_de_en_adaptive_len${length}_block${block}_thr${threshold} \
  --log_samples \
  --model_args model_path=${MODEL_PATH:-/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct},gen_length=${length},steps=${length},block_length=${block},decoder=fixed_apd,confidence_threshold=${threshold},min_parallel_tokens=1 \
  &> logs/wmt_de_en_adaptive_len${length}_block${block}_thr${threshold}.log