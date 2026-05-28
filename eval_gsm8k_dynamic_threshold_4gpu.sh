#!/bin/bash
#SBATCH --job-name=eval_gsm8k_dynamic
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=3090

source ~/.bashrc
conda activate ttrl_env

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0

mkdir -p logs
mkdir -p evals_results/auto_thresh

length=256
block=32
correct_ratio=95
max_threshold=0.95
min_threshold=0.05
default_threshold=0.95

threshold_json="token_threshold_reliability_fast/token_threshold_grid_c${correct_ratio}_mincount50.json"

accelerate launch --num_processes 1 eval_llada.auto_thresh.py \
  --tasks gsm8k_cot_zeroshot \
  --model llada_dist \
  --output_path evals_results/auto_thresh/gsm8k_dynamic_c${correct_ratio}_len${length}_block${block}_maxthr${max_threshold}_minthr${min_threshold} \
  --log_samples \
  --model_args model_path='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block},use_dynamic_threshold=true,dynamic_threshold_json=${threshold_json},max_threshold=${max_threshold},min_threshold=${min_threshold},default_threshold=${default_threshold},min_parallel_tokens=1 \
  &> logs/gsm8k_dynamic_c${correct_ratio}_len${length}_block${block}_maxthr${max_threshold}_minthr${min_threshold}.log