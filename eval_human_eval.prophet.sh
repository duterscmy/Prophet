#!/bin/bash
#SBATCH --job-name=eval_humaneval_prophet
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=a100

source ~/.bashrc
conda activate ttrl_env
cd /mnt/fast/nobackup/scratch4weeks/mc03002/prophet
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0
export HF_ALLOW_CODE_EVAL="1"

length=256

accelerate launch eval_llada.py \
  --tasks humaneval \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --log_samples \
  --output_path evals_results/prophet/humaneval-ns0-${length} \
  --model_args model_path='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',enable_early_exit=true,gen_length=256,steps=256,block_length=32,answer_length=5 \
  &> logs/prophet-humaneval-ns0-${length}.log