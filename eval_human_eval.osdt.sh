#!/bin/bash
#SBATCH --job-name=eval_human_eval_osdt
#SBATCH --time=1:00:00
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
export HF_ALLOW_CODE_EVAL="1"

length=256

accelerate launch eval_llada.py \
  --tasks humaneval \
  --model llada_dist \
  --num_fewshot 0 \
  --log_samples \
  --output_path evals_results/osdt/human_eval-ns0-${length} \
  --model_args model_path='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',enable_osdt=true,osdt_threshold_cap=0.80,osdt_epsilon_ratio=0.10,gen_length=256,steps=256,block_length=32 \
  &> logs/osdt-human_eval_cot_zeroshot-ns0-${length}.log