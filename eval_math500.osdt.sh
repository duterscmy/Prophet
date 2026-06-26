#!/bin/bash
#SBATCH --job-name=eval_math500_osdt
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

length=256

accelerate launch eval_llada.py \
  --tasks minerva_math500 \
  --model llada_dist \
  --num_fewshot 0 \
  --log_samples \
  --output_path evals_results/osdt/math500-ns0-${length} \
  --model_args model_path='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',enable_osdt=true,osdt_threshold_cap=0.75,osdt_epsilon_ratio=0.20,gen_length=256,steps=256,block_length=32 \
  &> logs/osdt-math500_cot_zeroshot-ns0-${length}.log