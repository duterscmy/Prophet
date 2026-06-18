#!/bin/bash
#SBATCH --job-name=eval_math_prophet
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

# constraints_text="200:The|201:answer|202:is"

accelerate launch eval_llada.py \
  --tasks minerva_math500 \
  --model llada_dist \
  --num_fewshot 0 \
  --log_samples \
  --output_path evals_results/prophet/math500-ns0-${length} \
  --model_args model_path='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',enable_early_exit=true,constraints_text="200:The|201:answer|202:is",gen_length=256,steps=256,block_length=32,answer_length=5 \
  &> logs/prophet-math500-ns0-${length}.log