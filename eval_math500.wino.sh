#!/bin/bash
#SBATCH --job-name=eval_math500_wino
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
export CUDA_VISIBLE_DEVICES=0
export HF_ALLOW_CODE_EVAL="1"
#  --limit 256
length=256

accelerate launch eval_llada.wino.py \
  --tasks minerva_math500 \
  --model llada_dist \
  --num_fewshot 0 \
  --log_samples \
  --output_path evals_results/wino/math500-ns0-${length} \
  --model_args model_path='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct-wino',gen_length=${length},steps=${length},block_length=32  \
  &> logs/wino-math500-ns0-${length}.log