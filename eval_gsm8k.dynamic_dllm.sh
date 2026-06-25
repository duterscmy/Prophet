#!/bin/bash
#SBATCH --job-name=eval_gsm8k_prophet
#SBATCH --time=4:00:00
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
#  --limit 256
length=256

# constraints_text="200:The|201:answer|202:is"

accelerate launch eval_llada.py \
  --tasks gsm8k_cot_zeroshot \
  --model llada_dist \
  --num_fewshot 0 \
  --log_samples \
  --output_path evals_results/dynamic_dllm/gsm8k-ns0-${length} \
  --model_args model_path='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',enable_dynamic_dllm=true,gen_length=256,steps=256,block_length=32 \
  &> logs/dynamic_dllm-gsm8k_cot_zeroshot-ns0-${length}.log