#!/bin/bash
#SBATCH --job-name="soar_gsm8k"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1               # 请求2块GPU
#SBATCH --time=3:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate soar

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0
export HF_ALLOW_CODE_EVAL="1"
length=256

accelerate launch eval_llada.py \
  --tasks gsm8k_cot_zeroshot \
  --model llada_dist \
  --num_fewshot 0 \
  --log_samples \
  --output_path evals_results/soar/gsm8k_cot_zeroshot-ns0-${length}\
  --model_args model_path='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',enable_early_exit=false,enable_soar=true,gen_length=${length},steps=${length},block_length=32 &> logs/soar0995-gsm8k_cot_zeroshot-ns0-${length}.log