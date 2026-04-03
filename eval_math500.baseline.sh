#!/bin/bash
#SBATCH --job-name="soar_eval"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # 请求2块GPU
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate soar

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0
#  --limit 256
length=256

accelerate launch eval_llada.py \
  --tasks minerva_math500 \
  --model llada_dist \
  --num_fewshot 0 \
  --output_path evals_results/baseline/math500-ns0-${length} --log_samples \
  --model_args model_path='/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct',enable_early_exit=false,enable_soar=false,gen_length=${length},steps=${length},block_length=32,answer_length=5 &> logs/baseline-math500-ns0-${length}.log