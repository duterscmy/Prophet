#!/bin/bash
#SBATCH --job-name="soar_eval"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                # 请求2块GPU
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate soar

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0
export HF_ALLOW_CODE_EVAL="1"
#  --limit 256

accelerate launch eval_llada.py \
  --tasks humaneval \
  --confirm_run_unsafe_code --model llada_dist \
  --output_path evals_results/prophet_soar/humaneval-ns0-${length} --log_samples \
  --model_args model_path='/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct',enable_early_exit=true,enable_soar=true,constraints_text="200:The|201:answer|202:is",gen_length=256,steps=256,block_length=32,answer_length=5 \