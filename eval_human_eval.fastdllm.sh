#!/bin/bash
#SBATCH --job-name="soar_eval"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1               # 请求2块GPU
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

task=humaneval
length=256
block_length=32
steps=256
mkdir -p logs

TORCHINDUCTOR_DISABLED=1 accelerate launch eval_llada.fastdllm.py --tasks humaneval \
  --confirm_run_unsafe_code --model llada_dist \
  --model_args model_path='/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True \
  --output_path evals_results/cache_parallel/humaneval-ns0-${length} --log_samples &> logs/cache_parallel-humaneval-ns0-${length}.log


TORCHINDUCTOR_DISABLED=1 accelerate launch eval_llada.fastdllm.py --tasks mbpp \
  --confirm_run_unsafe_code --model llada_dist \
  --model_args model_path='/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,show_speed=True \
  --output_path evals_results/cache_parallel/mbpp-ns0-${length} --log_samples &> logs/cache_parallel-mbpp-ns0-${length}.log