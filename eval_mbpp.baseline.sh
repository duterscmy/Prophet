#!/bin/bash
#SBATCH --job-name="mbpp_std_eval"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

source ~/.bashrc
conda activate soar

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ALLOW_CODE_EVAL="1"

mkdir -p logs
mkdir -p evals_results/baseline

length=256
block=32

# Note: eval_llada.py loads the model with torch_dtype=torch.bfloat16 internally.
accelerate launch --num_processes 4 eval_llada.auto_thresh.py \
  --tasks mbpp \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --num_fewshot 0 \
  --output_path evals_results/baseline/mbpp_standard_len${length}_block${block}_4gpu_bf16 \
  --log_samples \
  --model_args model_path='/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block},use_adaptive_parallel=false,use_dynamic_threshold=false \
  &> logs/baseline-mbpp_standard-len${length}-block${block}-4gpu-bf16.log