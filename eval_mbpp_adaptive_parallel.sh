#!/bin/bash
#SBATCH --job-name=eval_mbpp
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
export HF_ALLOW_CODE_EVAL="1"

mkdir -p logs
mkdir -p evals_results/adaptive_parallel

length=256
block=32
threshold=0.90

# Note: eval_llada.py loads the model with torch_dtype=torch.bfloat16 internally.
accelerate launch --num_processes 1 eval_llada.auto_thresh.py \
  --tasks mbpp \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --num_fewshot 3 \
  --output_path evals_results/adaptive_parallel/mbpp_adaptive_parallel_len${length}_block${block}_thr${threshold} \
  --log_samples \
  --model_args model_path='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block},use_adaptive_parallel=true,use_dynamic_threshold=false,confidence_threshold=${threshold},min_parallel_tokens=1 \
  &> logs/mbpp_adaptive_parallel-len${length}-block${block}_thr${threshold}.log