#!/bin/bash
#SBATCH --job-name=run_ace
#SBATCH --time=24:00:00
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
mkdir -p evals_results/baseline

length=256
block=32

# Note: eval_llada.py loads the model with torch_dtype=torch.bfloat16 internally.
accelerate launch --num_processes 1 eval_llada.auto_thresh.py \
  --include_path local_tasks \
  --tasks acecode_generate \
  --confirm_run_unsafe_code \
  --limit 5000 \
  --model llada_dist \
  --num_fewshot 0 \
  --output_path evals_results/baseline/acecode_standard_len${length}_block${block} \
  --log_samples \
  --model_args model_path='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block},use_adaptive_parallel=false,use_dynamic_threshold=false,print_detail_log=true \
  &> logs/acecode_standard-len${length}-block${block}.log
