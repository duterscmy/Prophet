#!/bin/bash
#SBATCH --job-name=eval_math500
#SBATCH --time=5:00:00
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
mkdir -p logs
mkdir -p evals_results/baseline

length=256
block=32
nshot=3

# Note: eval_llada.py loads the model with torch_dtype=torch.bfloat16 internally.
accelerate launch --num_processes 1 eval_llada.auto_thresh.py \
  --tasks minerva_math500 \
  --model llada_dist \
  --num_fewshot ${nshot} \
  --output_path evals_results/baseline/math500_standard_len${length}_block${block}_nshot${nshot} \
  --log_samples \
  --model_args model_path='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block},use_adaptive_parallel=false,use_dynamic_threshold=false \
  &> logs/math500_standard-len${length}-block${block}-nshot${nshot}.log