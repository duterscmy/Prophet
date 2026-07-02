#!/bin/bash
#SBATCH --job-name=eval_wmt_de_en_baseline
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=3090

source ~/.bashrc
conda activate ttrl_env
cd /mnt/fast/nobackup/scratch4weeks/mc03002/prophet
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
mkdir -p logs
mkdir -p evals_results/baseline

length=64
block=32

accelerate launch --num_processes 1 eval_llada.auto_thresh.py \
  --tasks wmt16-de-en \
  --model llada_dist \
  --output_path evals_results/baseline/wmt_de_en_baseline_len${length}_block${block}_4gpu \
  --log_samples \
  --model_args model_path='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block},use_adaptive_parallel=false,use_dynamic_threshold=false \
  &> logs/baseline-wmt_de_en_baseline-len${length}-block${block}-4gpu.log