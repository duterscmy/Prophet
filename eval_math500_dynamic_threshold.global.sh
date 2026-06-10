#!/bin/bash
#SBATCH --job-name=eval_math500_dynamic
#SBATCH --time=1:00:00
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
mkdir -p evals_results/auto_thresh

length=256
block=32
correct_ratio=99.5 #99 99.5 99.7 100
max_threshold=0.90
min_threshold=0.05
default_threshold=$max_threshold

min_count=400
min_accepted=200
threshold_json="token_threshold_on_trainset/global_v2_token_threshold_grid_p${correct_ratio}_mincount${min_count}_minaccepted${min_accepted}.json"

ls $threshold_json || (echo "Threshold json file not found: ${threshold_json}" && exit 1)

accelerate launch --num_processes 1 eval_llada.auto_thresh.py \
  --tasks minerva_math500 \
  --model llada_dist \
  --num_fewshot 0 \
  --output_path evals_results/auto_thresh/math500_dynamic_from_global_v2_c${correct_ratio}_mincount${min_count}_minaccepted${min_accepted}_len${length}_block${block}_maxthr${max_threshold}_minthr${min_threshold} \
  --log_samples \
  --model_args model_path='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block},use_dynamic_threshold=true,dynamic_threshold_json=${threshold_json},max_threshold=${max_threshold},min_threshold=${min_threshold},default_threshold=${default_threshold},min_parallel_tokens=1 \
  &> logs/math500_dynamic_from_global_v2_c${correct_ratio}_mincount${min_count}_minaccepted${min_accepted}_len${length}_block${block}_maxthr${max_threshold}_minthr${min_threshold}.log