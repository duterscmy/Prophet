#!/bin/bash
#SBATCH --job-name=eval_humaneval_dynamic
#SBATCH --time=00:30:00
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
mkdir -p evals_results/dynamic_threshold

length=256
block=32
correct_ratio=99.5  #99.0/99.5/100.0
max_threshold=0.80
min_threshold=0.01
default_threshold=$max_threshold
min_count=200
min_accepted=100
threshold_json="token_threshold_on_trainset/global_v2_token_threshold_grid_p${correct_ratio}_mincount${min_count}_minaccepted${min_accepted}.json"


# Note: eval_llada.py loads the model with torch_dtype=torch.bfloat16 internally.
accelerate launch --num_processes 1 eval_llada.auto_thresh.py \
  --tasks humaneval \
  --confirm_run_unsafe_code \
  --model llada_dist \
  --num_fewshot 0 \
  --output_path evals_results/dynamic_threshold/humaneval_dynamic_threshold_from_global_v2_c${correct_ratio}_mincount${min_count}_minaccepted${min_accepted}_len${length}_block${block}_maxthr${max_threshold}_minthr${min_threshold} \
  --log_samples \
  --model_args model_path='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block},use_adaptive_parallel=true,use_dynamic_threshold=true,dynamic_threshold_json=${threshold_json},max_threshold=${max_threshold},min_threshold=${min_threshold},default_threshold=${default_threshold},min_parallel_tokens=1 \
  &> logs/humaneval_dynamic_threshold_from_global_v2_c${correct_ratio}_mincount${min_count}_minaccepted${min_accepted}_len${length}_block${block}_maxthr${max_threshold}_minthr${min_threshold}.log