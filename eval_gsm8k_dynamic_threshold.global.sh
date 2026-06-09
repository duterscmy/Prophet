#!/bin/bash
#SBATCH --job-name="eval_gsm8k_dynamic"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate dllm

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
export HF_HOME=/projects/u6nc/public/mingyu/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export CUDA_VISIBLE_DEVICES=0
export HF_ALLOW_CODE_EVAL="1"

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
  --tasks gsm8k_cot_zeroshot \
  --model llada_dist \
  --num_fewshot 0 \
  --output_path evals_results/auto_thresh/gsm8k_dynamic_from_global_v2_c${correct_ratio}_mincount${min_count}_minaccepted${min_accepted}_len${length}_block${block}_maxthr${max_threshold}_minthr${min_threshold} \
  --log_samples \
  --model_args model_path='/lus/lfs1aip2/projects/public/u6nc/mingyu/models/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block},use_dynamic_threshold=true,dynamic_threshold_json=${threshold_json},max_threshold=${max_threshold},min_threshold=${min_threshold},default_threshold=${default_threshold},min_parallel_tokens=1 \
  &> logs/gsm8k_dynamic_from_global_v2_c${correct_ratio}_mincount${min_count}_minaccepted${min_accepted}_len${length}_block${block}_maxthr${max_threshold}_minthr${min_threshold}.log