#!/bin/bash
#SBATCH --job-name="eval_human_eval_dynamic"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate dllm
cd "${TC_APD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

export HF_HOME=/projects/u6nc/public/mingyu/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0
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
  --model_args model_path=${MODEL_PATH:-/lus/lfs1aip2/projects/public/u6nc/mingyu/models/LLaDA-8B-Instruct},gen_length=${length},steps=${length},block_length=${block},decoder=tc_apd,dynamic_threshold_json=${threshold_json},max_threshold=${max_threshold},min_threshold=${min_threshold},default_threshold=${default_threshold},min_parallel_tokens=1 \
  &> logs/humaneval_dynamic_threshold_from_global_v2_c${correct_ratio}_mincount${min_count}_minaccepted${min_accepted}_len${length}_block${block}_maxthr${max_threshold}_minthr${min_threshold}.log