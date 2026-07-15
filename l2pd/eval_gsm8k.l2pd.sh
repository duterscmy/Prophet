#!/bin/bash
#SBATCH --job-name=eval_gsm8k
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=a100

source ~/.bashrc
conda activate ttrl_env
cd /mnt/fast/nobackup/scratch4weeks/mc03002/prophet/l2pd

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
export HF_ALLOW_CODE_EVAL="1"

mkdir -p logs
mkdir -p ../evals_results/l2pd

# export HF_ALLOW_CODE_EVAL=1
# export HF_DATASETS_TRUST_REMOTE_CODE=true


task=gsm8k_cot_zeroshot 
gen_length=256
block_length=32
method=L2P
accept_thres=0.96
accelerate launch eval_llada.py --tasks ${task} \
    --model llada_dist \
    --confirm_run_unsafe_code \
    --num_fewshot 0 \
    --log_samples \
    --output_path ../evals_results/l2pd/humaneval_l2pd_len${gen_length}_block${block_length}_thr${accept_thres} \
    --model_args model_path='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',gen_length=$gen_length,steps=$gen_length,block_length=$block_length,method=$method,accept_thres=$accept_thres \
    > ../logs/humaneval_l2pd-len${gen_length}-block${block_length}_thr${accept_thres}.log 2>&1
