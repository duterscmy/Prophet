#!/bin/bash
#SBATCH --job-name="autoblock_eval"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

source ~/.bashrc
conda activate soar

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0

length=256

mkdir -p evals_results/autoblock
mkdir -p logs

accelerate launch --num_processes 1 eval_llada.auto_block.py \
  --tasks gsm8k_cot_zeroshot \
  --model llada_dist \
  --output_path evals_results/autoblock/gsm8k-length${length}-dense-gapblock \
  --log_samples \
  --model_args model_path='/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct',gen_length=${length},steps=${length},adaptive_candidate_mode=dense,gap_context_mode=block,use_length_compensation=false \
  &> logs/autoblock-gsm8k_cot_zeroshot-length${length}-dense-gapblock.log