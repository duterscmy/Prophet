#!/bin/bash
#SBATCH --job-name="gsm8k_eval"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

source ~/.bashrc
conda activate soar

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p logs
mkdir -p evals_results/baseline

length=256
block=32

accelerate launch --num_processes 4 eval_llada.py \
  --tasks gsm8k_cot_zeroshot \
  --model llada_dist \
  --output_path evals_results/baseline/gsm8k_standard_len${length}_block${block}_4gpu \
  --log_samples \
  --model_args model_path='/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block},use_adaptive_parallel=false,use_dynamic_threshold=false,constraints_text="200:The|201:answer|202:is" \
  &> logs/baseline-gsm8k_standard-len${length}-block${block}-4gpu.log