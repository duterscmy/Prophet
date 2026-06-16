#!/bin/bash
#SBATCH --job-name="prophet_gsm8k"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate soar

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0
#  --limit 256
length=256

# constraints_text="200:The|201:answer|202:is"

accelerate launch eval_llada.py \
  --tasks gsm8k_cot_zeroshot \
  --model llada_dist \
  --num_fewshot 0 \
  --log_samples \
  --output_path evals_results/prophet/gsm8k-ns0-${length} \
  --model_args model_path='/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct',enable_early_exit=true,enable_soar=false,gen_length=256,steps=256,block_length=32,answer_length=5 &> logs/prophet-gsm8k_cot_zeroshot-ns0-${length}.log