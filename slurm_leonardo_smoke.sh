#!/bin/bash
#SBATCH --job-name=tc_apd_smoke
#SBATCH --account=EUHPC_D36_236
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/slurm-%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/slurm-%j.err

set -euo pipefail

PROJECT_ROOT=/leonardo/home/userexternal/mcao0000/prophet
CONDA_ROOT=/leonardo/home/userexternal/mcao0000/miniconda3
MODEL_PATH=/leonardo_scratch/large/userexternal/mcao0000/models/LLaDA-8B-Instruct
HF_HOME=/leonardo_scratch/large/userexternal/mcao0000/huggingface
RUN_ROOT=/leonardo_scratch/large/userexternal/mcao0000/prophet-runs
SMOKE_DECODER="${SMOKE_DECODER:-standard}"
THRESHOLD_PATH="${PROJECT_ROOT}/token_threshold_on_trainset/math_train_token_threshold_grid_p99.5_mincount200_minaccepted100.json"
RUN_DIR="${RUN_ROOT}/smoke-${SMOKE_DECODER}-${SLURM_JOB_ID}"

case "${SMOKE_DECODER}" in
  standard)
    MODEL_ARGS="model_path=${MODEL_PATH},gen_length=32,steps=32,block_length=32,decoder=standard,print_detail_log=false"
    ;;
  tc_apd)
    MODEL_ARGS="model_path=${MODEL_PATH},gen_length=32,steps=32,block_length=32,decoder=tc_apd,threshold_path=${THRESHOLD_PATH},default_threshold=0.90,max_threshold=0.90,min_threshold=0.05,print_detail_log=false"
    ;;
  *)
    echo "Unsupported SMOKE_DECODER=${SMOKE_DECODER}" >&2
    exit 2
    ;;
esac

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate tc-apd

export HF_HOME
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${RUN_DIR}"
cd "${PROJECT_ROOT}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

python eval_llada.auto_thresh.py \
  --tasks gsm8k_cot_zeroshot \
  --model llada_dist \
  --batch_size 1 \
  --limit 1 \
  --output_path "${RUN_DIR}/results" \
  --log_samples \
  --model_args "${MODEL_ARGS}"

