#!/bin/bash

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?Run this script as a Slurm array}"

VARIANTS=(fine_tuned quantized_int8)
METHODS=(standard fixed_apd tc_apd)
VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
export MODEL_VARIANT="${VARIANTS[VARIANT_INDEX]}"
export METHOD="${METHODS[METHOD_INDEX]}"

exec /bin/bash /leonardo/home/userexternal/mcao0000/prophet/iclr27_rebuttal_leonardo/eval_p22_model_variant.sh
