# Running TC-APD on Leonardo

## Installed environment

- Miniconda: `/leonardo/home/userexternal/mcao0000/miniconda3`
- Conda environment: `tc-apd`
- Model: `/leonardo_scratch/large/userexternal/mcao0000/models/LLaDA-8B-Instruct`
- Hugging Face cache: `/leonardo_scratch/large/userexternal/mcao0000/huggingface`
- Run outputs: `/leonardo_scratch/large/userexternal/mcao0000/prophet-runs`

Activate the environment with:

```bash
source /leonardo/home/userexternal/mcao0000/miniconda3/etc/profile.d/conda.sh
conda activate tc-apd
```

The tested core versions are Python 3.10, PyTorch 2.2.2+cu121,
Transformers 4.38.2, Accelerate 0.27.2, Datasets 2.18.0, lm-eval 0.4.8,
and PEFT 0.10.0.

## Slurm settings

Leonardo A100 jobs use:

```text
account:   EUHPC_D36_236
partition: boost_usr_prod
GPU:       --gres=gpu:1
```

Use `boost_qos_dbg` for smoke tests of at most 30 minutes. Use
`boost_qos_bprod` for normal runs up to 24 hours or `boost_qos_lprod` for
longer runs up to four days.

The historical evaluation scripts still contain `--partition=a100` and
`conda activate ttrl_env`, which belong to the previous cluster. Update those
two settings before using a script for a full Leonardo experiment, or use a
Leonardo-specific wrapper.

## Smoke test

Validate the resource request without submitting:

```bash
sbatch --test-only slurm_leonardo_smoke.sh
```

Submit the one-example GSM8K inference and evaluation:

```bash
sbatch slurm_leonardo_smoke.sh
```

Run the same smoke test through TC-APD with the Minerva-Math threshold table:

```bash
sbatch --export=ALL,SMOKE_DECODER=tc_apd slurm_leonardo_smoke.sh
```

Monitor and inspect it with:

```bash
squeue -u mcao0000
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,AllocTRES
tail -f /leonardo_scratch/large/userexternal/mcao0000/prophet-runs/slurm-JOB_ID.out
tail -f /leonardo_scratch/large/userexternal/mcao0000/prophet-runs/slurm-JOB_ID.err
```

The smoke test uses one GSM8K example and a 32-token generation. It verifies
scheduling, CUDA, model loading, generation, lm-eval post-processing, and
result serialization; it is not a paper-quality accuracy run.

## Current gaps before full experiments

- The three full-confidence calibration logs are not present locally.
- The installed lm-eval 0.4.8 includes `gsm8k_cot_zeroshot` and `humaneval`.
  The repository's `lm-eval-tasks/math500.yaml` defines `minerva_math500`;
  pass `--include_path lm-eval-tasks` when running MATH500 evaluation.
- The historical task scripts need Leonardo partition, QoS, environment, and
  model-path updates before full submissions.

