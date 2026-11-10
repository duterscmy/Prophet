# ICLR 2027 rebuttal experiments on Leonardo

Keep Leonardo job scripts here. Historical Surrey cluster scripts remain at the
repository root so that existing experiment paths stay intact.

## HumanEval, Math calibration

Submit from the repository root:

```bash
sbatch iclr27_rebuttal_leonardo/humaneval_math_tc_apd.sh
```

This runs all 164 HumanEval problems with `gen_length=256`, `steps=256`,
`block_length=32`, and a maximum token threshold of 0.90. The decoder loads
the Math calibration `.json` lookup table; the matching `.csv` contains audit
statistics and is not a decoder input. The reference paper result is Pass@1
38.4% and mean NFE 66.9. The job writes lm-eval results and samples under
`/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/humaneval-math-JOB_ID/`
and decoder statistics to the Slurm stdout file.

HumanEval uses the same prompt, generation filter, and `code_eval` metric as
the installed lm-eval `humaneval` task. The Leonardo task sets the dataset
configuration to `openai_humaneval`, matching the public dataset's actual
configuration name in the offline cache. Generated code is executed by the
metric; the job explicitly enables unsafe code evaluation.

For future jobs, request 30 minutes for HumanEval, 2 hours for MATH500, and
4 hours for GSM8K. The 30-minute HumanEval run uses `boost_qos_dbg`; longer
runs should use `boost_qos_bprod` with the same A100 partition and account.

## Keeping the branch current

Work on `iclr27_rebuttal_leonardo`. Before each experiment batch, check
`git status --short --branch`, then use `git fetch origin` and merge or rebase
the intended upstream changes. Avoid pulling while a job is reading scripts
from this checkout. Publish the branch with
`git push -u origin iclr27_rebuttal_leonardo` when ready; no remote branch is
created by local branch creation alone.

## Uploading calibration logs

From your local machine, use the same SSH host or alias that you use to log
in to Leonardo. The destination directory is
`/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/calibration-logs/`.
Transfer the three original full-confidence `.log` files, then compare their
sizes or checksums on both machines before calibration.

## Offline datasets

Set `HF_HOME=/leonardo_scratch/large/userexternal/mcao0000/huggingface` and
`HF_DATASETS_CACHE=${HF_HOME}/datasets` in Slurm jobs. GSM8K (`gsm8k` and
`openai/gsm8k`, config `main`), HumanEval (`openai/openai_humaneval`, config
`openai_humaneval`), and MBPP (`google-research-datasets/mbpp`, config `full`)
have been verified with `HF_DATASETS_OFFLINE=1`. The `code_eval` metric is
also cached.

MATH500 Hub paths do not reopen reliably under Datasets 2.18 offline mode.
Use `minerva_math500_leonardo` or `minerva_math500_train_leonardo` with
`--include_path iclr27_rebuttal_leonardo/tasks`; these tasks read verified
local JSONL/Parquet snapshots with the original Minerva scoring functions.
The `tc-apd` environment includes `math-verify` and ANTLR 4.11 for this
scoring path. Both tasks have been initialized offline (500 and 12,000
examples respectively).

GPQA (`Idavidrein/gpqa`, config `gpqa_main`) requires Hugging Face account
access. It has not been cached on this server.
