# TC-APD

This repository contains the implementation and evaluation code for
**Token-Level Calibrated Adaptive Parallel Decoding (TC-APD)** from
*Beyond Global Thresholds: Token-Level Calibrated Adaptive Parallel Decoding
for Diffusion Language Models*.

- [ICLR 2027 OpenReview forum, reviews, and public discussion](https://openreview.net/forum?id=bm5Bvszv36)
- [OpenReview manuscript PDF](https://openreview.net/pdf?id=bm5Bvszv36)
- [Local ICLR submission manuscript](39088_Beyond_Global_Thresholds.pdf)
- Main backbone: LLaDA-8B-Instruct
- Main evaluation tasks: GSM8K, MATH500, and HumanEval

The local PDF is the manuscript submitted in this round. The OpenReview forum
is the source for reviewer comments and author-reviewer discussion. When these
artifacts differ, use the PDF for the submitted method and reported tables,
and use the forum to understand review concerns or requested follow-up work.

The repository reuses Prophet and lm-evaluation-harness infrastructure, but
the primary methods evaluated here are standard decoding, fixed-threshold
adaptive parallel decoding (APD), and TC-APD.

## End-to-end workflow

TC-APD has an offline calibration stage and an inference stage:

```text
calibration data
      |
      v
standard block-wise decoding with print_detail_log=true
      |
      v
log every masked position's intermediate top-1 token and confidence
      |
      v
compare each intermediate token with the final standard-decoding token
      |
      v
estimate one reliable confidence threshold per token id
      |
      v
token_id -> threshold lookup JSON
      |
      v
TC-APD inference and lm-eval evaluation
```

The lookup table is created offline. TC-APD inference only performs a token-id
lookup and does not add a model forward pass, train a predictor, or optimize
parameters online.

## Algorithm

### 1. Collect standard-decoding traces

At every denoising step, standard block-wise decoding records the current
top-1 token and confidence for **every still-masked position**, not only the
position selected at that step. A trace payload contains:

```text
selected_records    tokens eventually committed by standard decoding
all_token_records   all intermediate top-1 candidates and confidences
stats               NFE, TPF, elapsed time, and decoder information
```

For an intermediate prediction at generation position `i`, define

```text
final_consistent = (intermediate_token_i == final_standard_token_i).
```

This turns the full-confidence trace into observations
`(token_id, confidence, final_consistent)`.

### 2. Build the token-level lookup table

For token `v` and candidate threshold `tau`, let:

- `N_v` be the total number of observations for `v`;
- `A_v(tau)` be the number with confidence at least `tau`;
- `R_v(tau)` be their empirical final-consistency rate.

The table stores the lowest grid threshold satisfying

```text
N_v >= N_min
A_v(tau) >= A_min
R_v(tau) >= rho.
```

The paper configuration used by the lookup-generation commands is:

| Parameter | Value |
|---|---:|
| `rho` | 0.995 |
| `N_min` | 200 |
| `A_min` | 100 |
| Grid range | 1.000 to 0.010 |
| Grid step | 0.005 |
| Generation length | 256 |
| Block length | 32 |

Tokens that are too rare or have no feasible threshold are omitted from the
JSON and use `default_threshold` during inference. The accompanying CSV keeps
all eligible-token statistics for auditing.

### 3. TC-APD inference

At one inference step, the model predicts token `v_i` with confidence `c_i`
for each masked position `i`. TC-APD retrieves

```text
tau_i = lookup_table[v_i]
```

and commits every position satisfying `c_i >= tau_i` in parallel. If no
position qualifies, it commits the single highest-confidence position so that
decoding always progresses. Loaded thresholds are clipped to
`[min_threshold, max_threshold]`.

## Calibration sources used in this project

The paper experiments use these three full-confidence standard logs. Each
source contributes 5,000 calibration examples to the mixed-domain experiment:

| Source | Standard log | Output prefix |
|---|---|---|
| Math | `logs/math500_train_standard-len256-block32.log` | `math_train` |
| GSM8K | `logs/baseline-gsm8k_train_standard-len256-block32.log` | `gsm8k_train` |
| Code | `logs/acecode_standard-len256-block32.log` | `ace_train` |

There are three distinct calibration strategies in this repository:

1. **Source-specific calibration:** calibrate each log separately. This is
   used for cross-domain comparisons such as `from_math`, `from_gsm8k`, and
   `from_ace`.
2. **Paper mixed-domain calibration (`global_v2`):** pass all three raw logs to
   `calibrate_tc_apd.py` in one command. This combines 5,000 GSM8K, 5,000
   Minerva-Math, and 5,000 ACE-Code examples at the observation level, then
   estimates one threshold table using the same empirical consistency
   procedure as the main paper.
3. **Conservative maximum merge (`global_v3`):** calibrate each source
   separately, then use `merge_threshold.py`. For a token appearing in
   multiple tables, the merged threshold is their maximum. This is an extra
   repository variant and is not the mixed-domain strategy in the paper.

`global_v2` and `global_v3` are not equivalent: v2 pools raw observations
before calibration, whereas v3 combines already-calibrated tables. For the
mixed-domain results in Appendix A.2 and Table A2, use `global_v2`.

## Repository map

```text
tc_apd/
  calibration.py       log parsing, final-consistency labels, grid calibration
  decoder.py           standard, fixed APD, and TC-APD decoding
  thresholds.py        lookup-table loading, fallback, and clipping

calibrate_tc_apd.py     command-line lookup-table builder (JSON + audit CSV)
merge_threshold.py     maximum merge of source-specific JSON tables
eval_llada.auto_thresh.py
                        lm-eval adapter for standard, fixed APD, and TC-APD
generate.py             legacy/specialized decoders, including SOAR variants
analysis_confidence.ipynb
                        visualization only; lookup logic is no longer defined here
token_threshold_on_trainset/
                        source-specific and global lookup tables
eval*.sh                task/method-specific Slurm evaluation scripts
loop*.sh                threshold and operating-point sweeps
avg_nfe.sh              aggregate model forward calls from a log
avg_tpf.sh              aggregate tokens per forward from a log
```

`tc_apd/calibration.py` is independent of PyTorch. Lookup tables can therefore
be generated on a CPU-only machine after the standard traces have been
collected.

## Installation

A typical environment is:

```bash
conda create -n tc-apd python=3.10 -y
conda activate tc-apd
pip install torch transformers accelerate datasets numpy pandas tqdm
pip install lm-eval
```

HumanEval, MBPP, and ACE code execution also require the corresponding task
dependencies and explicit unsafe-code permission.

Set the project and model locations before running the Slurm scripts:

```bash
export TC_APD_ROOT=/path/to/Prophet
export MODEL_PATH=/path/to/LLaDA-8B-Instruct
cd "$TC_APD_ROOT"
```

Every evaluation script resolves the repository from its own location when
`TC_APD_ROOT` is not set. `MODEL_PATH` overrides the historical cluster path.

## Reproduce the lookup tables

### Step 1: collect the three standard logs

The required scripts already set `decoder=standard` and
`print_detail_log=true`:

```bash
sbatch eval_math500.baseline.train.sh
sbatch eval_gsm8k.baseline.train.sh
sbatch eval_ace.train.sh
```

They produce the three files listed above. Existing logs can be reused; the
calibration code streams them line by line and does not load the raw files all
at once.

### Step 2: calibrate one table per source

Math:

```bash
python calibrate_tc_apd.py \
  logs/math500_train_standard-len256-block32.log \
  --output token_threshold_on_trainset/math_train_token_threshold_grid_p99.5_mincount200_minaccepted100.json \
  --rho 0.995 \
  --min-token-observations 200 \
  --min-accepted-observations 100 \
  --threshold-start 1.0 \
  --threshold-end 0.01 \
  --grid-step 0.005
```

GSM8K:

```bash
python calibrate_tc_apd.py \
  logs/baseline-gsm8k_train_standard-len256-block32.log \
  --output token_threshold_on_trainset/gsm8k_train_token_threshold_grid_p99.5_mincount200_minaccepted100.json \
  --rho 0.995 \
  --min-token-observations 200 \
  --min-accepted-observations 100
```

ACE/code:

```bash
python calibrate_tc_apd.py \
  logs/acecode_standard-len256-block32.log \
  --output token_threshold_on_trainset/ace_train_token_threshold_grid_p99.5_mincount200_minaccepted100.json \
  --rho 0.995 \
  --min-token-observations 200 \
  --min-accepted-observations 100
```

Each command writes two files with the same stem:

- `.json`: the flat `token_id -> threshold` inference table;
- `.csv`: threshold, accepted coverage, successes, precision, status, and
  total count for every eligible token.

The JSON format remains compatible with the existing evaluation scripts:

```json
{
  "13": 0.815,
  "42": 0.76
}
```

Use `--max-traces-per-log N` only for quick smoke tests. Omit it for paper
tables.

### Step 3: build the paper mixed-domain table (`global_v2`)

This is the mixed-domain strategy described in Appendix A.2. All observations
from the three 5,000-example sources are pooled before threshold estimation:

```bash
python calibrate_tc_apd.py \
  logs/baseline-gsm8k_train_standard-len256-block32.log \
  logs/math500_train_standard-len256-block32.log \
  logs/acecode_standard-len256-block32.log \
  --output token_threshold_on_trainset/global_v2_token_threshold_grid_p99.5_mincount200_minaccepted100.json \
  --rho 0.995 \
  --min-token-observations 200 \
  --min-accepted-observations 100 \
  --threshold-start 1.0 \
  --threshold-end 0.01 \
  --grid-step 0.005
```

The existing 99.5% `global_v2` JSON contains 7,397 token thresholds.

### Step 4: optionally build the maximum-merge variant (`global_v3`)

`global_v3` is a separate conservative experiment. It takes the maximum of
the three independently calibrated source-specific thresholds:

```bash
python merge_threshold.py \
  token_threshold_on_trainset/math_train_token_threshold_grid_p99.5_mincount200_minaccepted100.json \
  token_threshold_on_trainset/gsm8k_train_token_threshold_grid_p99.5_mincount200_minaccepted100.json \
  token_threshold_on_trainset/ace_train_token_threshold_grid_p99.5_mincount200_minaccepted100.json \
  token_threshold_on_trainset/global_v3_token_threshold_grid_p99.5_mincount200_minaccepted100.json \
  --sort_keys
```

The existing 99.5% `global_v3` table has 6,860 token thresholds and exactly
matches this maximum-merge rule. Do not use v3 to reproduce the paper's
mixed-domain Table A2.

## Evaluation

### Primary comparison

| Method | Decoder argument | GSM8K | MATH500 | HumanEval |
|---|---|---|---|---|
| Standard | `decoder=standard` | `eval_gsm8k.baseline.sh` | `eval_math500.baseline.sh` | `eval_human_eval.baseline.sh` |
| Fixed APD | `decoder=fixed_apd` | `eval_gsm8k_adaptive_parallel.sh` | `eval_math500_adaptive_parallel.sh` | `eval_human_eval_adaptive_parallel.sh` |
| TC-APD main paper, Minerva-Math | `decoder=tc_apd` | `eval_gsm8k_dynamic_threshold.from_math.sh` | `eval_math500_dynamic_threshold.from_math.sh` | `eval_human_eval_dynamic.from_math.sh` |

Run, for example:

```bash
sbatch eval_gsm8k.baseline.sh
sbatch eval_gsm8k_adaptive_parallel.sh
sbatch eval_gsm8k_dynamic_threshold.from_math.sh
```

The historical filenames use `dynamic_threshold` for TC-APD. This is distinct
from the Dynamic-DLLM baseline, whose scripts contain `dynamic_dllm`.

All scripts whose filenames end in `*.global.sh` reference `global_v2`, so
they evaluate the paper's pooled mixed-domain table. Other explicitly named
v3 experiments may still use the maximum-merge table; check the threshold
filename before launching a job.

### Source-transfer evaluation

The source-specific lookup scripts measure calibration transfer:

| Evaluation task | Math table | GSM8K table | ACE table |
|---|---|---|---|
| GSM8K | `eval_gsm8k_dynamic_threshold.from_math.sh` | `eval_gsm8k_dynamic_threshold.from_gsm8k.sh` | `eval_gsm8k_dynamic_threshold.from_ace.sh` |
| MATH500 | `eval_math500_dynamic_threshold.from_math.sh` | `eval_math500_dynamic_threshold.from_gsm8k.sh` | `eval_math500_dynamic_threshold.from_ace.sh` |
| HumanEval | `eval_human_eval_dynamic.from_math.sh` | `eval_human_eval_dynamic.from_gsm8k.sh` | `eval_human_eval_dynamic.from_ace.sh` |

### Direct lm-eval examples

Standard decoding:

```bash
accelerate launch --num_processes 1 eval_llada.auto_thresh.py \
  --tasks gsm8k_cot_zeroshot \
  --model llada_dist \
  --num_fewshot 0 \
  --output_path evals_results/standard/gsm8k \
  --log_samples \
  --model_args model_path="$MODEL_PATH",decoder=standard,gen_length=256,steps=256,block_length=32
```

Fixed APD at 0.90:

```bash
accelerate launch --num_processes 1 eval_llada.auto_thresh.py \
  --tasks gsm8k_cot_zeroshot \
  --model llada_dist \
  --num_fewshot 0 \
  --output_path evals_results/fixed_apd/gsm8k-t090 \
  --log_samples \
  --model_args model_path="$MODEL_PATH",decoder=fixed_apd,confidence_threshold=0.90,gen_length=256,steps=256,block_length=32
```

TC-APD with the paper mixed-domain table:

```bash
TABLE=token_threshold_on_trainset/global_v2_token_threshold_grid_p99.5_mincount200_minaccepted100.json

accelerate launch --num_processes 1 eval_llada.auto_thresh.py \
  --tasks gsm8k_cot_zeroshot \
  --model llada_dist \
  --num_fewshot 0 \
  --output_path evals_results/tc_apd/gsm8k-t090 \
  --log_samples \
  --model_args model_path="$MODEL_PATH",decoder=tc_apd,threshold_path="$TABLE",default_threshold=0.90,max_threshold=0.90,min_threshold=0.05,gen_length=256,steps=256,block_length=32
```

Set `print_detail_log=false` for normal evaluation to avoid writing the large
candidate trace. Keep it `true` only when collecting calibration logs.

### Other baselines

The repository retains the comparison implementations and scripts:

| Method | Representative entry point/script |
|---|---|
| Prophet | `eval_llada.py`, `eval_gsm8k.prophet.sh` |
| WINO | `eval_llada.wino.py`, `eval_gsm8k.wino.sh` |
| Dynamic-DLLM | `eval_gsm8k.dynamic_dllm.sh` |
| OSDT | `eval_gsm8k.osdt.sh` |
| SOAR | `eval_gsm8k.soar.sh` |
| Fast-dLLM | `eval_llada.fastdllm.py`, `eval_gsm8k.fastdllm.sh` |
| Learn2PD | `l2pd/eval_gsm8k.l2pd.sh` |
| TC-APD + SOAR | `eval_gsm8k_dynamic_threshold_soar.from_math.sh` |

Equivalent scripts are provided for MATH500, HumanEval, MBPP, GPQA, BBH, and
WMT where the corresponding baseline supports the task.

### Metrics and aggregation

Keep the checkpoint, prompts, task normalization, generation length, and block
length fixed across methods. Report task accuracy or Pass@1 together with:

- NFE: model forward calls;
- speedup relative to 256-step standard decoding;
- TPF: committed tokens per forward;
- wall-clock time or throughput.

Aggregate NFE and TPF from an evaluation log with:

```bash
bash avg_nfe.sh logs/<evaluation>.log
bash avg_tpf.sh logs/<evaluation>.log
```

## Main paper results

The main LLaDA results reported in the paper are:

| Method | GSM8K Acc / NFE | MATH500 Acc / NFE | HumanEval Acc / NFE | Average Acc / NFE |
|---|---:|---:|---:|---:|
| Standard | 72.9 / 256.0 | 32.4 / 256.0 | 38.4 / 256.0 | 47.9 / 256.0 |
| Fixed APD, 0.90 | 72.9 / 77.5 | 32.4 / 106.0 | 37.8 / 100.7 | 47.7 / 94.7 |
| TC-APD, 0.90 | 72.1 / 56.4 | 33.4 / 82.5 | 38.4 / 66.9 | 48.0 / 68.6 |

At the 0.90 operating point, TC-APD reduces average NFE from 256.0 to 68.6
while preserving average accuracy.

These main LLaDA TC-APD numbers use the **single-domain Minerva-Math** lookup
table. They are not mixed-domain results.

### Mixed-domain result from Appendix A.2

The paper separately compares the 5,000-example Minerva-Math table with the
15,000-example mixed-domain table. At maximum threshold 0.90:

| Calibration | GSM8K Acc / NFE | MATH500 Acc / NFE | HumanEval Acc / NFE | Average Acc / NFE / Speedup |
|---|---:|---:|---:|---:|
| Minerva-Math, 5,000 examples | 72.1 / 56.4 | 33.4 / 82.5 | 38.4 / 66.9 | 48.0 / 68.6 / 3.73x |
| Mixed-domain `global_v2`, 15,000 examples | 72.3 / 60.4 | 33.4 / 85.9 | 38.4 / 68.8 | 48.0 / 71.7 / 3.57x |

For LLaDA, mixed-domain calibration does not outperform Minerva-Math at this
operating point. The paper therefore reports Minerva-Math calibration for its
main LLaDA results. The appendix also notes that the preferred calibration
source can depend on the backbone.

## Collaboration notes for future agents

Use the following source-of-truth order when continuing this project:

1. `39088_Beyond_Global_Thresholds.pdf` is the current ICLR submission and the
   authority for the algorithm, experiment definitions, and reported numbers.
2. The [OpenReview forum](https://openreview.net/forum?id=bm5Bvszv36) contains
   the official reviewer comments and public discussion for this submission.
   Treat reviewer requests as review context, not as already-implemented
   algorithm requirements.
3. `README.md` maps the paper concepts to this repository. In particular,
   preserve the distinction `global_v2 = pooled mixed-domain calibration` and
   `global_v3 = maximum merge of source-specific tables`.
4. `analysis_confidence.ipynb` is for visualization. Production lookup-table
   generation belongs in `tc_apd/calibration.py` and `calibrate_tc_apd.py`.

Before changing evaluation scripts, check the actual threshold filename in
the script rather than inferring the strategy from `.global.sh` alone.

## Backward compatibility

`eval_llada.auto_thresh.py` accepts both the explicit interface and historical
arguments:

- `use_adaptive_parallel=true` maps to `decoder=fixed_apd`;
- `use_dynamic_threshold=true` maps to `decoder=tc_apd`;
- `dynamic_threshold_json=...` is an alias for `threshold_path=...`.

New experiments should use `decoder=standard`, `decoder=fixed_apd`, or
`decoder=tc_apd` explicitly.

## Acknowledgments

This project reuses infrastructure and evaluation integrations from Prophet,
LLaDA, lm-evaluation-harness, and the compared diffusion-language-model
decoding methods.
