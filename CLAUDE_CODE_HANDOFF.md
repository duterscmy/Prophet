# Claude Code 接管说明：TC-APD ICLR 2027 Rebuttal

> 更新时间：2026-11-10（Europe/Brussels）  
> 项目目录：`/leonardo/home/userexternal/mcao0000/prophet`  
> 当前分支：`iclr27_rebuttal_leonardo`

## 1. 你现在接管的任务

这是 ICLR 2027 投稿论文 **Beyond Global Thresholds: Token-Level
Calibrated Adaptive Parallel Decoding for Diffusion Language Models** 的代码、
补充实验和 rebuttal 工作区。方法简称 **TC-APD**，主要 backbone 是
`LLaDA-8B-Instruct`。当前目标不是重新开发方法，而是：

1. 完成 reviewer 要求的 rebuttal 实验；
2. 核验并汇总所有新增结果；
3. 撰写逐 reviewer 回复；
4. 将最重要的新结果同步修改进论文正文和 appendix；
5. 保持所有统计口径、实验配置和结果路径可复现。

论文与 review：

- OpenReview forum（官方 reviews 和公开讨论）：
  <https://openreview.net/forum?id=bm5Bvszv36>
- OpenReview PDF：<https://openreview.net/pdf?id=bm5Bvszv36>
- 本地投稿 PDF：`/leonardo/home/userexternal/mcao0000/prophet/39088_Beyond_Global_Thresholds.pdf`
- 本地 review/rebuttal 总计划：
  `/leonardo/home/userexternal/mcao0000/prophet/iclr2027_review_rebuttal.md`
- 项目方法与复现说明：
  `/leonardo/home/userexternal/mcao0000/prophet/README.md`
- Leonardo 环境说明：
  `/leonardo/home/userexternal/mcao0000/prophet/LEONARDO.md`

三位 reviewer：

- `jgET`：rating 2，主要质疑 offline calibration 成本、domain/calibration
  sensitivity、accelerated trajectory distribution shift；
- `iRo1`：rating 2，明确表示如果补充 calibration-size 和 out-of-domain/chat
  结果，可能提高评分；
- `4QWr`：rating 3，主要质疑 source-selection bias、held-out statistical
  validity、confidence bounds、matched-compute fairness、配置/模型 transfer。

`iclr2027_review_rebuttal.md` 第 8 节是之前加入的 **“GPT_plus 的建议”**，
包含正文修改优先级、appendix 结构、claim 弱化和 revision 上传策略。注意：
该文件早期的 `[x]/[ ]` 状态没有跟所有后续实验同步，实际完成状态应以本文、
`iclr27_rebuttal_leonardo/analysis/` 和 scratch 结果为准。

## 2. 工作区和安全约束

当前工作树有用户及 rebuttal 未提交修改，不要 reset、checkout 覆盖或清理：

- modified：`eval_llada.auto_thresh.py`
- modified：`iclr2027_review_rebuttal.md`
- untracked：`iclr27_rebuttal_leonardo/`、`LEONARDO.md`、Leonardo task files 等

先运行：

```bash
cd /leonardo/home/userexternal/mcao0000/prophet
git status --short --branch
```

不要覆盖仓库根目录旧 Surrey cluster 脚本；Leonardo/rebuttal 新脚本都应放在：

```text
/leonardo/home/userexternal/mcao0000/prophet/iclr27_rebuttal_leonardo/
```

推理详细日志按用户要求写到 run root 下的 `logs/`，不要依赖 Slurm stdout。
`results/` 中的 `results_*.json` 是 lm-eval aggregate，`samples_*.jsonl` 是逐样本
结果；NFE 和 s/it 通常从对应 inference log 的 decoder statistics 汇总。

## 3. Leonardo 环境

```bash
source /leonardo/home/userexternal/mcao0000/miniconda3/etc/profile.d/conda.sh
conda activate tc-apd
export HF_HOME=/leonardo_scratch/large/userexternal/mcao0000/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
```

关键路径：

```text
Base model:
/leonardo_scratch/large/userexternal/mcao0000/models/LLaDA-8B-Instruct

Fine-tuned model:
/leonardo_scratch/large/userexternal/mcao0000/models/LLaDA-Instruct-JustGRPO-GSM8K

INT8 GPTQ model:
/leonardo_scratch/large/userexternal/mcao0000/models/llada-8b-instruct-8bit-gptq

INT8 environment:
/leonardo_scratch/large/userexternal/mcao0000/envs/tc-apd-gptq/bin/python

All run outputs:
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs

Uploaded calibration logs:
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/calibration-logs/complete
```

Slurm：

```text
account: EUHPC_D36_236
GPU partition: boost_usr_prod
GPU: --gres=gpu:1
normal/long QoS used: boost_qos_lprod
CPU analysis/calibration partition: lrd_all_serial
```

主评估默认配置：`gen_length=256, steps=256, block_length=32,
batch_size=1, max/default threshold=0.90, min threshold=0.01`。

## 4. 方法与 calibration 口径

TC-APD 从 Standard decoding 的完整中间 top-1 candidate trace 中构造
`(token_id, confidence, final_consistent)`，然后为每个 token 选择满足条件的最低
threshold。论文设置：

```text
rho = 0.995
minimum total count = 200
minimum accepted count = 100
threshold range = 0.01 ... 1.00
grid step = 0.005
candidate threshold count = 199
unsupported-token default threshold = 0.90
```

经验 `rho=0.995` 是 nominal empirical target，不是 formal guarantee。
经验 `R_v(tau)` 没有做 isotonic regression 或 monotonic envelope，有限样本下不保证
单调。运行时阈值会按 min/max threshold clip。

主 Math-5000 threshold：

```text
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-size-ablation/thresholds/math_n5000_seed2027_token_threshold_grid_p99.5_mincount200_minaccepted100.json
```

## 5. 已完成的核心结果与位置

### 5.1 P0.1 Calibration-size ablation（seed 2027）

状态：完成。Math calibration size 为 100/200/500/1000/2000/5000，评估
GSM8K、MATH500、HumanEval；包含 Standard 和 Fixed 0.90。

主报告：

```text
iclr27_rebuttal_leonardo/analysis/calibration_size/P01_CALIBRATION_SIZE_SUMMARY.md
iclr27_rebuttal_leonardo/analysis/calibration_size/calibration_size_full_comparison.csv
iclr27_rebuttal_leonardo/analysis/calibration_size/calibration_size_macro_summary.csv
iclr27_rebuttal_leonardo/analysis/calibration_size/evaluation_metrics_long.csv
iclr27_rebuttal_leonardo/analysis/calibration_size/figure_p01_calibration_size_tradeoff.{png,pdf}
iclr27_rebuttal_leonardo/analysis/calibration_size/figure_p01_threshold_support.{png,pdf}
```

原始结果：

```text
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-size-ablation/{thresholds,logs,results,scheduler}
```

关键表（Acc/NFE/s）：

| Setting | GSM8K | MATH500 | HumanEval | Macro |
|---|---:|---:|---:|---:|
| Standard | 73.16/256.00/10.158 | 31.40/256.00/10.469 | 37.20/256.00/11.454 | 47.25/256.00/10.694 |
| Fixed 0.90 | 73.09/77.62/3.112 | 31.80/106.27/4.393 | 37.20/98.67/4.450 | 47.36/94.19/3.985 |
| Math-100 | 73.31/59.71/2.404 | 32.40/82.38/3.421 | 38.41/72.87/3.305 | 48.04/71.65/3.043 |
| Math-200 | 73.39/61.79/2.495 | 31.60/85.80/3.570 | 36.59/72.35/3.302 | 47.19/73.31/3.122 |
| Math-500 | 74.07/60.77/2.438 | 32.20/85.16/3.525 | 37.20/67.23/3.055 | 47.82/71.05/3.006 |
| Math-1000 | 72.71/58.80/2.377 | 31.60/83.83/3.489 | 38.41/68.51/3.118 | 47.57/70.38/2.995 |
| Math-2000 | 72.86/58.92/2.380 | 31.60/83.56/3.492 | 35.98/68.62/3.105 | 46.81/70.37/2.992 |
| Math-5000 | 73.24/57.17/2.308 | 32.20/83.10/3.445 | 37.80/67.13/3.067 | 47.75/69.14/2.940 |

主结论：100 条已恢复 5000 条宏平均 NFE reduction 的 98.7%；accuracy 不随
calibration size 单调变化。不要把 final-output re-tokenization coverage 误称为精确
candidate-trace fallback。

### 5.2 500 vs 5000 高频 token coverage

状态：完成。

```text
iclr27_rebuttal_leonardo/analysis/n500_high_frequency_coverage/N500_HIGH_FREQUENCY_COVERAGE.md
iclr27_rebuttal_leonardo/analysis/n500_high_frequency_coverage/summary.json
iclr27_rebuttal_leonardo/analysis/n500_high_frequency_coverage/top500_candidate_tokens.csv
```

在同一批 5000 条 Math Standard traces 的 21,120,000 个中间 candidate records 上：

- 500/5000 表有效 threshold token：410 / 1453；
- 500 对 5000 表的 token type recall：28.22%；
- 500 表 candidate mass coverage：92.93%；
- 5000 表 candidate mass coverage：98.22%；
- 500 恢复 5000-table candidate mass 的 94.61%；
- Top-200 高频 token 占全部 mass 的 88.92%，且被 500 表全部覆盖；
- 410 个共同 token 的 threshold mean absolute difference 为 0.103，故应称
  “compact high-frequency approximation”，不能称完全重建。

### 5.3 P0.4 Independent held-out consistency

状态：完成。4000 条 Math Standard traces 估计 threshold，1000 条不重叠 traces
验证。

```text
iclr27_rebuttal_leonardo/analysis/heldout_consistency/P04_HELDOUT_CONSISTENCY_SUMMARY.md
iclr27_rebuttal_leonardo/analysis/heldout_consistency/per_token_heldout_validation.csv
iclr27_rebuttal_leonardo/analysis/heldout_consistency/heldout_support_group_summary.csv
```

关键结果：raw selected threshold 在 held-out 上 realized consistency 99.443%，
95% Wilson CI `[99.431%, 99.454%]`；full lookup policy 为 99.447%。该实验只验证
unseen **Standard states**，不能自动推出 accelerated-state guarantee。

### 5.4 P1.1 Standard → TC-APD distribution shift

最终输出/答案层面分析已完成：

```text
iclr27_rebuttal_leonardo/analysis/standard_tc_apd_shift/P11_STANDARD_TC_APD_SHIFT.md
iclr27_rebuttal_leonardo/analysis/standard_tc_apd_shift/{final_answer_agreement,answer_transitions,sequence_agreement,per_sample_sequence_metrics}.csv
```

主 Math-5000 设置在 1319 个 GSM8K 样本上：

- final-answer agreement 76.27%；
- exact final sequence agreement 20.17%，即 79.83% sequence disagreement；
- correctness agreement 91.43%；
- Standard correct→TC incorrect：56；Standard incorrect→TC correct：57；
- accuracy delta：+0.08 pp；
- EOS-aware token agreement：60.54%；mean edit similarity：0.892。

解释：TC-APD 明显改变生成路径，但 aggregate correctness flips 基本平衡。不要把
sequence reproduction 当作 TC-APD 目标。

真正逐 iteration trace 的两个 GPU 任务也已完成（GSM8K limit 500）：

```text
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-p11-iteration-trajectories/logs/gsm8k_standard_limit500_len256_block32_iteration_traces.log
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-p11-iteration-trajectories/logs/gsm8k_tc_apd_limit500_len256_block32_iteration_traces.log
```

但分析 job `64002021` 因 `ModuleNotFoundError: tc_apd` 失败。修复方法是在项目根目录
设置 `PYTHONPATH` 后重跑，不需要重新做 GPU 推理：

```bash
cd /leonardo/home/userexternal/mcao0000/prophet
export PYTHONPATH=/leonardo/home/userexternal/mcao0000/prophet:${PYTHONPATH:-}
sbatch iclr27_rebuttal_leonardo/run_p11_iteration_trajectory_analysis.sh
```

提交前检查该 shell 是否会再次覆盖 `PYTHONPATH`。

### 5.5 P1.2 Calibration subset sampling sensitivity

100 条三 seed 已完整完成并汇总：

```text
iclr27_rebuttal_leonardo/analysis/threshold_seed_sensitivity/P12_THRESHOLD_SEED_SENSITIVITY.md
iclr27_rebuttal_leonardo/analysis/threshold_seed_sensitivity/downstream_seed_sensitivity.csv
```

seed 2027/2028/2029 有效 thresholds：141/127/142；union 191，三者 intersection
90（47.12%）。GSM8K NFE `59.56 ± 0.75`，MATH500 `82.16 ± 0.53`，HumanEval
`71.24 ± 3.10`。Threshold membership/value 对采样敏感，但 aggregate NFE 更稳定。

500 条三 seed threshold 和下游 GPU 结果均已完成；threshold 报告在：

```text
iclr27_rebuttal_leonardo/analysis/threshold_seed_sensitivity_n500/P12_THRESHOLD_SEED_SENSITIVITY.md
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-seed-sensitivity-n500/{thresholds,logs,results}
```

500 三 seed 有效 thresholds：410/408/439；union 550，intersection 301
（54.73%）。下游结果文件存在，但 Markdown 尚未加入 downstream mean/std，接管后
应统一汇总。

200/1000/2000/5000 的新增 seed 2028/2029 状态见第 7 节。

### 5.6 P1.3 Confidence-bound threshold

状态：Wilson calibration 和三个主任务推理均完成。

```text
iclr27_rebuttal_leonardo/analysis/confidence_bound/P13_CONFIDENCE_BOUND_CALIBRATION.md
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-confidence-bound/{logs,results,thresholds,scheduler}
```

经验规则有效 token 1453；one-sided 95% Wilson lower-bound rule 有效 token 491，
仅保留 33.79%，且 491 个共同 token 全部得到相等或更高 threshold。无有效 Wilson
threshold 的 token 仍 fallback 0.90，因此这不是 vocabulary-wide formal guarantee。

### 5.7 P1.4 Matched NFE / wall-clock

状态：完成。Fixed APD threshold 0.77 vs Math-5000 TC-APD 0.90，20,000 次 paired
bootstrap。

```text
iclr27_rebuttal_leonardo/analysis/matched_compute/P14_MATCHED_COMPUTE_SUMMARY.md
iclr27_rebuttal_leonardo/analysis/matched_compute/paired_bootstrap_summary.csv
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-matched-compute
```

| Task | Fixed acc/NFE/s | TC acc/NFE/s | Paired accuracy delta (95% CI) |
|---|---:|---:|---:|
| GSM8K | 72.33/56.54/2.295 | 73.24/57.17/2.308 | +0.91 pp [-0.68,+2.50] |
| MATH500 | 30.80/82.27/3.438 | 32.20/83.10/3.445 | +1.40 pp [-0.60,+3.60] |
| HumanEval | 37.20/69.38/3.135 | 37.80/67.13/3.067 | +0.61 pp [-1.83,+3.05] |

所有 accuracy CI 都包含 0，不应声称统计显著优势。

### 5.8 P2.1 Offline calibration cost and amortization

状态：完成。

```text
iclr27_rebuttal_leonardo/analysis/calibration_amortization/P21_CALIBRATION_AMORTIZATION.md
iclr27_rebuttal_leonardo/analysis/calibration_amortization/amortization_by_calibration_size.csv
```

按单 A100、MATH Standard 10.469 GPU-seconds/trace 统一计算：

| Size | A100 GPU-hours | Mean break-even generations |
|---:|---:|---:|
| 100 | 0.291 | 138 |
| 200 | 0.582 | 278 |
| 500 | 1.454 | 686 |
| 1000 | 2.908 | 1368 |
| 2000 | 5.816 | 2734 |
| 5000 | 14.541 | 6788 |

阈值表 CPU 构建只花 124 秒/8 cores；主要成本是 Standard trace collection。不能称
“negligible calibration cost”，应称可摊销的 one-time cost。

## 6. OOD、长文本和 transfer 实验

### 6.1 IFEval（length 256）

状态：Standard、Fixed 0.90、Math calibration 100--5000 全部完成。

```text
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-ifeval/{logs,results,scheduler}
```

| Setting | Inst strict | Prompt strict | NFE | s/it |
|---|---:|---:|---:|---:|
| Standard | 66.79% | 57.49% | 256.00 | 10.066 |
| Fixed 0.90 | 66.79% | 57.49% | 137.46 | 5.462 |
| Math-100 | 66.19% | 57.12% | 128.08 | 5.101 |
| Math-200 | 66.91% | 57.49% | 128.66 | 5.102 |
| Math-500 | 66.55% | 57.67% | 126.74 | 5.044 |
| Math-1000 | 66.31% | 57.30% | 126.33 | 5.036 |
| Math-2000 | 66.79% | 56.93% | 126.13 | 5.000 |
| Math-5000 | 67.03% | 57.86% | 124.62 | 4.976 |

### 6.2 WMT16 EN-RO

状态：length 256 的 Standard、Fixed、Math-100--5000 已完成；length 512 按用户要求
取消。WMT 是 500 条 calibration 不完全接近 5000 的主要例外，适合作为跨语言
long-tail 边界，不要隐藏该负面结果。

```text
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-wmt16-en-ro/{logs,results,scheduler}
iclr27_rebuttal_leonardo/tasks/wmt16_en_ro_leonardo.yaml
```

### 6.3 ARC-Challenge 与 XSum-500

状态：两者的 Standard、Fixed、Math-100--5000 均完成（每个目录 8 个 aggregate
result files）。

```text
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-arc-challenge/{logs,results,scheduler}
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-xsum-500/{logs,results,scheduler}
```

XSum 只评估固定 500 条子集；本地 task：
`iclr27_rebuttal_leonardo/tasks/xsum_500_leonardo.yaml`。

### 6.4 WritingPrompts-100 长文本压力测试

状态：length 1024，Standard、Fixed、Math-100--5000 全部完成。

```text
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-writingprompts-100-len1024/{logs,results,scheduler}
iclr27_rebuttal_leonardo/tasks/writingprompts_100_leonardo.yaml
```

主结论：Math-5000 TC-APD 相对 Standard 约 1.50x NFE speedup、1.48x wall-clock
speedup；保留的退化指标是 sentence repetition 和 4-gram repetition。它是长文本
efficiency/degeneration stress test，不是完整创作质量评估，不能用 repetition 证明
human-perceived quality。

### 6.5 P2.2 decoding configuration transfer

状态：以下 setting 对 Standard/Fixed 0.90/TC-APD(Math-5000) 和三个主任务均已跑：

1. length 256, block 16；
2. length 256, block 8；
3. length 512, block 32；
4. length 256, block 32 + chat template。

```text
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/p22-transfer/{logs,results,scheduler}
iclr27_rebuttal_leonardo/eval_p22_transfer_array.sh
```

共有 36 个组合，实际 aggregate result 为 35 个：缺失的是用户明确要求取消且不重跑的
`GSM8K length512 Standard`；不要把它误报成失败或重新提交。其余结果完整。

### 6.6 Fine-tuned 与 INT8 model transfer

状态：Fine-tuned 和 INT8 的 Standard/Fixed/TC-APD 均已覆盖 GSM8K、MATH500、
HumanEval。

```text
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/p22-model-variants/{logs,results,scheduler}
iclr27_rebuttal_leonardo/eval_p22_model_variant.sh
```

Fine-tuned 使用 `LLaDA-Instruct-JustGRPO-GSM8K`，沿用 base tokenizer；完整三任务。
INT8 GPTQ 因非常慢，只跑 GSM8K 300、MATH500 300、HumanEval 164 条。每种模型均
比较 Standard、Fixed 0.90、TC-APD Math-5000。该结果只能说明旧表在这些 variant
上仍可产生加速；accuracy 变化意味着不能声称模型权重/量化变化后永远无需重新校准。

## 7. 当前正在运行：多 calibration subset seed robustness

用户最新要求：对 calibration size 100/200/500/1000/2000/5000 各用三个随机
subset seeds（2027/2028/2029），在 GSM8K、MATH500、HumanEval 上汇报 sampling
robustness。

当前批次只补 seed 2028/2029 的 200/1000/2000/5000；100 和 500 已跑过。

```text
Calibration job: 64033378  COMPLETED
HumanEval array: 64033380  COMPLETED (8/8)
MATH500 array:   64033381  COMPLETED (8/8)
GSM8K array:     64033383  RUNNING/PENDING at handoff time
```

实时检查：

```bash
squeue -j 64033383 -o '%.18i %.30j %.2t %.10M %.10l %R'
sacct -j 64033378,64033380,64033381,64033383 \
  --format=JobIDRaw,JobName%32,State,Elapsed,Timelimit,ExitCode -X
```

新结果：

```text
/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/rebuttal-seed-sensitivity-multisize/{thresholds,logs,results,scheduler}
```

array mapping：

```text
index 0: size 200,  seed 2028
index 1: size 200,  seed 2029
index 2: size 1000, seed 2028
index 3: size 1000, seed 2029
index 4: size 2000, seed 2028
index 5: size 2000, seed 2029
index 6: size 5000, seed 2028
index 7: size 5000, seed 2029
```

全部完成后，用户要求一个完整表：

- 列顺序：GSM8K → MATH500 → HumanEval → Avg；
- 每个任务列：Acc、NFE、s/it、SpeedUp；
- Standard 和 Fixed 0.90 各一行；
- 每个 calibration size 四行：seed 2027、2028、2029、Mean ± Std；
- 这里 reviewer 要的是 `Mean ± Std`（sample standard deviation, `ddof=1`），不要把
  ± 后面的量误算成 variance；
- Avg 是三个任务的 macro average，SpeedUp 的定义必须沿用现有表：每任务用对应
  Standard 除以当前 setting，再对三个任务取平均或明确报告 macro ratio；不要混用。

还应同步生成每个 size 的 threshold stability：有效 token 数、union/intersection、
Jaccard、共同 token threshold mean absolute difference、within-0.05 agreement。
可复用 `iclr27_rebuttal_leonardo/analyze_threshold_seed_sensitivity.py`，但目前脚本的
输入目录布局要与 multisize run root 对齐。

## 8. 接管后的最高优先级 TODO

1. **监控并完成 job 64033383**，检查所有 8 个 GSM8K result/log；若单个 task
   TIMEOUT/FAILED，只重跑对应 array index。
2. **汇总 100--5000 三 seed 表**，按第 7 节格式生成 CSV + Markdown；补齐 500
   report 的 downstream mean/std。
3. **重跑逐 iteration 的 CPU analysis**，只修 `PYTHONPATH`，禁止重复 GPU inference；
   将结果写入新的 `analysis/iteration_trajectory/` 或脚本约定目录。
4. **将 OOD、P2.2、model-variant 的 raw results 做成正式 CSV/Markdown 汇总**。
   Raw results 已完成，但不少表目前只存在于对话或 scratch JSON/log，仓库里缺统一
   report。
5. **更新 `iclr2027_review_rebuttal.md` 的 checkbox/状态**，但保留第 8 节
   GPT_plus 建议；不要根据旧 checkbox 重跑已完成实验。
6. **撰写 rebuttal 与 paper revision**。正文优先放 calibration-size、IFEval/OOD、
   held-out consistency、matched compute；详细 seed sensitivity、Wilson、trajectory、
   configuration/model transfer 和 WritingPrompts 放 appendix。
7. **P0.3 source-selection protocol 是写作/诚实披露问题**：如果原 Math source 是观察
   benchmark 后选的，必须承认 exploratory/test-informed，不能称 pre-registered。

## 9. Claim 边界（非常重要）

不要写：

- `guarantees 99.5% consistency`；
- `preserves Standard trajectories/correctness`；
- `500 calibration examples are always sufficient`；
- `generalizes universally across domains/models/configurations`；
- `calibration cost is negligible`；
- `WritingPrompts repetition proves creative-writing quality`；
- `quantization/fine-tuning never requires recalibration`。

建议写：

- `empirically targets 99.5% consistency`；
- `achieves 99.443% held-out consistency on independent Standard traces`；
- `preserves aggregate task accuracy in the evaluated settings`；
- `500 traces provide a compact high-frequency approximation and are close to
  5000 on most evaluated tasks, with WMT as an exception`；
- `transfers across the evaluated configurations and variants`；
- `one-time calibration cost is amortizable`。

区分三个概念：

1. Standard-state held-out consistency；
2. accelerated-state agreement/distribution shift；
3. downstream task accuracy。

TC-APD commit 不可逆，因此“selected token 与 TC-APD 自己最终 token 一致”为 100%
是 by construction 的平凡事实，不能当 deployment guarantee。

## 10. 推荐给 Claude Code 的第一组命令

```bash
cd /leonardo/home/userexternal/mcao0000/prophet
git status --short --branch
sed -n '1,620p' iclr2027_review_rebuttal.md
sed -n '1,520p' README.md
find iclr27_rebuttal_leonardo/analysis -maxdepth 2 -type f | sort
squeue -u mcao0000 -o '%.18i %.30j %.2t %.10M %.10l %R'
sacct -j 64033378,64033380,64033381,64033383 \
  --format=JobIDRaw,JobName%32,State,Elapsed,Timelimit,ExitCode -X
```

先读取已有 summary 和 raw result，再决定是否提交任务。不要因为本地 TODO 未打勾就
重跑实验，也不要删除 scratch 结果。优先完成多 seed 汇总和 iteration analysis，随后
进入 rebuttal/paper 写作整合。
