# ICLR 2027 Review & Rebuttal Plan --- TC-APD

Paper: **Beyond Global Thresholds: Token-Level Calibrated Adaptive
Parallel Decoding for Diffusion Language Models**\
OpenReview: https://openreview.net/forum?id=bm5Bvszv36\
Submission: 39088

This document consolidates the three official reviews available on
OpenReview. It merges overlapping weaknesses and questions into
actionable work packages, prioritizes the explicit score-improvement
opportunity, and maps every reviewer concern to a TODO.

## 1. Review snapshot

  ----------------------------------------------------------------------------
  Reviewer                                    Rating Main assessment
  --------------------- ---------------------------- -------------------------
  jgET                             2 --- Weak Reject Likes the empirical
                                                     observation and
                                                     lightweight design;
                                                     questions offline cost,
                                                     calibration-data/domain
                                                     sensitivity, and
                                                     trajectory changes.

  iRo1                             2 --- Weak Reject Likes the token-dependent
                                                     consistency observation;
                                                     explicitly says
                                                     additional
                                                     calibration-size and
                                                     out-of-domain results
                                                     could lead to a higher
                                                     score.

  4QWr                             3 --- Weak Accept Finds the method and
                                                     efficiency results
                                                     convincing, but raises
                                                     concerns about
                                                     calibration protocol,
                                                     statistical validity,
                                                     distribution shift, and
                                                     evaluation fairness.
  ----------------------------------------------------------------------------

### Overall strategy

Prioritize: (1) calibration data-size ablation and chat/out-of-domain
evaluation; (2) calibration-source selection protocol; (3) independent
calibration validation; (4) statistical support for threshold selection;
and (5) matched-compute comparisons. These address the clearest
score-improvement opportunity and the most consequential technical
concerns.

Do not spend the rebuttal period indiscriminately adding benchmarks. The
central questions are whether calibration is statistically credible,
whether it generalizes, and whether its offline cost is justified.

## 2. Prioritized master TODO

## 2.0 实验状态快照（2026-11-10 更新）

下表是实际完成状态，优先级高于本节之后的 checkbox 文字。所有报告路径都在
`iclr27_rebuttal_leonardo/analysis/` 下，原始结果在
`/leonardo_scratch/large/userexternal/mcao0000/prophet-runs/`。

| 项目 | 状态 | 报告目录 |
|---|---|---|
| P0.1 Calibration-size ablation | 完成（seed 2027，100--5000） | `calibration_size/` |
| P0.1b 500 vs 5000 高频 coverage | 完成 | `n500_high_frequency_coverage/` |
| P0.2 OOD / chat 评估 | 完成（IFEval、ARC-Challenge、XSum-500、WMT16 en-ro、WritingPrompts-1024） | `ood_tasks/` |
| P0.3 Calibration-source 选择协议 | 待写作披露，无需新实验 | --- |
| P0.4 Held-out consistency 验证 | 完成（4000 估计 / 1000 验证） | `heldout_consistency/` |
| P1.1 最终输出层 distribution shift | 完成 | `standard_tc_apd_shift/` |
| P1.1b 逐 iteration trajectory audit | 完成（500 条配对 GSM8K 轨迹） | `iteration_trajectory_shift/` |
| P1.2 Threshold 采样敏感性 | 100/500 完成；200--5000 的 GSM8K seed 2028/2029 在 job 64033383 中 | `threshold_seed_sensitivity/`、`threshold_seed_sensitivity_n500/`、`multisize_seed_sensitivity/` |
| P1.3 Wilson confidence-bound 规则 | 完成 | `confidence_bound/` |
| P1.4 Matched-NFE paired bootstrap | 完成 | `matched_compute/` |
| P2.1 Offline cost 与摊销 | 完成 | `calibration_amortization/` |
| P2.2 配置与模型 variant transfer | 完成（block 8/16/32、length 512、chat template、INT8、fine-tuned） | `configuration_and_model_transfer/` |
| P2.3 长文本与 less-structured 生成 | 完成（WritingPrompts-100 @1024、XSum-500）；WMT length 512 按用户要求取消 | `ood_tasks/` |
| P2.4 与 online dynamic threshold 结合 | 讨论项，沿用已有 SOAR 集成 | --- |
| P2.5 降低 trace 收集成本 | 讨论项 | --- |

已知有意缺失、不要重跑的运行：WMT16 en-ro length 512 fixed-threshold probe；
P2.2 的 GSM8K length 512 Standard。

### P0 --- Highest priority

-   [x] **P0.1 Calibration data-size ablation**\
    Reviewers: iRo1 (explicit score-increase signal), 4QWr.\
    Vary calibration examples while fixing the domain and procedure.
    Suggested sizes: 1K, 2.5K, 5K, 10K (add 500 only if inexpensive).
    Report accuracy, NFE, speedup, per-token support/coverage,
    unsupported-token fallback rate, and threshold distribution. Plot
    calibration size against the speed--quality trade-off. Answer: do
    results stabilize around the current default, or is TC-APD highly
    data-hungry?\
    **Completed on Leonardo (single subset seed 2027):** Math calibration
    sizes 100, 200, 500, 1K, 2K, and 5K were evaluated on GSM8K,
    MATH500, and HumanEval against Standard and fixed-threshold APD.
    Reproducible tables, threshold/support statistics, and PNG/PDF figures
    are in `iclr27_rebuttal_leonardo/analysis/calibration_size/`. Exact
    calibration-audit fallback status is reported separately from the
    approximate final-output coverage obtained by re-tokenizing saved
    generations; candidate-level traces were not collected in these runs.

-   [x] **P0.2 Chat / out-of-domain evaluation**\
    Reviewers: iRo1 (explicit score-increase signal), jgET, 4QWr.\
    Add one manageable chat or instruction-following benchmark. Compare
    Standard decoding, fixed-threshold APD, and TC-APD with the same
    prompts and evaluation protocol. Report task quality and NFE /
    wall-clock efficiency. Prefer reusing the existing calibration table
    if the goal is to test cross-domain transfer; clearly state whether
    recalibration is used. Answer: does token-level calibration remain
    useful beyond math and code?

-   [ ] **P0.3 Calibration-source selection protocol and test-informed
    selection bias**\
    Reviewer: 4QWr.\
    State exactly how Minerva-Math calibration for LLaDA and
    mixed-domain calibration for Dream were selected, and whether final
    evaluation benchmarks influenced the choice. Separate calibration
    data, validation data for source/configuration selection, and final
    test data. If the original choice was informed by benchmark results,
    do not describe it as pre-specified; add a validation-only selection
    protocol or report a fixed, pre-declared source as a clean
    evaluation. Explicitly state whether calibration examples overlap
    with evaluation examples.

-   [x] **P0.4 Independent validation of offline consistency
    estimates**\
    Reviewer: 4QWr.\
    Split denoising traces into threshold-estimation and held-out
    validation subsets. Estimate token thresholds on the first subset,
    then measure realized final consistency on independent
    Standard-decoding traces. Report support counts and, where possible,
    results by support/coverage group. Do not claim that a nominal
    target such as (`\rho=0.995`{=tex}) is a formal guarantee unless the
    statistical evidence supports it. Answer: do estimated
    token-specific consistency rates generalize to unseen calibration
    traces?

### P1 --- Important technical concerns

-   [x] **P1.1 Distribution shift between Standard and accelerated
    trajectories**\
    Reviewers: jgET, 4QWr.\
    Separate (a) token-level agreement / trajectory changes, (b)
    final-answer agreement and answer transitions, and (c) downstream
    task accuracy. Reuse existing final-answer agreement,
    correct-to-incorrect / incorrect-to-correct transitions, EOS-aware
    agreement, and trajectory analyses. Explain that reproducing the
    Standard trajectory is not TC-APD's objective: it changes commitment
    decisions, and the relevant question is whether those changes harm
    task-level quality. Be transparent about sequence-level
    disagreement.

-   [x] **P1.2 Threshold candidate enumeration, monotonicity, and
    sampling noise**\
    Reviewer: 4QWr.\
    Document how candidate (`\tau`{=tex}) values are enumerated and
    whether monotonicity of empirical (R_v(`\tau`{=tex})) is enforced.
    If not, state so: finite-sample empirical conditional consistency
    need not be strictly monotonic. Assess threshold sensitivity using
    multiple calibration subsets or bootstrap resamples; report
    threshold variability or the fraction of tokens whose selected
    thresholds change substantially.

-   [x] **P1.3 (A\_{`\min`{=tex}}=100) versus (`\rho=0.995`{=tex});
    confidence bounds**\
    Reviewer: 4QWr.\
    Explain that (A\_{`\min`{=tex}}) is a minimum-support constraint for
    empirical estimation, not a statistical guarantee that the true
    consistency probability is at least 0.995. Report support
    distributions and, if feasible, binomial confidence intervals /
    lower confidence bounds. Do not interpret 100/100 observed successes
    as proof of a true 99.5% success probability. Consider a lightweight
    comparison between the current empirical threshold and a
    conservative confidence-bound-based rule; report its effect on
    coverage, NFE, and accuracy.

-   [x] **P1.4 Matched NFE / wall-clock budgets with paired
    uncertainty**\
    Reviewer: 4QWr.\
    Select Fixed APD and TC-APD operating points with closely matched
    NFE; add wall-clock matching if feasible. Evaluate on the same
    examples and report paired uncertainty, e.g. bootstrap confidence
    intervals for task accuracy and efficiency. Answer: does TC-APD
    retain an advantage at closely matched compute budgets, rather than
    only at the same maximum threshold?

### P2 --- Practicality, transfer, and extensions

-   [x] **P2.1 Offline calibration cost and amortization**\
    Reviewers: jgET, 4QWr.\
    Report calibration GPU-hours and hardware. Estimate break-even
    generated sequences: \[ N\^\* =
    `\frac{C_{\mathrm{calibration}}}{C_{\mathrm{baseline}}-C_{\mathrm{TC\text{-}APD}}}`{=tex}.
    \] Keep hardware and cost units consistent. State when recalibration
    is needed (e.g. model, tokenizer, or decoding setup changes).

-   [x] **P2.2 Transfer across decoding configurations**\
    Reviewer: 4QWr.\
    Address block size, maximum generation length, prompting template,
    quantization, and fine-tuned variants. Existing
    zero-shot-calibration to 3-shot MATH500 transfer is relevant
    evidence for prompting-format transfer. Prioritize block size and
    generation length if feasible. Do not infer cross-backbone or
    cross-tokenizer transfer from same-model prompt transfer.

-   [x] **P2.3 Long-context and creative / less-structured generation**\
    Reviewers: jgET, 4QWr.\
    A longer generation setting and the chat evaluation in P0.2 can
    jointly address much of this concern. Discuss the possible trade-off
    between aggressive early commitment and robustness for long or
    open-ended generations. Other languages are mentioned by Reviewer
    4QWr, but need not be a first-priority experiment if resources are
    limited.

-   [ ] **P2.4 Combination with online dynamic threshold methods**\
    Reviewer: jgET.\
    Explain complementary roles: TC-APD supplies offline token-specific
    thresholds, while online methods adapt decisions using
    inference-time signals. Use the existing SOAR integration as
    evidence of plug-and-play compatibility. A further combined method
    is optional and should not displace P0/P1 work.

-   [ ] **P2.5 Reduce offline trace-collection burden**\
    Reviewer: jgET.\
    Explain why full denoising traces are currently used to obtain
    self-final consistency labels. Discuss partial trajectories,
    subsampled denoising steps, or more efficient trace collection as
    future directions. A new method or large experiment is not essential
    unless inexpensive.

## 3. Reviewer-by-reviewer coverage

This section maps every weakness and question in the three reviews to
the TODO items above.

### Reviewer jgET --- Rating 2 (Weak Reject)

**Weakness 1 --- Offline calibration dependency** - Large-scale
denoising traces incur non-trivial one-off GPU cost. - Threshold tables
are model-specific and cannot be transferred zero-shot across DLM
backbones. - Mapped to: P2.1 (cost), P2.2 (transfer), P2.5 (reduce trace
burden).

**Weakness 2 --- Calibration-data sensitivity** - Performance depends on
offline trace distribution; mixed-domain calibration does not always
improve results. - Mapped to: P0.1 (data size), P0.3 (source-selection
protocol), P0.4 (independent validation).

**Weakness 3 --- Trajectory-change risk** - Earlier commits change
intermediate trajectories; individual outputs may differ from
conservative full denoising even if aggregate accuracy is similar. -
Mapped to: P1.1.

**Questions** 1. Can the heavy trace-collection burden be reduced or
eliminated, e.g. without full denoising trajectories? → P2.5. 2. How can
calibration tables generalize across domains without per-domain
recalibration? → P0.2, P0.3, P2.2. A chat evaluation is a first test,
not proof of universal transfer. 3. Can TC-APD be combined with online
dynamic threshold techniques to mitigate rare-token fallback? → P2.4. 4.
What is the trade-off between aggressive early commitment and robustness
for long-context / creative text generation? → P0.2, P2.3.

### Reviewer iRo1 --- Rating 2 (Weak Reject; explicit score-improvement signal)

**Weakness 1 --- Limited calibration data-size analysis** - No study of
how calibration-set size affects performance and robustness. - Mapped
to: P0.1.

**Weakness 2 --- Limited out-of-domain evaluation** - Evaluation mainly
covers math and code, without chat tasks. - Mapped to: P0.2.

**Questions** 1. How does calibration-set size affect performance and
robustness? → P0.1. 2. Can evaluation across math, code, and chat better
establish generalizability? → P0.2.

**Important:** The reviewer explicitly says they would be willing to
increase their score if the additional calibration-size and
out-of-domain results further validate robustness and effectiveness.
Therefore P0.1 and P0.2 should be the first two new experiments if
compute and time are constrained.

### Reviewer 4QWr --- Rating 3 (Weak Accept)

**Weakness 1 --- Possible test-informed calibration selection** -
Calibration sources were compared on the same final evaluation
benchmarks and the source with the best observed trade-off was
selected. - Mapped to: P0.3.

**Weakness 2 --- Mismatch between calibration and deployment
trajectories** - Consistency is estimated under Standard decoding, but
early commitment changes later model states; low sequence agreement
indicates distribution shift. - Mapped to: P1.1 and P0.4.

**Weakness 3 --- No out-of-sample calibration assessment** - No held-out
evidence that nominal (`\rho=0.995`{=tex}) is realized on independent
traces or TC-APD trajectories. - Mapped to: P0.4 and P1.1. Keep
independent Standard-trace validation distinct from
accelerated-trajectory diagnostics.

**Weakness 4 --- Potential threshold-selection bias** - The minimum
threshold satisfying empirical
(R_v(`\tau`{=tex})`\geq`{=tex}`\rho`{=tex}) is selected on the same
samples used to estimate (R_v); no confidence bounds, isotonic
calibration, or separate threshold-selection split. - Mapped to: P0.4,
P1.2, P1.3.

**Weakness 5 --- Limited support at a stringent target** -
(A\_{`\min`{=tex}}=100) may provide weak statistical evidence for
(`\rho=0.995`{=tex}). - Mapped to: P1.3.

**Weakness 6 --- Limited experimental scope** - Two models, three
benchmarks, fixed block size, and maximum generation length 256; unclear
behavior for longer generations, other languages, and less-structured
tasks. - Mapped to: P0.2, P2.2, P2.3.

**Weakness 7 --- Non-negligible offline cost** - Calibration takes about
3.5 hours on four A100 GPUs; recalibration may be needed after model,
tokenizer, or decoding changes. - Mapped to: P2.1, P2.2.

**Questions** 1. How are candidate (`\tau`{=tex}) values enumerated? Is
(R_v(`\tau`{=tex})) monotonic? How sensitive are thresholds to sampling
noise? → P1.2. 2. What is the realized consistency rate on independent
calibration traces and TC-APD-generated trajectories? → P0.4, P1.1. 3.
Were Minerva-Math for LLaDA and mixed-domain calibration for Dream
selected before final benchmark evaluation? Can a validation-only
source-selection protocol be reported? → P0.3. 4. Why is
(A\_{`\min`{=tex}}=100) sufficient for (`\rho=0.995`{=tex})? Can lower
confidence bounds be reported? → P1.3. 5. How much improvement remains
at tightly matched NFE or wall-clock budgets, with paired uncertainty? →
P1.4. 6. Which changes require recalibration: block size, generation
length, prompting, quantization, fine-tuned variants? → P2.2. 7. What is
the amortization point for offline calibration cost? → P2.1.

## 4. Existing analyses that can be reused

Based on the work already discussed for the current paper draft, reuse
the following rather than repeating them:

-   Vocabulary-wide token-conditioned consistency analysis.
-   Token-level trajectory changes and decoding-path visualization.
-   Final-answer agreement.
-   Correct-to-incorrect and incorrect-to-correct transitions.
-   EOS-aware agreement.
-   Threshold coverage, unsupported-token fallback, and capping
    analysis.
-   Zero-shot calibration transferred to 3-shot MATH500.
-   SOAR plug-and-play integration.
-   NFE and wall-clock throughput measurements.

These provide a starting point for distribution-shift, transfer, and
efficiency responses. They do **not** replace the new independent-trace
validation, calibration-size ablation, chat evaluation, or
calibration-source protocol clarification.

## 5. Suggested consolidated outputs

  -----------------------------------------------------------------------
  Output                  Contents                Main concern
  ----------------------- ----------------------- -----------------------
  Table A                 Calibration size        iRo1, 4QWr
                          vs. Accuracy / NFE /    
                          Speedup / coverage      

  Figure A                Calibration size        iRo1, 4QWr
                          vs. speed--quality      
                          trade-off and threshold 
                          / coverage stability    

  Table B                 Chat benchmark:         iRo1, jgET, 4QWr
                          Standard / Fixed APD /  
                          TC-APD                  

  Table C                 Held-out consistency:   4QWr
                          estimation split        
                          vs. independent         
                          validation split        

  Figure B or Table D     Threshold support,      4QWr
                          confidence bounds, and  
                          threshold sensitivity   

  Table E                 Matched-NFE comparison  4QWr
                          with paired accuracy    
                          uncertainty             

  Table F                 Calibration cost and    jgET, 4QWr
                          break-even generation   
                          count                   

  Appendix subsection     Existing trajectory,    jgET, 4QWr
                          answer-transition,      
                          EOS-aware, and coverage 
                          analyses                
  -----------------------------------------------------------------------

Keep the main paper focused on the most important new evidence; place
detailed statistical and behavioral analyses in the appendix.

## 6. Rebuttal writing principles

1.  Acknowledge legitimate limitations directly. Do not claim offline
    empirical consistency is a formal guarantee of
    accelerated-trajectory correctness.
2.  Separate the two distributions: held-out Standard traces test
    offline calibration estimation; TC-APD trajectories measure
    deployment-time distribution shift.
3.  Do not overclaim transfer. Same-backbone prompt transfer does not
    imply cross-model or cross-tokenizer transfer.
4.  Be precise about (`\rho`{=tex}). It is an empirical calibration
    target, not automatically a guaranteed population probability.
5.  Be transparent about source selection. If a source was selected
    after observing benchmark performance, do not present it as
    pre-registered or independently chosen.
6.  Use paired comparisons where possible: same examples and paired
    uncertainty are more informative than unpaired aggregate scores.
7.  Keep the method framing consistent: TC-APD is offline token-level
    threshold calibration with a lightweight inference-time lookup. Its
    goal is not to preserve Standard decoding outputs or trajectories
    exactly.
8.  Prioritize the explicit score-improvement opportunity: address
    iRo1's calibration-size and chat-domain requests first.

## 7. Recommended execution order

If time or compute is limited:

1.  Calibration data-size ablation.
2.  Chat / instruction-following evaluation.
3.  Calibration-source selection protocol clarification and
    validation-only evaluation.
4.  Independent held-out consistency validation.
5.  Threshold support / confidence-bound analysis.
6.  Matched-NFE comparison with paired uncertainty.
7.  Synthesize trajectory and final-answer distribution-shift analyses
    using existing results.
8.  Offline-cost amortization.
9.  Block-size / generation-length transfer.
10. Remaining discussion-only items: online combination and reduced
    trace collection.

### Bottom line

The strongest rebuttal will not simply add more benchmarks. It will show
that **TC-APD's token-specific thresholds can be estimated from finite
data, remain meaningful on independent traces, generalize beyond the
calibration domain, and provide measurable inference savings that
justify the offline cost**.

## 8. GPT_plus 的建议：除作者回复外，同步修改论文

ICLR 2027 允许作者在公开讨论期结束前修改标题、摘要、正文和补充材料。
Rebuttal 版本主文页数上限为 10 页，appendix 不限页数；但 reviewer 和
Area Chair 不保证阅读每一次 revision。因此建议等待主要新增分析完成后，
集中上传一次结构清晰、改动可追踪的修订稿，并在对每位 reviewer 的回复中
明确指出新增 section、table 和 appendix 的位置。

### 8.1 主文必须加入的内容

1.  **Calibration robustness 主结果。** 使用新增的主文页数放置一个紧凑
    表格或组合图，重点比较 Standard、Fixed 0.90、Math calibration 100、
    500 和 5000。报告三个主任务的平均 accuracy、NFE、speedup、有效
    token 数量和高频 candidate-mass coverage。强调：500 trajectories
    得到 410 个有效 thresholds，覆盖全部 Top-200 高频 candidate token，
    并覆盖 92.93% 的 candidate-token mass；其主任务及多数 OOD 任务的
    质量和速度已接近 5000 trajectories。不要声称 500 条在所有 domain
    都足够，WMT 是明确的长尾/跨语言例外。

2.  **Out-of-domain 结果。** 主文优先放 IFEval、ARC-Challenge、XSum 和
    WMT 的紧凑结果：IFEval 提供 instruction-following/chat evidence，
    ARC 提供稳定的离散自动评估，XSum 提供生成任务，WMT 作为较弱 transfer
    的诚实边界。WritingPrompts-100 放在 appendix，正文只用一句话概括：
    在 1024-token stress test 上，TC-APD 达到约 1.50x NFE speedup 和
    1.48x wall-clock speedup，自动重复指标仅有轻微变化。不要将 repetition
    指标描述为完整的创作质量评价。

3.  **Calibration protocol 的精确定义。** 明确写出候选阈值为
    0.01--1.00、步长 0.005，共 199 个 candidate values；对每个满足
    total support >= 200 的 token，选择满足 empirical consistency
    和 accepted support >= 100 的最低 threshold，以最大化 coverage。
    Empirical R_v(tau) 没有做 isotonic regression 或 monotonic envelope，
    finite-sample R_v(tau) 不保证严格单调。A_min=100 是 minimum-support
    constraint，不是统计保证；rho=0.995 是 nominal empirical target，
    不是 formal guarantee。Unsupported token 使用 default threshold 0.90。

4.  **Independent held-out validation。** 主文至少报告：使用 4000 条
    Standard traces 估计 threshold，在 1000 条不重叠 Standard traces 上
    获得 99.443% realized consistency，95% Wilson CI 为
    [99.431%, 99.454%]。只有最高 support group（>=1000 observations）
    达到约 99.503%，低 support group 泛化较弱。因此应将 99.5% 描述为
    empirical target，而不是严格保证。还要明确该实验只验证 unseen
    Standard states，不能自动保证 accelerated states。

5.  **Calibration-source selection 与 test-informed bias。** 明确说明
    Minerva-Math calibration source 的选择过程、是否观察过主测试结果、
    calibration examples 与 evaluation examples 是否重叠。如果原 source
    是在 exploratory benchmark comparisons 后选择的，应直接承认，不能
    描述成 pre-specified。新增 IFEval、ARC、XSum、WritingPrompts 等任务
    未参与原 source selection，可作为更干净的 held-out domain evidence。
    可采用如下表述：

    > The original source was selected after exploratory benchmark
    > comparisons; therefore, we do not treat the original in-domain
    > results as a source-selection-unbiased estimate. Our newly added
    > out-of-domain benchmarks were not used for calibration-source
    > selection.

6.  **Matched-compute 与 offline cost。** 主文用一句话或一行平均结果说明：
    Fixed threshold 0.77 与 TC-APD 的 NFE/时间已基本匹配；TC-APD 在三个
    主任务上的 paired accuracy point estimate 均略高，但 confidence
    intervals 包含 0，因此不能声称显著优势。同时报告 calibration
    GPU-hours、硬件和 break-even generation count N*，说明它是可摊销的
    one-time cost，而不是 negligible cost。

7.  **Iteration-level trajectory audit。** 新增的配对轨迹实验完成后，
    主文或 appendix 应清楚区分：(a) threshold policy 在 Standard states
    上的 realized consistency；(b) 同一 policy 在 TC-APD accelerated
    states 上相对 paired Standard final token 的 agreement；(c) 最终任务
    accuracy。TC-APD commit 是不可逆的，因此 selected token 与 TC-APD
    自己最终 token 的一致率为 100% by construction，不能把这个平凡结果
    当作部署保证。

### 8.2 Appendix 建议结构

Appendix 建议完整收录以下内容：

1.  100--5000 calibration-size 完整表格、coverage 和 figures；
2.  IFEval、WMT、ARC-Challenge、XSum 的 calibration-size sweeps；
3.  held-out consistency 的 estimation/validation support groups；
4.  100-trace 与 500-trace 多 seed threshold sensitivity；
5.  Wilson confidence-bound table 及 GSM8K/MATH500/HumanEval 结果；
6.  matched-NFE paired bootstrap confidence intervals；
7.  final-answer agreement、C->I/I->C、EOS-aware agreement、sequence shift
    和 iteration-level trajectory audit；
8.  block size 8/16、generation length 512、chat template transfer；
9.  INT8 quantization 与 fine-tuned model transfer；
10. WritingPrompts-100、length 1024 的 NFE、wall time 和 repetition；
11. calibration GPU-hours、threshold CPU cost 和 amortization analysis。

Appendix 虽然不限页数，但 reviewer 不一定阅读。因此 calibration-size、
OOD、held-out validation 和 source-selection honesty 必须进入主文，而不是
只放 appendix。

### 8.3 全文 claim 校正

上传 revision 前应全文搜索并审查以下词语：`guarantee`、`ensures`、
`preserves correctness`、`generalizes across domains`、
`negligible calibration cost`、`robust to decoding configurations`。
建议分别改为更准确的：

-   `empirically targets`；
-   `achieves high held-out consistency`；
-   `preserves aggregate task accuracy in our evaluations`；
-   `transfers across the evaluated domains/configurations`；
-   `amortizable one-time calibration cost`。

量化和 fine-tuned model 结果说明原 threshold table 仍能产生加速，但
accuracy 可能变化，因此不能写成“模型参数变化不需要 recalibration”。
同样，Wilson lower-bound rule 只对获得有效 conservative threshold 的
token 提供更强有限样本证据，不是 vocabulary-wide formal guarantee。

### 8.4 不建议做的修改

-   不要大改标题或方法定位，避免 revision 看起来像另一篇论文；
-   不要将所有新增 benchmark 塞入 abstract 或主文；
-   不要把论文改成 benchmark collection；
-   不要声称 500 条 calibration 在所有 domain 都充分；
-   不要将 WritingPrompts repetition 当作人类创作质量；
-   不要通过大规模重新排版制造无意义的 pdfdiff；
-   不要继续无目的扩充 benchmark，优先完成统计解释和写作整合。

### 8.5 Revision 上传与 reviewer 回复策略

建议采用一次主要 revision：

1.  等待 iteration-level trajectory audit 完成；
2.  一次性更新主文和 appendix；
3.  确保主文严格不超过 10 页并保持匿名；
4.  上传 revised PDF 和必要的匿名 supplementary/code；
5.  分别回复每位 reviewer；
6.  每条回复开头列出 revision 中的具体位置，例如：

    > We uploaded a revised manuscript. The main additions are the
    > calibration-size ablation (Sec. X/Table X), out-of-domain
    > evaluation (Sec. X/Table X), held-out consistency validation
    > (Sec. X), and matched-compute comparisons (Appendix X).

回复用于逐项回答 reviewer，revision 则应使论文在不依赖讨论串的情况下
也能独立呈现新增证据。当前最能改变评分的不是继续增加实验，而是把
calibration-size、OOD、held-out statistical validity、source-selection
honesty 和 matched-compute evidence 正式写进论文。
