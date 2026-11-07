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

### P0 --- Highest priority

-   [ ] **P0.1 Calibration data-size ablation**\
    Reviewers: iRo1 (explicit score-increase signal), 4QWr.\
    Vary calibration examples while fixing the domain and procedure.
    Suggested sizes: 1K, 2.5K, 5K, 10K (add 500 only if inexpensive).
    Report accuracy, NFE, speedup, per-token support/coverage,
    unsupported-token fallback rate, and threshold distribution. Plot
    calibration size against the speed--quality trade-off. Answer: do
    results stabilize around the current default, or is TC-APD highly
    data-hungry?

-   [ ] **P0.2 Chat / out-of-domain evaluation**\
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

-   [ ] **P0.4 Independent validation of offline consistency
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

-   [ ] **P1.1 Distribution shift between Standard and accelerated
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

-   [ ] **P1.2 Threshold candidate enumeration, monotonicity, and
    sampling noise**\
    Reviewer: 4QWr.\
    Document how candidate (`\tau`{=tex}) values are enumerated and
    whether monotonicity of empirical (R_v(`\tau`{=tex})) is enforced.
    If not, state so: finite-sample empirical conditional consistency
    need not be strictly monotonic. Assess threshold sensitivity using
    multiple calibration subsets or bootstrap resamples; report
    threshold variability or the fraction of tokens whose selected
    thresholds change substantially.

-   [ ] **P1.3 (A\_{`\min`{=tex}}=100) versus (`\rho=0.995`{=tex});
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

-   [ ] **P1.4 Matched NFE / wall-clock budgets with paired
    uncertainty**\
    Reviewer: 4QWr.\
    Select Fixed APD and TC-APD operating points with closely matched
    NFE; add wall-clock matching if feasible. Evaluate on the same
    examples and report paired uncertainty, e.g. bootstrap confidence
    intervals for task accuracy and efficiency. Answer: does TC-APD
    retain an advantage at closely matched compute budgets, rather than
    only at the same maximum threshold?

### P2 --- Practicality, transfer, and extensions

-   [ ] **P2.1 Offline calibration cost and amortization**\
    Reviewers: jgET, 4QWr.\
    Report calibration GPU-hours and hardware. Estimate break-even
    generated sequences: \[ N\^\* =
    `\frac{C_{\mathrm{calibration}}}{C_{\mathrm{baseline}}-C_{\mathrm{TC\text{-}APD}}}`{=tex}.
    \] Keep hardware and cost units consistent. State when recalibration
    is needed (e.g. model, tokenizer, or decoding setup changes).

-   [ ] **P2.2 Transfer across decoding configurations**\
    Reviewer: 4QWr.\
    Address block size, maximum generation length, prompting template,
    quantization, and fine-tuned variants. Existing
    zero-shot-calibration to 3-shot MATH500 transfer is relevant
    evidence for prompting-format transfer. Prioritize block size and
    generation length if feasible. Do not infer cross-backbone or
    cross-tokenizer transfer from same-model prompt transfer.

-   [ ] **P2.3 Long-context and creative / less-structured generation**\
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
