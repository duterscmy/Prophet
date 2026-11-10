# Rebuttal — Beyond Global Thresholds: Token-Level Calibrated Adaptive Parallel Decoding for Diffusion Language Models

Submission 39088 · OpenReview: <https://openreview.net/forum?id=bm5Bvszv36>

We thank all three reviewers for their careful and constructive reviews. The
reviews converge on four questions: how much calibration data is needed, whether
calibration transfers beyond math and code, whether the calibration statistics
are sound and verified out of sample, and whether the efficiency gain survives a
fair compute comparison. We ran new experiments for each, and we have revised the
paper accordingly.

**Conventions used in every table below.** All runs use LLaDA-8B-Instruct with
`gen_length=256`, `steps=256`, `block_length=32`, `batch_size=1`,
`max/default threshold=0.90`, `min threshold=0.01`, unless stated otherwise.
`NFE` is the mean number of model forward calls per generated sequence and
`s/it` the mean decoding seconds per sequence. `SpeedUp` is the ratio of
macro-average NFE (Standard / setting); wall-clock ratios are reported
separately where they matter. "Math-*m*" denotes a table calibrated on *m*
Standard-decoding trajectories from the Minerva-Math training split.

---

## Reviewer jgET

We appreciate the recognition of the empirical observation and of the
lightweight design, and we address the three concerns in turn.

### W1 & Q1 — Offline calibration cost, and reducing the trace-collection burden

We agree this cost must be stated explicitly rather than described as
negligible, so we now report it as a one-time, amortizable cost with full
accounting. Normalizing everything to one A100 (Math Standard decoding costs
10.47 GPU-seconds per trajectory), and defining the break-even point as the
number of deployed sequences at which the saved inference time repays the
offline collection:

| Calibration trajectories | A100 GPU-hours | Mean break-even generations |
|---:|---:|---:|
| 100 | 0.29 | 138 |
| 500 | 1.45 | 686 |
| 5,000 | 14.54 | 6,788 |

Building the threshold tables from saved traces costs 124 seconds on 8 CPU cores
and 0 GPU-hours, so trajectory collection dominates entirely.

Our new calibration-size study (please see our reply to Reviewer iRo1) is also
the most direct answer to Q1: **100 trajectories already recover almost all of
the measured NFE reduction** of the 5,000-trajectory table, which lowers the
offline cost to 0.29 A100 GPU-hours and the break-even point to roughly 138
generated sequences. The reason is that the useful signal is concentrated in
high-frequency tokens: a 500-trajectory table contains only 28.2% as many
token-specific entries as the 5,000-trajectory table, yet covers all top-200
candidate tokens and 92.9% of the candidate-token probability mass.

Eliminating full trajectories altogether is a direction we find promising and do
not yet claim: self-final consistency labels currently require knowing each
position's final token, which is why we collect complete denoising traces.
Subsampling denoising steps, truncating trajectories after the block in which
most positions have converged, and reusing traces collected during ordinary
serving are all compatible with our estimator, and we now discuss them as future
work rather than as results.

On transfer: we confirm that tables are tied to a model and tokenizer and that we
make no cross-backbone zero-shot claim. New experiments (please see 4QWr W6 & Q6)
show that one table does keep accelerating decoding across block sizes 8/16/32,
generation length 512, a chat template, INT8 GPTQ weights, and a fine-tuned
checkpoint sharing the base tokenizer — while accuracy does move under the model
variants, so we now recommend recalibration after changing weights.

### W2 & Q2 — Calibration-data sensitivity and cross-domain robustness

This is a fair concern, and we tested it instead of arguing it. Reusing the Math
table **without any recalibration**, we added four out-of-domain tasks:
instruction following (IFEval), natural-language understanding
(ARC-Challenge), news summarization (XSum-500) and open-ended generation
(WritingPrompts-100 at generation length 1024). Full tables are in our reply to
Reviewer iRo1 (W2 & Q2). From 500 calibration trajectories upward, quality is
preserved and NFE is consistently below the fixed-threshold baseline on all four.

We also report the clearest negative result we found: on **WMT16 en-ro**, the
Math table costs 1.58 BLEU relative to Standard decoding (14.38 → 12.80) while
still giving a 5.67x NFE reduction. A cross-lingual task has a different
long-tail token distribution, and this is exactly where a math-calibrated table
should be expected to lose quality. We therefore do not claim domain-universal
robustness; we claim that one compact high-frequency table transfers across the
evaluated English tasks, and that a materially different token distribution calls
for recalibration or a mixed-domain source. We now also report token-coverage and
held-out consistency diagnostics that let a practitioner detect this case before
deployment, instead of discovering it from a quality drop.

### W3 & Q3 — Trajectory change, and the early-commit / robustness trade-off

We fully agree with the premise, and we think it is better to quantify the shift
than to downplay it. We re-ran GSM8K with per-iteration candidate logging and
analyzed 500 paired Standard / TC-APD trajectories:

| Population | Observations | Agreement with the paired Standard final token |
|---|---:|---:|
| Policy-eligible candidates on Standard states | 963,585 | 99.06% |
| Policy-eligible candidates on TC-APD states | 119,370 | 62.42% |
| Actual TC-APD commits | 128,000 | 62.95% |

Agreement decays by block (94.0% in block 1 to 48.6% in block 7) because the
reference conditions on a prefix the accelerated run no longer shares: once the
two runs differ at one position, the Standard token at later positions is no
longer the token a correct accelerated run should produce. So this table measures
accumulated path divergence, not a per-step error rate.

At the output level, 79.8% of final token sequences differ from Standard, yet
final-answer agreement is 76.3%, binary correctness agrees on 91.4% of examples,
and the flips nearly cancel (56 correct→incorrect vs. 57 incorrect→correct, i.e.
+0.08 pp accuracy). We have made the framing in the paper explicit: reproducing
the Standard trajectory is not TC-APD's objective; the relevant question is
whether the changed commitment decisions harm task-level quality, and in our
evaluations they do not in aggregate — while individual outputs do change, as the
reviewer notes.

For long-form and less-structured generation we added a 1,024-token stress test
(WritingPrompts-100) and XSum-500. TC-APD reaches a 1.50x NFE and 1.48x
wall-clock speedup at 1,024 tokens, with automatic degeneration metrics moving
only slightly (4-gram repetition 28.7% for Standard vs. 29.1% for Math-5000;
31.9% at Math-500 is the weakest point). We are explicit that repetition
statistics measure degeneration, not human-judged writing quality, so we present
this as an efficiency and degeneration stress test rather than a creative-quality
evaluation.

### Q4 — Combination with online adaptive thresholding

We see the two as complementary and composable rather than competing. TC-APD
contributes an offline, token-conditioned *prior* on how much confidence a given
token needs; online schedulers contribute inference-time *state*, such as
remaining masks, block progress or step budget, which an offline table cannot
observe. Concretely, a token-level table can simply replace the single global
threshold inside such a scheduler, and our paper already includes a plug-and-play
TC-APD + SOAR composition demonstrating this. For the rare-token fallback the
reviewer highlights, this is the natural remedy: uncovered tokens currently fall
back to the global default 0.90, and an online signal is a better decision rule
there than a constant. We now discuss this explicitly, and we agree a dedicated
hybrid method is a promising follow-up rather than a claim of this paper.

---

## Reviewer iRo1

We thank the reviewer for the clear and actionable requests, and for the
willingness to reconsider the score. We ran both requested studies.

### W1 & Q1 — Effect of calibration-set size

The reviewer is right that the paper's choice of 5,000 trajectories was made only
to give each token enough observations for threshold estimation, and that we never
asked how much calibration data is actually needed. We therefore recalibrated
LLaDA-8B-Instruct from 100 to 5,000 Math trajectories and re-evaluated all three
in-domain benchmarks:

| Calibration size | Valid token thresholds | GSM8K Acc | GSM8K NFE | MATH500 Acc | MATH500 NFE | HumanEval Acc | HumanEval NFE | Avg Acc | Avg NFE | SpeedUp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | — | 73.16 | 256.00 | 31.40 | 256.00 | 37.20 | 256.00 | 47.25 | 256.00 | 1.00 |
| Fixed threshold 0.90 | — | 73.09 | 77.62 | 31.80 | 106.27 | 37.20 | 98.67 | 47.36 | 94.19 | 2.72 |
| 100 | 141 | 73.31 | 59.71 | 32.40 | 82.38 | 38.41 | 72.87 | 48.04 | 71.65 | 3.57 |
| 200 | 235 | 73.39 | 61.79 | 31.60 | 85.80 | 36.59 | 72.35 | 47.19 | 73.31 | 3.49 |
| 500 | 410 | 74.07 | 60.77 | 32.20 | 85.16 | 37.20 | 67.23 | 47.82 | 71.05 | 3.60 |
| 1,000 | 619 | 72.71 | 58.80 | 31.60 | 83.83 | 38.41 | 68.51 | 47.57 | 70.38 | 3.64 |
| 2,000 | 910 | 72.86 | 58.92 | 31.60 | 83.56 | 35.98 | 68.62 | 46.81 | 70.37 | 3.64 |
| 5,000 | 1,453 | 73.24 | 57.17 | 32.20 | 83.10 | 37.80 | 67.13 | 47.75 | 69.13 | 3.70 |

Three observations. First, **100 trajectories already match the
fixed-threshold baseline in average accuracy while needing substantially fewer
forward calls** (71.65 vs. 94.19 NFE). Second, more data mainly buys speed, not
accuracy: NFE falls monotonically from 71.65 to 69.13 as the table grows from 100
to 5,000 entries. Third, small tables do fluctuate in quality — 200 trajectories
dip slightly — and the accuracy/speed trade-off becomes stable from 500 upward.
Accuracy is not monotone in calibration size, and we no longer suggest otherwise.

Why 500 trajectories nearly match 5,000 is explained by token frequency. On the
21.1M intermediate candidate records of the 5,000-trajectory pool:

| Metric | 500 trajectories | 5,000 trajectories |
|---|---:|---:|
| Valid token thresholds | 410 | 1,453 |
| Candidate-token probability mass covered | 92.93% | 98.22% |

The 500-trajectory table covers **all top-200 candidate tokens**, which alone
account for 88.9% of candidate mass — EOS, newline and space, digits 0–6,
frequent English words such as *the*, *of*, *is*, and math symbols such as
`$ \ = + - ^ { } \frac`. On the 410 tokens the two tables share, thresholds still
differ by 0.103 on average, so we describe the small table as a *compact
high-frequency approximation* rather than a reconstruction of the large one.

Because "robustness" also means insensitivity to *which* trajectories are drawn,
we added a sampling-robustness study: three independent calibration subsets
(seeds 2027/2028/2029) per size, each recalibrated and re-evaluated. For the two
small-data operating points (`Mean ± Std`, sample standard deviation):

| Size | GSM8K Acc | GSM8K NFE | MATH500 Acc | MATH500 NFE | HumanEval Acc | HumanEval NFE | Avg Acc | Avg NFE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 73.24 ± 0.13 | 59.56 ± 0.75 | 31.00 ± 1.25 | 82.16 ± 0.53 | 37.20 ± 1.06 | 71.24 ± 3.10 | 47.14 ± 0.79 | 70.99 ± 1.16 |
| 500 | 73.14 ± 0.91 | 60.62 ± 1.12 | 31.33 ± 1.33 | 83.45 ± 2.08 | 36.99 ± 0.93 | 69.64 ± 2.14 | 47.15 ± 1.00 | 71.24 ± 0.72 |

Efficiency is highly reproducible across subsets (average NFE varies by about
1–2%), while *which* tokens receive a threshold is much more subset-dependent: the
three 100-trajectory tables share only 47.1% of their token union (54.7% at 500),
and thresholds for shared tokens differ by 0.110 on average (0.104 at 500). The
honest reading is that the aggregate speed/quality operating point is stable at
small calibration sizes, but individual token thresholds are not yet
well-determined there.

### W2 & Q2 — Out-of-domain evaluation beyond math and code

We agree this was the main gap. Reusing the **same** Math table with **no
recalibration**, we added instruction following, natural-language understanding,
summarization and open-ended generation, and swept calibration size on each:

| Calibration size | IFEval inst-strict ↑ | IFEval prompt-strict ↑ | IFEval NFE ↓ | ARC-C Acc ↑ | ARC-C NFE ↓ | XSum R-1 ↑ | XSum R-2 ↑ | XSum NFE ↓ | WP 4-gram rep. ↓ | WP NFE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 66.79% | 57.49% | 256.00 | 84.64% | 256.00 | 29.95 | 8.80 | 256.00 | 28.71% | 1024.00 |
| Fixed threshold 0.90 | 66.79% | 57.49% | 137.46 | 84.64% | 13.57 | 29.93 | 8.79 | 46.75 | 29.01% | 725.41 |
| 100 | 66.19% | 57.12% | 128.08 | 66.04% | 10.34 | 29.88 | 8.79 | 37.05 | 29.59% | 692.77 |
| 200 | 66.91% | 57.49% | 128.66 | 83.79% | 10.27 | 29.98 | 9.07 | 36.66 | 30.18% | 694.31 |
| 500 | 66.55% | 57.67% | 126.74 | 84.73% | 10.08 | 29.92 | 9.02 | 35.95 | 31.86% | 683.85 |
| 1,000 | 66.31% | 57.30% | 126.33 | 84.04% | 9.99 | 29.96 | 9.00 | 35.87 | 29.96% | 684.01 |
| 2,000 | 66.79% | 56.93% | 126.13 | 84.56% | 10.11 | 29.94 | 9.02 | 36.17 | 31.12% | 679.73 |
| 5,000 | 67.03% | 57.86% | 124.62 | 84.56% | 10.12 | 29.88 | 8.98 | 35.81 | 29.07% | 681.3 |

WritingPrompts-100 uses generation length 1,024; the other three use 256. XSum
uses a fixed 500-example subset.

The four tasks agree: from 500 calibration trajectories upward, task quality is
preserved relative to both Standard and fixed-threshold decoding, while NFE is
consistently lower than the fixed threshold — including a 2.05x NFE and 2.02x
wall-clock speedup on IFEval at Math-5000, and 7.15x NFE on XSum. We highlight two
honest caveats. **ARC-Challenge at 100 trajectories collapses to 66.04%** (−18.6
pp), recovering from 200 onward; this is the clearest evidence in our study that
very small tables are risky out of domain, and it is why we do not claim that 100
trajectories suffice in general. And on WritingPrompts the 500-trajectory table is
the weakest degeneration point (31.9% vs. 28.7% 4-gram repetition), so we present
these numbers as a degeneration stress test, not as creative-writing quality.

Together with the WMT16 en-ro boundary case discussed in our reply to Reviewer
jgET (W2 & Q2), we believe this gives a fair picture: the method generalizes
across the evaluated English math, code, chat, understanding and summarization
tasks without recalibration, and we state its limits rather than overclaiming.

---

## Reviewer 4QWr

We are grateful for the detailed statistical critique — it improved the paper
most. We agree with essentially all of it and have corrected the claims
accordingly.

### W4 & Q1 — Threshold enumeration, monotonicity and sampling noise

Full protocol, now stated explicitly in the paper. Candidate thresholds are
enumerated on a fixed grid from 0.01 to 1.00 in steps of 0.005 (199 candidates).
A token is eligible if it has at least 200 total candidate observations; among
eligible tokens we select the **lowest** candidate τ whose empirical conditional
consistency *R_v(τ)* reaches ρ = 0.995 **and** whose accepted support is at least
100, which maximizes coverage subject to the constraints. Tokens without a valid
threshold fall back to the global default 0.90.

We confirm that **monotonicity of *R_v(τ)* is not enforced**: we apply neither
isotonic regression nor a monotone envelope, and finite-sample *R_v(τ)* is indeed
not guaranteed to be monotone. We have removed any wording implying otherwise.

On sampling noise, we measured it directly with three independent calibration
subsets per size (details in our reply to iRo1, W1 & Q1):

| Calibration size | Valid thresholds (3 seeds) | Union | 3-way intersection | Jaccard | Mean \|Δτ\| on shared tokens | Shared tokens within 0.05 |
|---:|---|---:|---:|---:|---:|---:|
| 100 | 141 / 127 / 142 | 191 | 90 | 47.1% | 0.110 | 26.2% |
| 500 | 410 / 408 / 439 | 550 | 301 | 54.7% | 0.104 | 32.5% |
| 1,000 | 619 / 620 / 614 | 791 | 470 | 59.4% | 0.088 | 41.8% |
| 2,000 | 910 / 916 / 911 | 1,102 | 730 | 66.2% | 0.073 | 51.9% |

The reviewer's concern is confirmed at the token level: threshold membership and
values are genuinely sensitive to the calibration draw, and only about half of
the tokens are selected by all three subsets at small sizes. Agreement improves
monotonically with more data. What is stable is the aggregate operating point —
average NFE varies by 1–2% across subsets. We now present TC-APD's
reliability claim at that aggregate level and explicitly flag per-token
thresholds as noisy estimates at small calibration sizes. (At 5,000 the subset
equals our full trajectory pool, so the three seeds coincide by construction; we
do not report that as evidence of sampling robustness.)

### W2 & W3 & Q2 — Realized consistency out of sample, and on TC-APD's own trajectories

We ran both experiments the reviewer asks for and we separate them carefully,
because they answer different questions.

**(a) Independent Standard traces.** We split the Math traces once into 4,000 for
threshold estimation and 1,000 disjoint traces for validation. On the held-out
traces the accepted tokens realize **99.443% consistency, 95% Wilson CI
[99.431%, 99.454%]** (99.447% under the full lookup policy, including the default
fallback). The estimator therefore generalizes to unseen Standard states, but
*not uniformly across tokens*: consistency degrades sharply with support.

| Held-out accepted support per token | Token types | Realized consistency |
|---|---:|---:|
| 1–9 | 47 | 94.09% |
| 10–24 | 125 | 95.14% |
| 25–49 | 190 | 97.04% |
| 50–99 | 272 | 98.45% |
| ≥ 100 | 599 | 99.47% |

Only well-supported tokens approach the nominal target. We consequently describe
ρ = 0.995 as an **empirical calibration target, not a formal guarantee**, and we
have removed guarantee-style wording throughout.

**(b) TC-APD's own trajectories.** This is the more important question, and it
needs one caveat stated up front: because TC-APD commits are irreversible, the
agreement between an accepted token and TC-APD's *own* final token is 100% by
construction, so that number is vacuous and we do not report it as a deployment
guarantee. We therefore re-ran GSM8K with per-iteration logging and measured, on
500 paired trajectories, agreement against the **paired Standard final token at
the same position**: 99.06% for policy-eligible candidates on Standard states
versus **62.42% on accelerated states** (62.95% for actual commits), decaying
from 94.0% in block 1 to 48.6% in block 7. The shift the reviewer predicted from
Appendix D is real and substantial.

We also note that the 99.06% figure on GSM8K Standard states is itself below the
in-domain held-out 99.443%, which we attribute to the Math → GSM8K domain shift
rather than to the runtime threshold cap (capped candidates are only 0.17% of the
eligible population and agree at 99.94%).

Two honest qualifications on interpreting (b). The accelerated-state reference
conditions on a prefix the accelerated run no longer shares, so the decay
measures *accumulated* path divergence rather than a per-step error rate; and
sequence divergence does not translate into aggregate quality loss here
(final-answer agreement 76.3%, correctness agreement 91.4%, 56 correct→incorrect
vs. 57 incorrect→correct, accuracy +0.08 pp). Our position in the revision is
therefore: offline calibration is validated as an *estimator* on independent
Standard traces; deployment-time behaviour is a measurable distribution shift
that we report rather than claim away; and the quantity that matters for users is
downstream task quality at matched compute.

### W1 & Q3 — Calibration-source selection

We accept this criticism without reservation. The Minerva-Math source for LLaDA
and the mixed-domain source for Dream were chosen **after** exploratory
comparisons that included the same benchmarks used for final evaluation. We
therefore no longer present that choice as pre-specified, and we have added the
following statement to the paper:

> The original calibration source was selected after exploratory benchmark
> comparisons; we therefore do not treat the original in-domain results as a
> source-selection-unbiased estimate.

We also confirm that calibration examples never overlap with evaluation examples:
calibration uses Minerva-Math *training*-split trajectories only.

For a cleaner estimate we point to the benchmarks added during this rebuttal —
IFEval, ARC-Challenge, XSum-500, WritingPrompts-100 and WMT16 en-ro — none of
which played any role in selecting the calibration source. They are genuinely
held-out domains for the already-fixed Math table, and the method's behaviour on
them (including the WMT regression we report) is unbiased by source selection.

### W5 & Q4 — Why A_min = 100, and confidence bounds

The reviewer is right, and we have corrected the framing: **A_min = 100 is a
minimum-support constraint for empirical estimation, not statistical evidence
that the true consistency probability is at least 0.995**. Observing 100
successes out of 100 does not establish a 99.5% success probability, and we no
longer imply it. The support-group table above is the empirical counterpart of
this concern: tokens with little accepted support generalize clearly worse.

We also implemented the confidence-bound rule the reviewer suggests. Requiring a
one-sided 95% **Wilson lower bound** to reach 99.5% at the selected threshold:

| Statistic | Empirical rule | Wilson lower-bound rule |
|---|---:|---:|
| Eligible token types | 1,898 | 1,898 |
| Valid token thresholds | 1,453 | 491 |
| Accepted calibration records at selected thresholds | 7,777,598 | 7,364,280 |

The conservative rule keeps 33.79% of the token types, and on all 491 shared
tokens it assigns an **equal or higher** threshold — i.e. the empirical rule is
indeed optimistic, exactly as the reviewer suspected, and the bias is one-sided.
We report this variant and its downstream GSM8K / MATH500 / HumanEval results in
the revision. We are also explicit about what it does *not* give: tokens with no
valid conservative threshold still fall back to 0.90, so this is a stronger
finite-sample statement on a calibrated subset, not a vocabulary-wide guarantee.

### Q5 — Matched-compute comparison with paired uncertainty

We agree that comparing at an equal *maximum threshold* is not a fair efficiency
comparison. We tuned the fixed threshold to 0.77 so that its NFE and wall-clock
cost closely match TC-APD (Math-5000, max 0.90), evaluated both on identical
examples, and computed 20,000 paired bootstrap resamples:

| Task | Fixed 0.77 — Acc / NFE / s/it | TC-APD — Acc / NFE / s/it | Paired Δ accuracy (95% CI) |
|---|---:|---:|---:|
| GSM8K | 72.33 / 56.54 / 2.295 | 73.24 / 57.17 / 2.308 | +0.91 pp [−0.68, +2.50] |
| MATH500 | 30.80 / 82.27 / 3.438 | 32.20 / 83.10 / 3.445 | +1.40 pp [−0.60, +3.60] |
| HumanEval | 37.20 / 69.38 / 3.135 | 37.80 / 67.13 / 3.067 | +0.61 pp [−1.83, +3.05] |

At tightly matched compute, TC-APD's accuracy point estimate is higher on all
three tasks, but **every confidence interval contains zero**, so we explicitly do
not claim a statistically significant accuracy advantage at matched NFE. We state
the honest conclusion instead: TC-APD's reliable benefit is that it reaches a
better speed/quality operating point automatically, without tuning a global
threshold per task, and at matched compute it is at least as good as a
hand-tuned fixed threshold.

### W6 & Q6 — Which changes require recalibration

We tested this directly, reusing the single Math-5000 table calibrated at
`length 256, block 32` with no recalibration. Macro-average over GSM8K, MATH500
and HumanEval, each compared against Standard decoding in its *own*
configuration:

| Changed setting | Quality vs. Standard | NFE speedup retained |
|---|---|---|
| Block length 16 | −0.45 pp | 2.86–3.96x |
| Block length 8 | +0.52 pp | 2.51–3.26x |
| Generation length 512 | +0.60 pp on MATH500/HumanEval | 4.20–4.91x |
| Chat template | +0.15 / +2.60 pp on GSM8K/MATH500 | 3.08–4.76x |
| INT8 GPTQ weights | +2.67 / +2.67 / −3.66 pp | 3.24–4.55x |
| Fine-tuned checkpoint (same tokenizer) | −4.25 / −6.40 / −2.44 pp | 4.56–7.68x |

Our reading, stated conservatively in the paper: **block size, generation length
and prompting template transfer well**; quantization keeps the speedup with
mixed accuracy effects; and the **fine-tuned checkpoint is the case where we
recommend recalibration**, since reusing the base-model table costs 2.4–6.4 pp
there (and more than the fixed threshold loses). Nothing here transfers across a
different backbone or tokenizer, and we make no such claim. Two caveats for
transparency: HumanEval under the chat template collapses for *all* decoders
(Standard 8.54%), so that cell measures the prompting format and not threshold
transfer; and the INT8 runs use reduced example budgets (300/300/164) because
that variant decodes very slowly.

### W7 & Q7 — Amortization point

Reported in full in our reply to Reviewer jgET (W1 & Q1). Normalized to one
A100, the break-even point is about **138 generated sequences for a 100-trajectory
table, 686 for 500 trajectories and 6,788 for 5,000 trajectories**, with
table construction itself costing 124 CPU-seconds and no GPU time. A
conservative end-to-end accounting from the raw collection log (which includes
model setup and task-side post-processing) is 1.074x these numbers. Since the
100-trajectory table already recovers nearly all of the measured NFE reduction,
short-lived deployments can amortize the cost within a few hundred generations,
while larger tables become economical for sustained serving. We now describe
this as an amortizable one-time cost and have removed any suggestion that it is
negligible.

---

## Summary of changes to the paper

- New calibration-size ablation (100–5,000) with coverage and threshold statistics.
- New out-of-domain evaluation: IFEval, ARC-Challenge, XSum-500,
  WritingPrompts-100 (length 1,024) and WMT16 en-ro, with no recalibration.
- New independent held-out consistency validation with Wilson intervals and
  support-group breakdown.
- New per-iteration paired trajectory audit of Standard vs. TC-APD states.
- New multi-subset sampling-robustness study of threshold stability.
- New Wilson lower-bound threshold variant.
- New matched-NFE comparison with paired bootstrap confidence intervals.
- New configuration and model-variant transfer study.
- Calibration cost and amortization analysis.
- Claim corrections throughout: ρ = 0.995 is described as an empirical target
  rather than a guarantee, A_min = 100 as a support constraint, calibration cost
  as amortizable rather than negligible, and the calibration source as selected
  after exploratory benchmark comparisons.

We thank the reviewers again for feedback that materially improved both the
evidence and the precision of our claims, and we are happy to run further
experiments during the discussion period.
