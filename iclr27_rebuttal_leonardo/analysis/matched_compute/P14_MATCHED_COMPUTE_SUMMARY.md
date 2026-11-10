# P1.4 Matched-compute paired comparison

Fixed APD uses threshold 0.77; TC-APD uses the Math-5000 empirical table with maximum/default threshold 0.90. All differences and confidence intervals are paired over identical examples and defined as **TC-APD − Fixed**.

| Benchmark | Fixed acc. | TC acc. | Accuracy Δ (95% paired CI) | Fixed NFE | TC NFE | NFE Δ (95% CI) | Fixed s/item | TC s/item | Time Δ (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 72.33% | 73.24% | +0.91 pp [-0.68, +2.50] | 56.54 | 57.17 | +0.63 [+0.24, +1.03] | 2.295 | 2.308 | +0.013 [-0.003, +0.029] |
| MATH500 | 30.80% | 32.20% | +1.40 pp [-0.60, +3.60] | 82.27 | 83.10 | +0.83 [+0.19, +1.52] | 3.438 | 3.445 | +0.008 [-0.019, +0.035] |
| HumanEval | 37.20% | 37.80% | +0.61 pp [-1.83, +3.05] | 69.38 | 67.13 | -2.24 [-5.54, +0.85] | 3.135 | 3.067 | -0.068 [-0.211, +0.068] |

Bootstrap uses 20,000 paired resamples with seed 42. NFE and time relative gaps are retained in `paired_bootstrap_summary.csv` to assess whether the selected operating points are sufficiently matched.
