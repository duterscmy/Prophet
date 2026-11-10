# P0.1 Calibration-size ablation

Single calibration-subset seed: 2027. Accuracy metrics are HumanEval pass@1, MATH500 math_verify, and GSM8K flexible-extract.

Every speedup below is the ratio of macro-averaged values: NFE (or seconds) is averaged across the three benchmarks first, and the speedup is the ratio of those two averages, not the mean of three separate per-benchmark ratios. The unqualified `Speedup` columns use NFE as the basis.

## Main summary

| Calibration trajectories | Macro Acc. (%) | Mean NFE | s/item | NFE speedup | Wall-clock speedup | Valid thresholds | Output-token coverage (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 48.04 | 71.65 | 3.043 | 3.57× | 3.51× | 141 | 66.74 |
| 200 | 47.19 | 73.31 | 3.122 | 3.49× | 3.42× | 235 | 71.27 |
| 500 | 47.82 | 71.05 | 3.006 | 3.60× | 3.56× | 410 | 74.80 |
| 1000 | 47.57 | 70.38 | 2.995 | 3.64× | 3.57× | 619 | 79.80 |
| 2000 | 46.81 | 70.37 | 2.992 | 3.64× | 3.57× | 910 | 82.62 |
| 5000 | 47.75 | 69.14 | 2.940 | 3.70× | 3.64× | 1,453 | 86.39 |

## Full comparison

| Setting | GSM8K Acc/NFE/s | MATH500 Acc/NFE/s | HumanEval Acc/NFE/s | Macro Acc/NFE/s | Speedup | Valid thresholds |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 73.16/256.00/10.158 | 31.40/256.00/10.469 | 37.20/256.00/11.454 | 47.25/256.00/10.694 | 1.00× | – |
| Fixed threshold 0.90 | 73.09/77.62/3.112 | 31.80/106.27/4.393 | 37.20/98.67/4.450 | 47.36/94.19/3.985 | 2.72× | – |
| Math calibration, 100 trajectories | 73.31/59.71/2.404 | 32.40/82.38/3.421 | 38.41/72.87/3.305 | 48.04/71.65/3.043 | 3.57× | 141 |
| Math calibration, 200 trajectories | 73.39/61.79/2.495 | 31.60/85.80/3.570 | 36.59/72.35/3.302 | 47.19/73.31/3.122 | 3.49× | 235 |
| Math calibration, 500 trajectories | 74.07/60.77/2.438 | 32.20/85.16/3.525 | 37.20/67.23/3.055 | 47.82/71.05/3.006 | 3.60× | 410 |
| Math calibration, 1000 trajectories | 72.71/58.80/2.377 | 31.60/83.83/3.489 | 38.41/68.51/3.118 | 47.57/70.38/2.995 | 3.64× | 619 |
| Math calibration, 2000 trajectories | 72.86/58.92/2.380 | 31.60/83.56/3.492 | 35.98/68.62/3.105 | 46.81/70.37/2.992 | 3.64× | 910 |
| Math calibration, 5000 trajectories | 73.24/57.17/2.308 | 32.20/83.10/3.445 | 37.80/67.13/3.067 | 47.75/69.14/2.940 | 3.70× | 1,453 |

## Rebuttal-ready observations

- 100 trajectories recover 98.7% of the macro NFE reduction obtained with 5,000 trajectories.
- Valid token-specific thresholds grow from 141 to 1,453.
- Macro NFE changes from 71.65 at 100 trajectories to 69.14 at 5,000 trajectories.
- Macro accuracy ranges from 46.81% to 48.04% and is not monotonic in calibration size.
- Accuracy uncertainty intervals in evaluation_metrics_long.csv use lm-eval standard errors and a normal 95% approximation.

## Coverage interpretation

`fallback_no_valid_threshold` is an exact calibration-audit status among token types with sufficient total count. Final-output coverage is an approximate deployment statistic computed by re-tokenizing saved final responses; candidate-level decoder traces were intentionally disabled for these jobs, so it must not be described as exact candidate-trace fallback.

## Per-benchmark dynamic results

| Size | Benchmark | Acc. (%) | 95% CI | NFE | s/item | Wall-clock speedup |
|---:|---|---:|---:|---:|---:|---:|
| 100 | GSM8K | 73.31 | [70.93, 75.70] | 59.71 | 2.404 | 4.23× |
| 100 | HumanEval | 38.41 | [30.95, 45.88] | 72.87 | 3.305 | 3.47× |
| 100 | MATH500 | 32.40 | [28.29, 36.51] | 82.38 | 3.421 | 3.06× |
| 200 | GSM8K | 73.39 | [71.00, 75.77] | 61.79 | 2.495 | 4.07× |
| 200 | HumanEval | 36.59 | [29.19, 43.98] | 72.35 | 3.302 | 3.47× |
| 200 | MATH500 | 31.60 | [27.52, 35.68] | 85.80 | 3.570 | 2.93× |
| 500 | GSM8K | 74.07 | [71.71, 76.44] | 60.77 | 2.438 | 4.17× |
| 500 | HumanEval | 37.20 | [29.78, 44.62] | 67.23 | 3.055 | 3.75× |
| 500 | MATH500 | 32.20 | [28.10, 36.30] | 85.16 | 3.525 | 2.97× |
| 1000 | GSM8K | 72.71 | [70.30, 75.11] | 58.80 | 2.377 | 4.27× |
| 1000 | HumanEval | 38.41 | [30.95, 45.88] | 68.51 | 3.118 | 3.67× |
| 1000 | MATH500 | 31.60 | [27.52, 35.68] | 83.83 | 3.489 | 3.00× |
| 2000 | GSM8K | 72.86 | [70.46, 75.26] | 58.92 | 2.380 | 4.27× |
| 2000 | HumanEval | 35.98 | [28.61, 43.34] | 68.62 | 3.105 | 3.69× |
| 2000 | MATH500 | 31.60 | [27.52, 35.68] | 83.56 | 3.492 | 3.00× |
| 5000 | GSM8K | 73.24 | [70.85, 75.63] | 57.17 | 2.308 | 4.40× |
| 5000 | HumanEval | 37.80 | [30.36, 45.25] | 67.13 | 3.067 | 3.73× |
| 5000 | MATH500 | 32.20 | [28.10, 36.30] | 83.10 | 3.445 | 3.04× |
