# P1.2 Calibration-subset sampling robustness (sizes 100-5000, 3 seeds)

Each calibration size is re-sampled with three independent subset seeds
(2027 / 2028 / 2029) from the same pool of Math Standard-decoding traces,
recalibrated, and re-evaluated on all three main benchmarks at
`gen_length=256, steps=256, block_length=32, max/default threshold=0.90,
min threshold=0.01`.

`±` is the sample standard deviation across the three subset seeds
(`ddof=1`). `SpeedUp` is the per-task NFE ratio
`Standard NFE / setting NFE`. `Avg SpeedUp` is the ratio of the
*macro-averaged* NFE values -- NFE is averaged across the three
benchmarks first, and the speedup is the ratio of those two
averages -- not the mean of three separate per-task speedup ratios,
which would overweight a benchmark with a small NFE denominator.

| Setting | GSM8K Acc | GSM8K NFE | GSM8K s/it | GSM8K SpeedUp | MATH500 Acc | MATH500 NFE | MATH500 s/it | MATH500 SpeedUp | HumanEval Acc | HumanEval NFE | HumanEval s/it | HumanEval SpeedUp | Avg Acc | Avg NFE | Avg s/it | Avg SpeedUp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 73.16 | 256.00 | 10.158 | 1.00 | 31.40 | 256.00 | 10.469 | 1.00 | 37.20 | 256.00 | 11.454 | 1.00 | 47.25 | 256.00 | 10.694 | 1.00 |
| Fixed threshold 0.90 | 73.09 | 77.62 | 3.112 | 3.30 | 31.80 | 106.27 | 4.393 | 2.41 | 37.20 | 98.67 | 4.450 | 2.59 | 47.36 | 94.19 | 3.985 | 2.72 |
| Math-100 seed 2027 | 73.31 | 59.71 | 2.404 | 4.29 | 32.40 | 82.38 | 3.421 | 3.11 | 38.41 | 72.87 | 3.305 | 3.51 | 48.04 | 71.65 | 3.043 | 3.57 |
| Math-100 seed 2028 | 73.09 | 58.74 | 2.368 | 4.36 | 30.00 | 82.54 | 3.435 | 3.10 | 36.59 | 67.66 | 3.082 | 3.78 | 46.56 | 69.65 | 2.962 | 3.68 |
| Math-100 seed 2029 | 73.31 | 60.23 | 2.428 | 4.25 | 30.60 | 81.56 | 3.396 | 3.14 | 36.59 | 73.19 | 3.322 | 3.50 | 46.83 | 71.66 | 3.049 | 3.57 |
| Math-100 Mean ± Std | 73.24 ± 0.13 | 59.56 ± 0.75 | 2.400 ± 0.030 | 4.30 ± 0.05 | 31.00 ± 1.25 | 82.16 ± 0.53 | 3.417 ± 0.020 | 3.12 ± 0.02 | 37.20 ± 1.06 | 71.24 ± 3.10 | 3.236 ± 0.134 | 3.60 ± 0.16 | 47.14 ± 0.79 | 70.99 ± 1.16 | 3.018 ± 0.049 | 3.61 ± 0.06 |
| Math-200 seed 2027 | 73.39 | 61.79 | 2.495 | 4.14 | 31.60 | 85.80 | 3.570 | 2.98 | 36.59 | 72.35 | 3.302 | 3.54 | 47.19 | 73.31 | 3.122 | 3.49 |
| Math-200 seed 2028 | 72.40 | 59.35 | 2.389 | 4.31 | 30.40 | 82.49 | 3.427 | 3.10 | 35.98 | 69.36 | 3.147 | 3.69 | 46.26 | 70.40 | 2.988 | 3.64 |
| Math-200 seed 2029 | 72.86 | 60.73 | 2.443 | 4.22 | 32.00 | 82.54 | 3.445 | 3.10 | 35.37 | 68.30 | 3.106 | 3.75 | 46.74 | 70.52 | 2.998 | 3.63 |
| Math-200 Mean ± Std | 72.88 ± 0.49 | 60.62 ± 1.23 | 2.442 ± 0.053 | 4.22 ± 0.09 | 31.33 ± 0.83 | 83.61 ± 1.90 | 3.481 ± 0.078 | 3.06 ± 0.07 | 35.98 ± 0.61 | 70.00 ± 2.10 | 3.185 ± 0.103 | 3.66 ± 0.11 | 46.73 ± 0.47 | 71.41 ± 1.65 | 3.036 ± 0.075 | 3.59 ± 0.08 |
| Math-500 seed 2027 | 74.07 | 60.77 | 2.438 | 4.21 | 32.20 | 85.16 | 3.525 | 3.01 | 37.20 | 67.23 | 3.055 | 3.81 | 47.82 | 71.05 | 3.006 | 3.60 |
| Math-500 seed 2028 | 73.09 | 61.66 | 2.489 | 4.15 | 32.00 | 84.05 | 3.482 | 3.05 | 37.80 | 70.40 | 3.209 | 3.64 | 47.63 | 72.04 | 3.060 | 3.55 |
| Math-500 seed 2029 | 72.25 | 59.43 | 2.404 | 4.31 | 29.80 | 81.14 | 3.375 | 3.16 | 35.98 | 71.30 | 3.248 | 3.59 | 46.01 | 70.62 | 3.009 | 3.62 |
| Math-500 Mean ± Std | 73.14 ± 0.91 | 60.62 ± 1.12 | 2.444 ± 0.043 | 4.22 ± 0.08 | 31.33 ± 1.33 | 83.45 ± 2.08 | 3.461 ± 0.077 | 3.07 ± 0.08 | 36.99 ± 0.93 | 69.64 ± 2.14 | 3.170 ± 0.102 | 3.68 ± 0.11 | 47.15 ± 1.00 | 71.24 ± 0.72 | 3.025 ± 0.030 | 3.59 ± 0.04 |
| Math-1000 seed 2027 | 72.71 | 58.80 | 2.377 | 4.35 | 31.60 | 83.83 | 3.489 | 3.05 | 38.41 | 68.51 | 3.118 | 3.74 | 47.57 | 70.38 | 2.995 | 3.64 |
| Math-1000 seed 2028 | 73.01 | 58.45 | 2.365 | 4.38 | 31.40 | 83.72 | 3.471 | 3.06 | 37.20 | 68.42 | 3.133 | 3.74 | 47.20 | 70.20 | 2.990 | 3.65 |
| Math-1000 seed 2029 | 71.80 | 57.47 | 2.324 | 4.45 | 32.60 | 82.17 | 3.421 | 3.12 | 35.98 | 69.27 | 3.152 | 3.70 | 46.79 | 69.64 | 2.966 | 3.68 |
| Math-1000 Mean ± Std | 72.50 ± 0.63 | 58.24 ± 0.69 | 2.355 ± 0.028 | 4.40 ± 0.05 | 31.87 ± 0.64 | 83.24 ± 0.93 | 3.460 ± 0.035 | 3.08 ± 0.03 | 37.20 ± 1.22 | 68.73 ± 0.47 | 3.134 ± 0.017 | 3.72 ± 0.03 | 47.19 ± 0.39 | 70.07 ± 0.39 | 2.983 ± 0.016 | 3.65 ± 0.02 |
| Math-2000 seed 2027 | 72.86 | 58.92 | 2.380 | 4.34 | 31.60 | 83.56 | 3.492 | 3.06 | 35.98 | 68.62 | 3.105 | 3.73 | 46.81 | 70.37 | 2.992 | 3.64 |
| Math-2000 seed 2028 | 72.86 | 57.35 | 2.312 | 4.46 | 32.40 | 82.75 | 3.423 | 3.09 | 38.41 | 67.71 | 3.107 | 3.78 | 47.89 | 69.27 | 2.947 | 3.70 |
| Math-2000 seed 2029 | 72.55 | 57.63 | 2.377 | 4.44 | 32.00 | 82.83 | 3.462 | 3.09 | 36.59 | 68.34 | 3.101 | 3.75 | 47.05 | 69.60 | 2.980 | 3.68 |
| Math-2000 Mean ± Std | 72.76 ± 0.18 | 57.97 ± 0.84 | 2.356 ± 0.038 | 4.42 ± 0.06 | 32.00 ± 0.40 | 83.05 ± 0.45 | 3.459 ± 0.035 | 3.08 ± 0.02 | 36.99 ± 1.27 | 68.22 ± 0.47 | 3.104 ± 0.003 | 3.75 ± 0.03 | 47.25 ± 0.57 | 69.75 ± 0.56 | 2.973 ± 0.023 | 3.67 ± 0.03 |
| Math-5000 seed 2027 | 73.24 | 57.17 | 2.308 | 4.48 | 32.20 | 83.10 | 3.445 | 3.08 | 37.80 | 67.13 | 3.067 | 3.81 | 47.75 | 69.14 | 2.940 | 3.70 |
| Math-5000 seed 2028 | 73.24 | 57.17 | 2.327 | 4.48 | 32.20 | 83.10 | 3.456 | 3.08 | 37.80 | 67.13 | 3.074 | 3.81 | 47.75 | 69.14 | 2.952 | 3.70 |
| Math-5000 seed 2029 | 73.24 | 57.17 | 2.320 | 4.48 | 32.20 | 83.10 | 3.445 | 3.08 | 37.80 | 67.13 | 3.047 | 3.81 | 47.75 | 69.14 | 2.937 | 3.70 |
| Math-5000 Mean ± Std | 73.24 ± 0.00 | 57.17 ± 0.00 | 2.318 ± 0.010 | 4.48 ± 0.00 | 32.20 ± 0.00 | 83.10 ± 0.00 | 3.449 ± 0.007 | 3.08 ± 0.00 | 37.80 ± 0.00 | 67.13 ± 0.00 | 3.062 ± 0.014 | 3.81 ± 0.00 | 47.75 ± 0.00 | 69.14 ± 0.00 | 2.943 ± 0.008 | 3.70 ± 0.00 |

## Threshold-table stability across subset seeds

| Size | Valid thresholds (2027 / 2028 / 2029) | Union | Intersection | Jaccard | Pairwise mean \|Δτ\| | Pairwise within 0.05 |
|---:|---|---:|---:|---:|---:|---:|
| 100 | 141 / 127 / 142 | 191 | 90 | 47.12% | 0.110 | 26.2% |
| 200 | 235 / 221 / 230 | 314 | 153 | 48.73% | 0.119 | 27.8% |
| 500 | 410 / 408 / 439 | 550 | 301 | 54.73% | 0.104 | 32.5% |
| 1000 | 619 / 620 / 614 | 791 | 470 | 59.42% | 0.088 | 41.8% |
| 2000 | 910 / 916 / 911 | 1102 | 730 | 66.24% | 0.073 | 51.9% |
| 5000 | 1453 / 1453 / 1453 | 1453 | 1453 | 100.00% | 0.000 | 100.0% |

**Saturated sizes.** At calibration size 5000 the requested subset is as large as the available pool of Math Standard-decoding traces, so the three subset seeds select the same traces and produce byte-identical threshold tables. The rows for those sizes therefore report run-to-run wall-clock jitter under greedy decoding, not calibration-subset sampling variability, and must not be presented as evidence that larger calibration sets are insensitive to subset sampling.

## Interpretation boundary

Threshold membership and individual threshold values are sensitive to which
calibration subset is drawn, while aggregate NFE and wall-clock efficiency are
considerably more stable. Accuracy differences across subset seeds are within
the range expected from benchmark sampling noise at these sample sizes and
should not be read as a monotone effect of calibration size.

## Files

- `multiseed_full_table.csv`: the table above, including the `Mean ± Std` rows.
- `multiseed_mean_std_statistics.csv`: numeric means and sample standard deviations.
- `multiseed_per_run.csv`: one row per size/seed/benchmark with result and log paths.
- `multiseed_threshold_stability.csv`: per-size threshold membership and value agreement.
- `missing_runs.csv`: settings that were not complete at generation time (if any).
- `summary.json`: machine-readable dump.
