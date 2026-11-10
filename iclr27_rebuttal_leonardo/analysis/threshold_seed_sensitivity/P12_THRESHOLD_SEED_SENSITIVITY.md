# P1.2 Threshold sensitivity across 100-trace subsets

Three independently sampled 100-of-5000 Math Standard-trace subsets were calibrated with seeds 2027, 2028, and 2029.

## Table size and overlap

| Seed | Valid thresholds | Eligible tokens | Valid / eligible | Coverage of union |
|---:|---:|---:|---:|---:|
| 2027 | 141 | 186 | 75.81% | 73.82% |
| 2028 | 127 | 173 | 73.41% | 66.49% |
| 2029 | 142 | 187 | 75.94% | 74.35% |

Union: **191** token ids; all-three intersection: **90** (47.12% of union).

## Pairwise threshold sensitivity

| Seeds | Shared | Jaccard | Exact agreement | Within 0.05 | Mean |Δτ| | P90 |Δτ| | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2027/2028 | 101 | 0.605 | 2.97% | 31.68% | 0.105 | 0.230 | 0.483 |
| 2027/2029 | 106 | 0.599 | 0.00% | 30.19% | 0.113 | 0.235 | 0.461 |
| 2028/2029 | 102 | 0.611 | 2.94% | 23.53% | 0.112 | 0.214 | 0.299 |

## Main overlapping tokens

The 100 highest-support tokens shared by all seeds are in `top100_common_tokens.csv`, including decoded token text, per-seed thresholds, and support. Full membership and seed-unique token lists are provided as CSV files.

## Downstream evaluation

All evaluations use generation length 256, block size 32, default/max threshold 0.90, and one evaluation run per independently calibrated table.

| Calibration seed | GSM8K Acc. | GSM8K NFE | GSM8K s/it | MATH500 Acc. | MATH500 NFE | MATH500 s/it | HumanEval Acc. | HumanEval NFE | HumanEval s/it |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2027 | 73.31 | 59.71 | 2.404 | 32.40 | 82.38 | 3.421 | 38.41 | 72.87 | 3.305 |
| 2028 | 73.09 | 58.74 | 2.368 | 30.00 | 82.54 | 3.435 | 36.59 | 67.66 | 3.082 |
| 2029 | 73.31 | 60.23 | 2.428 | 30.60 | 81.56 | 3.396 | 36.59 | 73.19 | 3.322 |
| Mean +/- SD | 73.24 +/- 0.13 | 59.56 +/- 0.75 | 2.400 +/- 0.030 | 31.00 +/- 1.25 | 82.16 +/- 0.53 | 3.417 +/- 0.020 | 37.20 +/- 1.06 | 71.24 +/- 3.10 | 3.236 +/- 0.134 |

Accuracy is reported in percentage points. SD is the sample standard deviation across the three calibration subset seeds. Detailed values are in `downstream_seed_sensitivity.csv`.

## Interpretation

At 100 calibration trajectories, individual lookup-table membership and threshold values are sampling-sensitive: only 90 of 191 union token ids occur in every table, and exact threshold agreement is rare. However, aggregate decoding cost is much more stable on GSM8K and MATH500 (NFE SD below 1), while HumanEval has larger NFE variation (SD 3.10). Accuracy is essentially unchanged on GSM8K, but varies by 2.4 points on MATH500 and 1.83 points on HumanEval. Therefore, 100 trajectories are sufficient for a strong low-cost result, but not for claiming a fully converged token-level table.
