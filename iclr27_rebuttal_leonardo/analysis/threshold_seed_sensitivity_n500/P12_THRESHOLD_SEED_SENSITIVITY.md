# P1.2 Threshold sensitivity across 500-trace subsets

Three independently sampled 500-of-5000 Math Standard-trace subsets were calibrated with seeds 2027, 2028, and 2029.

## Table size and overlap

| Seed | Valid thresholds | Eligible tokens | Valid / eligible | Coverage of union |
|---:|---:|---:|---:|---:|
| 2027 | 410 | 535 | 76.64% | 74.55% |
| 2028 | 408 | 528 | 77.27% | 74.18% |
| 2029 | 439 | 541 | 81.15% | 79.82% |

Union: **550** token ids; all-three intersection: **301** (54.73% of union).

## Pairwise threshold sensitivity

| Seeds | Shared | Jaccard | Exact agreement | Within 0.05 | Mean |Δτ| | P90 |Δτ| | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2027/2028 | 324 | 0.656 | 5.56% | 36.73% | 0.097 | 0.228 | 0.477 |
| 2027/2029 | 342 | 0.675 | 6.73% | 33.04% | 0.110 | 0.240 | 0.371 |
| 2028/2029 | 342 | 0.677 | 4.97% | 33.63% | 0.106 | 0.229 | 0.420 |

## Main overlapping tokens

The 100 highest-support tokens shared by all seeds are in `top100_common_tokens.csv`, including decoded token text, per-seed thresholds, and support. Full membership and seed-unique token lists are provided as CSV files.

## Interpretation

This analysis measures sampling sensitivity of lookup-table support and selected thresholds. Downstream accuracy/NFE sensitivity is evaluated separately on GSM8K, MATH500, and HumanEval.
