# High-frequency token coverage: 500 vs 5,000 Math traces

Candidate frequency is measured over all intermediate top-1 token records in the same 5,000 Standard-decoding Math traces used for calibration.

| Metric | Value |
|---|---:|
| Valid thresholds (500 / 5,000) | 410 / 1,453 |
| Shared valid token ids | 410 |
| 500 type recall of 5,000 table | 28.22% |
| Candidate mass covered by 500 table | 92.93% |
| Candidate mass covered by 5,000 table | 98.22% |
| 5,000-table candidate mass recovered by shared 500 tokens | 94.61% |

## Frequency-ranked coverage

| Top K token types | Candidate mass in Top K | 500 covered types | 5,000 covered types | 500 recall of 5,000 valid types | 500 recall of 5,000 valid mass |
|---:|---:|---:|---:|---:|---:|
| 10 | 45.09% | 10 | 10 | 100.00% | 100.00% |
| 25 | 63.70% | 25 | 25 | 100.00% | 100.00% |
| 50 | 75.58% | 50 | 50 | 100.00% | 100.00% |
| 100 | 83.20% | 100 | 100 | 100.00% | 100.00% |
| 200 | 88.92% | 200 | 200 | 100.00% | 100.00% |
| 500 | 94.65% | 375 | 499 | 75.15% | 97.97% |
| 1,000 | 97.40% | 403 | 974 | 41.38% | 95.51% |
| 2,000 | 99.05% | 410 | 1,453 | 28.22% | 94.61% |

## Shared-threshold stability

Among 410 shared token ids, mean absolute threshold difference is 0.103, median is 0.080, and 37.80% are within 0.05.

The 500-trace table should therefore be described as a compact high-frequency approximation, not as an identical reconstruction of the 5,000-trace table.
