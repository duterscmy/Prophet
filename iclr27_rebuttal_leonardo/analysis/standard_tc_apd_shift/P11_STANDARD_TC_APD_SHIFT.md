# P1.1 Standard → TC-APD trajectory distribution shift

Paired GSM8K analysis over **1,319** examples. The primary answer metric below is lm-eval's `flexible-extract`; strict-match is retained in the CSVs.

| Setting | Final-answer agreement¹ | C→I² | I→C³ | Acc. Δ | EOS-aware token agreement⁴ | Exact sequence | Sequence disagreement | Mean edit similarity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_0.90 | 94.01% (1319/1319 pairs) | 13 (1.35%) | 12 (3.39%) | -0.08 pp | 94.33% | 78.85% | 21.15% | 0.987 |
| tc_n100 | 75.74% (1319/1319 pairs) | 50 (5.18%) | 52 (14.69%) | +0.15 pp | 60.19% | 19.26% | 80.74% | 0.905 |
| tc_n200 | 77.71% (1319/1319 pairs) | 55 (5.70%) | 58 (16.38%) | +0.23 pp | 64.26% | 21.91% | 78.09% | 0.909 |
| tc_n500 | 76.19% (1319/1319 pairs) | 46 (4.77%) | 58 (16.38%) | +0.91 pp | 62.11% | 22.21% | 77.79% | 0.903 |
| tc_n1000 | 75.13% (1319/1319 pairs) | 68 (7.05%) | 62 (17.51%) | -0.45 pp | 57.48% | 19.79% | 80.21% | 0.880 |
| tc_n2000 | 76.80% (1319/1319 pairs) | 59 (6.11%) | 55 (15.54%) | -0.30 pp | 62.21% | 23.20% | 76.80% | 0.899 |
| tc_n5000 | 76.27% (1319/1319 pairs) | 56 (5.80%) | 57 (16.10%) | +0.08 pp | 60.54% | 20.17% | 79.83% | 0.892 |

¹ Agreement is conditional on both runs yielding an extractable answer; pair coverage is shown in parentheses. This avoids counting two extraction failures as agreement.

² Percentage denominator is the 965 examples Standard gets correct. ³ Percentage denominator is the 354 examples Standard gets incorrect. Accuracy Δ equals `(I→C − C→I) / 1319`.

⁴ lm-eval responses are already truncated at the configured EOS/EOT/stop strings. Position-wise token agreement uses the Standard effective length as denominator; missing accelerated positions are mismatches.

For the main `tc_n5000` setting, 79.83% of final token sequences differ from Standard, yet flexible-answer agreement is 76.27%, binary correctness agrees on 91.43% of examples, and the correctness flips nearly balance (56 C→I versus 57 I→C; +0.08 pp accuracy). Thus TC-APD clearly induces a decoding-path distribution shift, but sequence reproduction is not its objective and sequence disagreement does not directly imply quality loss.

## Interpretation boundary

The available lm-eval samples contain final stop-truncated generations, filtered answers, and correctness, but not per-iteration mask/commit states. Therefore `Exact sequence`, `Sequence disagreement`, prefix, and edit metrics measure disagreement of the realized final generation sequence. They demonstrate decoding-path distribution shift, but are not a direct measurement of internal denoising-trajectory agreement. A true iteration-level trajectory statistic requires rerunning with `selected_records/all_token_records` logging enabled.

## Files

- `final_answer_agreement.csv`: strict and flexible answer agreement plus extraction coverage.
- `answer_transitions.csv`: all four correctness transitions and conditional flip rates.
- `sequence_agreement.csv`: EOS-aware sequence/prefix/edit diagnostics.
- `per_sample_sequence_metrics.csv`: paired sample-level sequence diagnostics.
