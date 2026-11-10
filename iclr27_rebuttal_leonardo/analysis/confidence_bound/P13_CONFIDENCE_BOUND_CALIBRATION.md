# P1.3 Confidence-bound calibration summary

A one-sided 95% Wilson lower bound must reach the target consistency 99.5% at the selected threshold.

| Statistic | Empirical rule | Confidence-bound rule |
|---|---:|---:|
| Eligible token types | 1,898 | 1,898 |
| Valid token thresholds | 1,453 | 491 |
| Selected accepted calibration records | 7,777,598 | 7,364,280 |
| Thresholds above 0.90 | — | 42 |

The conservative table retains 33.79% of the empirical table's valid token types. Among 491 common token types, 100.00% receive an equal or higher threshold.

Inference preserves conservative thresholds up to 1.0 instead of clipping them to 0.9. Tokens without a valid confidence-bound threshold retain the existing default fallback 0.9, so this experiment evaluates a conservative calibrated subset and is not a formal vocabulary-wide guarantee.
