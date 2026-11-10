# P0.4 Independent held-out consistency validation

Math Standard traces were split once with seed 2027: 4,000 traces for threshold estimation and 1,000 disjoint traces for validation.

| Validation policy | Accepted records | Realized consistency | 95% Wilson CI |
|---|---:|---:|---:|
| Raw selected thresholds | 1,545,621 | 99.443% | [99.431%, 99.454%] |
| Runtime-clipped thresholds | 1,546,522 | 99.442% | [99.430%, 99.453%] |
| Full lookup policy | 1,565,252 | 99.447% | [99.436%, 99.459%] |

## Token-level diagnostics

- Estimated valid token thresholds: 1,290.
- Tokens with nonzero held-out accepted support: 1,233.
- Tokens meeting empirical 99.5% consistency on held-out data: 70.97% (raw thresholds).
- Tokens whose 95% Wilson lower bound reaches 99.5%: 3.08%.
- Runtime clipping changes 41 token thresholds.

## Interpretation boundary

This validates the offline consistency estimator on independent Standard-decoding traces. It does not establish consistency under TC-APD-generated trajectories and must not be presented as a formal deployment-time guarantee.
