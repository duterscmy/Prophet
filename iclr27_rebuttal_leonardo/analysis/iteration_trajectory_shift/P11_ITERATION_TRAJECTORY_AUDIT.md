# P1.1 Iteration-level Standard → TC-APD trajectory audit

Paired analysis over 500 GSM8K prompts with full 256-token pre-stop trajectories.

| Population | Observations | Agreement with paired Standard final token |
|---|---:|---:|
| Math-5000 policy-eligible candidates on Standard states | 963,585 | 99.063% |
| — calibrated tokens below the 0.90 cap | 864,199 | 98.974% |
| — calibrated tokens clipped at the 0.90 cap | 1,644 | 99.939% |
| Math-5000 policy-eligible candidates on TC-APD states | 119,370 | 62.420% |
| Actual TC-APD selected commits | 128,000 | 62.954% |

The policy-eligible consistency shift from Standard to accelerated states is **-36.643 percentage points**.
Final position-wise agreement is **62.954%**; exact full-256-token sequence agreement is **20.600%**.
Mean TC-APD NFE is **57.224**.

## Interpretation boundary

TC-APD commits are irreversible, so agreement of a selected token with TC-APD's own final token would be 100% by construction. The non-trivial deployment diagnostic therefore compares accelerated-state candidates and commits with the paired Standard final token at the same generation position. This measures trajectory distribution shift; it is not a formal guarantee of downstream correctness.

Two further boundaries apply. First, accelerated-state agreement is measured against a Standard reference that conditions on a different prefix, so the divergence compounds along the trajectory: once the two runs disagree at one position, the Standard final token at later positions is no longer the token a correctly behaving accelerated run should produce. The block-wise table therefore reports accumulated divergence, not a per-block error rate. Second, realized consistency on these Standard states is below the nominal 99.5% empirical target, and the clipped/uncapped split shows that the deployed max_threshold = 0.90 cap is not the reason: capping touches only a small fraction of policy-eligible candidates and those agree at a higher rate than the uncapped calibrated tokens. The residual gap is better attributed to the domain shift between the Math calibration traces and these GSM8K evaluation states, which is consistent with the in-domain held-out validation reaching a higher realized consistency on unseen Math traces.
