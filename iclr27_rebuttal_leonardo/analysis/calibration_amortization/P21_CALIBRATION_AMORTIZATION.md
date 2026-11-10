# P2.1 Offline calibration cost and amortization

## Cost model

All primary costs are normalized to **one NVIDIA A100 on Leonardo**, so numerator and denominator use the same hardware and wall-clock unit. Math calibration traces use Standard decoding with `gen_length=256`, `steps=256`, and `block_length=32`. The measured MATH500 Standard cost is **10.469 GPU-seconds per trace**.

For benchmark \(b\) and calibration size \(m\):

\[C_{cal}(m)=m\,t_{Math,Standard},\qquad N_b^*(m)=\frac{C_{cal}(m)}{t_{b,Standard}-t_{b,TC-APD(m)}}.\]

| Calibration trajectories | Calibration A100 GPU-hours | GSM8K break-even | MATH500 break-even | HumanEval break-even | Mean break-even |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.291 | 136 | 149 | 129 | 138 |
| 200 | 0.582 | 274 | 304 | 257 | 278 |
| 500 | 1.454 | 679 | 754 | 624 | 686 |
| 1,000 | 2.908 | 1,346 | 1,500 | 1,256 | 1,368 |
| 2,000 | 5.816 | 2,693 | 3,002 | 2,508 | 2,734 |
| 5,000 | 14.541 | 6,669 | 7,453 | 6,242 | 6,788 |

Break-even values are rounded up to the first whole deployed sequence that repays the offline GPU cost. For example, the 100-trace table costs only 0.291 A100 GPU-hours and amortizes after roughly 128–149 generations across the three tasks. The 5,000-trace table costs 14.541 A100 GPU-hours and needs about 6241–7453 generations.

## Threshold-table construction overhead

Generating all six threshold tables from the saved traces took **124 seconds on 8 CPU cores** (0.276 CPU-hours) and **0 GPU-hours**. This is negligible relative to trajectory collection and is excluded from the GPU break-even numerator.

The six rows are alternative calibration budgets, not additive costs. A single 5,000-trace collection can be subsampled to produce every smaller table; its total collection cost remains the 5,000-row cost rather than the sum of all rows.
If all six datasets were collected independently, their normalized costs would sum to 25.591 A100 GPU-hours, but that is not how this ablation was constructed.

## Conservative source-log cross-check

The uploaded 5,000-trace source run spans approximately **15.622 single-GPU hours** end to end, including model setup and task-side postprocessing. That is 1.074× the Leonardo-normalized generation estimate. Because the original log does not record an unambiguous GPU model and includes non-generation overhead, it is not mixed into the primary formula. Multiplying the reported break-even counts by 1.074 gives a conservative end-to-end estimate.

## When recalibration is needed

Recalibration should be performed when changing model weights/backbone, tokenizer or vocabulary, numerical execution such as quantization, the confidence/commit policy, generation length, step count, or block length. Prompt/template or deployment-domain changes may reuse a table, but should first be checked with held-out consistency and token-coverage diagnostics; a material drop calls for recalibration. Merely increasing the number of deployed generations does not require recalibration.

## Rebuttal-ready takeaway

Even with conservative accounting, calibration is a one-time cost repaid after hundreds of generations for a 100-trace table and several thousand for a 5,000-trace table. Since the 100-trace table already recovers nearly all of the measured NFE reduction, it is the most attractive operating point for short-lived deployments; larger tables become economical for sustained serving workloads.
