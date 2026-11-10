# P2.2 Configuration and model-variant transfer of one fixed threshold table

Every row below reuses the single Math-5000 threshold table calibrated once on
LLaDA-8B-Instruct Standard traces at `gen_length=256, block_length=32`
(`rebuttal-size-ablation/thresholds/math_n5000_seed2027_token_threshold_grid_p99.5_mincount200_minaccepted100.json`). No setting below was recalibrated.

`SpeedUp` columns are ratios against the Standard run of the *same*
configuration or model variant, so each block is internally matched.

## Decoding-configuration transfer

### length 256, block 16

| Task | Setting | Acc (%) | Δ Acc vs Standard | Mean NFE | NFE speedup | s/it | Wall-clock speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K | Standard | 73.39 | --- | 256.00 | 1.00x | 10.155 | 1.00x |
| GSM8K | Fixed APD 0.90 | 73.54 | +0.15 | 83.87 | 3.05x | 3.357 | 3.03x |
| GSM8K | TC-APD Math-5000 | 72.86 | -0.53 | 64.63 | 3.96x | 2.592 | 3.92x |
| MATH500 | Standard | 31.40 | --- | 256.00 | 1.00x | 10.385 | 1.00x |
| MATH500 | Fixed APD 0.90 | 31.20 | -0.20 | 111.15 | 2.30x | 4.559 | 2.28x |
| MATH500 | TC-APD Math-5000 | 31.20 | -0.20 | 89.56 | 2.86x | 3.727 | 2.79x |
| HumanEval | Standard | 38.41 | --- | 256.00 | 1.00x | 11.395 | 1.00x |
| HumanEval | Fixed APD 0.90 | 38.41 | +0.00 | 108.93 | 2.35x | 4.880 | 2.34x |
| HumanEval | TC-APD Math-5000 | 37.80 | -0.61 | 78.68 | 3.25x | 3.582 | 3.18x |

### length 256, block 32 + chat template

| Task | Setting | Acc (%) | Δ Acc vs Standard | Mean NFE | NFE speedup | s/it | Wall-clock speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K | Standard | 70.05 | --- | 256.00 | 1.00x | 10.289 | 1.00x |
| GSM8K | Fixed APD 0.90 | 69.90 | -0.15 | 72.75 | 3.52x | 2.960 | 3.48x |
| GSM8K | TC-APD Math-5000 | 70.20 | +0.15 | 53.77 | 4.76x | 2.200 | 4.68x |
| MATH500 | Standard | 33.20 | --- | 256.00 | 1.00x | 10.641 | 1.00x |
| MATH500 | Fixed APD 0.90 | 33.40 | +0.20 | 106.58 | 2.40x | 4.473 | 2.38x |
| MATH500 | TC-APD Math-5000 | 35.80 | +2.60 | 83.13 | 3.08x | 3.499 | 3.04x |
| HumanEval | Standard | 8.54 | --- | 256.00 | 1.00x | 11.853 | 1.00x |
| HumanEval | Fixed APD 0.90 | 8.54 | +0.00 | 79.46 | 3.22x | 3.723 | 3.18x |
| HumanEval | TC-APD Math-5000 | 6.10 | -2.44 | 61.54 | 4.16x | 2.875 | 4.12x |

### length 256, block 8

| Task | Setting | Acc (%) | Δ Acc vs Standard | Mean NFE | NFE speedup | s/it | Wall-clock speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K | Standard | 72.33 | --- | 256.00 | 1.00x | 10.077 | 1.00x |
| GSM8K | Fixed APD 0.90 | 72.78 | +0.45 | 95.16 | 2.69x | 3.791 | 2.66x |
| GSM8K | TC-APD Math-5000 | 73.46 | +1.14 | 78.52 | 3.26x | 3.110 | 3.24x |
| MATH500 | Standard | 30.80 | --- | 256.00 | 1.00x | 10.464 | 1.00x |
| MATH500 | Fixed APD 0.90 | 31.00 | +0.20 | 121.35 | 2.11x | 4.984 | 2.10x |
| MATH500 | TC-APD Math-5000 | 30.60 | -0.20 | 102.08 | 2.51x | 4.186 | 2.50x |
| HumanEval | Standard | 37.80 | --- | 256.00 | 1.00x | 11.524 | 1.00x |
| HumanEval | Fixed APD 0.90 | 39.02 | +1.22 | 124.64 | 2.05x | 5.579 | 2.07x |
| HumanEval | TC-APD Math-5000 | 38.41 | +0.61 | 97.82 | 2.62x | 4.384 | 2.63x |

### length 512, block 32

| Task | Setting | Acc (%) | Δ Acc vs Standard | Mean NFE | NFE speedup | s/it | Wall-clock speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K | Fixed APD 0.90 | 75.21 | n/a | 101.50 | n/a | 5.736 | n/a |
| GSM8K | TC-APD Math-5000 | 75.89 | n/a | 74.43 | n/a | 4.199 | n/a |
| MATH500 | Standard | 36.20 | --- | 512.00 | 1.00x | 28.903 | 1.00x |
| MATH500 | Fixed APD 0.90 | 36.60 | +0.40 | 155.57 | 3.29x | 8.966 | 3.22x |
| MATH500 | TC-APD Math-5000 | 37.40 | +1.20 | 121.82 | 4.20x | 6.987 | 4.14x |
| HumanEval | Standard | 34.15 | --- | 512.00 | 1.00x | 32.086 | 1.00x |
| HumanEval | Fixed APD 0.90 | 34.15 | +0.00 | 156.30 | 3.28x | 9.849 | 3.26x |
| HumanEval | TC-APD Math-5000 | 34.15 | +0.00 | 104.32 | 4.91x | 6.612 | 4.85x |

Not run in this configuration (cancelled by the authors, deliberately not resubmitted): GSM8K Standard. The affected accelerated rows therefore have no same-configuration Standard reference and their speedup cells are left empty rather than compared against a different configuration.

## Model-variant transfer

### Fine-tuned (JustGRPO-GSM8K)

Threshold table unchanged; tokenizer unchanged.

| Task | Setting | Examples | Acc (%) | Δ Acc vs Standard | Mean NFE | NFE speedup | s/it | Wall-clock speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | Standard | 1319 | 87.41 | --- | 256.00 | 1.00x | 10.247 | 1.00x |
| GSM8K | Fixed APD 0.90 | 1319 | 84.76 | -2.65 | 41.28 | 6.20x | 1.664 | 6.16x |
| GSM8K | TC-APD Math-5000 | 1319 | 83.17 | -4.25 | 33.34 | 7.68x | 1.357 | 7.55x |
| MATH500 | Standard | 500 | 37.40 | --- | 256.00 | 1.00x | 10.417 | 1.00x |
| MATH500 | Fixed APD 0.90 | 500 | 32.20 | -5.20 | 72.30 | 3.54x | 3.024 | 3.45x |
| MATH500 | TC-APD Math-5000 | 500 | 31.00 | -6.40 | 56.20 | 4.56x | 2.352 | 4.43x |
| HumanEval | Standard | 164 | 42.68 | --- | 256.00 | 1.00x | 11.479 | 1.00x |
| HumanEval | Fixed APD 0.90 | 164 | 41.46 | -1.22 | 71.84 | 3.56x | 3.278 | 3.50x |
| HumanEval | TC-APD Math-5000 | 164 | 40.24 | -2.44 | 48.30 | 5.30x | 2.191 | 5.24x |

### INT8 GPTQ

Threshold table unchanged; tokenizer unchanged. Evaluated on reduced example budgets (164, 300) because this variant decodes very slowly.

| Task | Setting | Examples | Acc (%) | Δ Acc vs Standard | Mean NFE | NFE speedup | s/it | Wall-clock speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | Standard | 300 | 73.67 | --- | 256.00 | 1.00x | 70.154 | 1.00x |
| GSM8K | Fixed APD 0.90 | 300 | 75.67 | +2.00 | 77.39 | 3.31x | 21.220 | 3.31x |
| GSM8K | TC-APD Math-5000 | 300 | 76.33 | +2.67 | 56.21 | 4.55x | 15.421 | 4.55x |
| MATH500 | Standard | 300 | 33.33 | --- | 256.00 | 1.00x | 70.511 | 1.00x |
| MATH500 | Fixed APD 0.90 | 300 | 33.67 | +0.33 | 102.37 | 2.50x | 28.211 | 2.50x |
| MATH500 | TC-APD Math-5000 | 300 | 36.00 | +2.67 | 79.03 | 3.24x | 21.807 | 3.23x |
| HumanEval | Standard | 164 | 42.07 | --- | 256.00 | 1.00x | 71.228 | 1.00x |
| HumanEval | Fixed APD 0.90 | 164 | 42.68 | +0.61 | 92.51 | 2.77x | 25.757 | 2.77x |
| HumanEval | TC-APD Math-5000 | 164 | 38.41 | -3.66 | 60.27 | 4.25x | 16.817 | 4.24x |

## Settings that need qualification

**HumanEval under the chat template.** All three decoders collapse to single-digit
pass@1 in that configuration (Standard 8.54%), so the LLaDA chat template breaks
the HumanEval completion format itself. That block measures the prompting format,
not threshold transfer, and should not be read as a TC-APD failure or success.

**Fine-tuned checkpoint.** Reusing the base-model table on the GRPO fine-tuned
checkpoint still accelerates decoding, but accuracy drops relative to Standard on
all three tasks for both accelerated decoders, and more for TC-APD than for the
fixed threshold. Changing model weights therefore does shift the confidence
distribution the table was calibrated on, and recalibration is advisable after
fine-tuning.

## Interpretation boundary

These results show that an existing token-level table keeps producing a
speedup under changed block sizes, a longer generation budget, a chat
prompting template, GPTQ INT8 weights, and a fine-tuned checkpoint that
shares the base tokenizer. They do not show that recalibration is never
needed: accuracy does move under the model variants, and nothing here
transfers across a different backbone or a different tokenizer.

## Files

- `configuration_transfer_table.csv`: per configuration, task, and decoder.
- `model_variant_table.csv`: fine-tuned and INT8 variants (smoke runs excluded).
- `missing_runs.csv`: result directories without a complete aggregate or log.
- `summary.json`: machine-readable dump of both families.
