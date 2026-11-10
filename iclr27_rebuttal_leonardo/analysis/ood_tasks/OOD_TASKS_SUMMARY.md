# P0.2 / P2.3 Out-of-domain evaluation

All runs use the LLaDA-8B-Instruct Math calibration tables produced for the
P0.1 size ablation. No task below was recalibrated, and none of these tasks
participated in the original calibration-source selection.

## IFEval

lm-eval task `ifeval_leonardo`, 541 examples, generation length 256, block length 32.

| Setting | Prompt-level strict acc. (%) | Δ vs Standard | Mean NFE | NFE speedup | Mean decode time (s) | Wall-clock speedup |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 57.49 | --- | 256.0 | 1.00x | 10.07 | 1.00x |
| Fixed APD 0.90 | 57.49 | +0.00 | 137.5 | 1.86x | 5.46 | 1.84x |
| TC-APD Math-100 | 57.12 | -0.37 | 128.1 | 2.00x | 5.10 | 1.97x |
| TC-APD Math-200 | 57.49 | +0.00 | 128.7 | 1.99x | 5.10 | 1.97x |
| TC-APD Math-500 | 57.67 | +0.18 | 126.7 | 2.02x | 5.04 | 2.00x |
| TC-APD Math-1000 | 57.30 | -0.18 | 126.3 | 2.03x | 5.04 | 2.00x |
| TC-APD Math-2000 | 56.93 | -0.55 | 126.1 | 2.03x | 5.01 | 2.01x |
| TC-APD Math-5000 | 57.86 | +0.37 | 124.6 | 2.05x | 4.98 | 2.02x |

Secondary metrics: `inst_level_strict_acc` 0.6679/0.6703, `prompt_level_loose_acc` 0.6118/0.6137, `inst_level_loose_acc` 0.7014/0.7002 (Standard / TC-APD Math-5000).

## ARC-Challenge

lm-eval task `arc_challenge_chat`, 1,172 examples, generation length 256, block length 32.

| Setting | Exact match (%) | Δ vs Standard | Mean NFE | NFE speedup | Mean decode time (s) | Wall-clock speedup |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 84.64 | --- | 256.0 | 1.00x | 11.93 | 1.00x |
| Fixed APD 0.90 | 84.64 | +0.00 | 13.6 | 18.86x | 0.65 | 18.47x |
| TC-APD Math-100 | 66.04 | -18.60 | 10.3 | 24.77x | 0.49 | 24.30x |
| TC-APD Math-200 | 83.79 | -0.85 | 10.3 | 24.93x | 0.50 | 24.04x |
| TC-APD Math-500 | 84.73 | +0.09 | 10.1 | 25.41x | 0.48 | 24.80x |
| TC-APD Math-1000 | 84.04 | -0.60 | 10.0 | 25.62x | 0.48 | 25.03x |
| TC-APD Math-2000 | 84.56 | -0.09 | 10.1 | 25.31x | 0.48 | 24.73x |
| TC-APD Math-5000 | 84.56 | -0.09 | 10.1 | 25.30x | 0.48 | 24.66x |

## XSum-500

lm-eval task `xsum_500_leonardo`, 500 examples, generation length 256, block length 32.

| Setting | ROUGE-L (%) | Δ vs Standard | Mean NFE | NFE speedup | Mean decode time (s) | Wall-clock speedup |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 21.84 | --- | 256.0 | 1.00x | 20.01 | 1.00x |
| Fixed APD 0.90 | 21.83 | -0.01 | 46.8 | 5.48x | 3.93 | 5.09x |
| TC-APD Math-100 | 21.91 | +0.07 | 37.0 | 6.91x | 3.18 | 6.29x |
| TC-APD Math-200 | 22.08 | +0.25 | 36.7 | 6.98x | 3.17 | 6.32x |
| TC-APD Math-500 | 22.11 | +0.27 | 36.0 | 7.12x | 3.11 | 6.44x |
| TC-APD Math-1000 | 22.05 | +0.21 | 35.9 | 7.14x | 3.09 | 6.47x |
| TC-APD Math-2000 | 22.01 | +0.17 | 36.2 | 7.08x | 3.07 | 6.51x |
| TC-APD Math-5000 | 21.99 | +0.15 | 35.8 | 7.15x | 3.05 | 6.55x |

Secondary metrics: `rouge1` 0.2995/0.2988, `rouge2` 0.0880/0.0898 (Standard / TC-APD Math-5000).

## WMT16 en-ro

lm-eval task `wmt16_en_ro_leonardo`, 1,999 examples, generation length 256, block length 32.

| Setting | BLEU | Δ vs Standard | Mean NFE | NFE speedup | Mean decode time (s) | Wall-clock speedup |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 14.38 | --- | 256.0 | 1.00x | 9.99 | 1.00x |
| Fixed APD 0.90 | 14.26 | -0.12 | 53.3 | 4.80x | 2.10 | 4.75x |
| TC-APD Math-100 | 12.62 | -1.76 | 45.6 | 5.62x | 1.80 | 5.55x |
| TC-APD Math-200 | 12.29 | -2.09 | 45.3 | 5.65x | 1.80 | 5.55x |
| TC-APD Math-500 | 12.34 | -2.04 | 44.9 | 5.71x | 1.76 | 5.67x |
| TC-APD Math-1000 | 12.49 | -1.89 | 44.9 | 5.70x | 1.77 | 5.64x |
| TC-APD Math-2000 | 12.63 | -1.75 | 45.2 | 5.66x | 1.79 | 5.57x |
| TC-APD Math-5000 | 12.80 | -1.58 | 45.1 | 5.67x | 1.78 | 5.62x |

Secondary metrics: `chrf` 41.3683/38.5261, `ter` 72.7256/74.7832 (Standard / TC-APD Math-5000).

## WritingPrompts-100 (len 1024)

lm-eval task `writingprompts_100_leonardo`, 100 examples, generation length 1024, block length 32.

| Setting | 4-gram repetition (lower is better) | Δ vs Standard | Mean NFE | NFE speedup | Mean decode time (s) | Wall-clock speedup |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 0.29 | --- | 1024.0 | 1.00x | 110.23 | 1.00x |
| Fixed APD 0.90 | 0.29 | +0.00 | 725.4 | 1.41x | 79.61 | 1.38x |
| TC-APD Math-100 | 0.30 | +0.01 | 692.8 | 1.48x | 75.08 | 1.47x |
| TC-APD Math-200 | 0.30 | +0.01 | 694.3 | 1.47x | 75.25 | 1.46x |
| TC-APD Math-500 | 0.32 | +0.03 | 683.9 | 1.50x | 74.23 | 1.49x |
| TC-APD Math-1000 | 0.30 | +0.01 | 684.0 | 1.50x | 74.54 | 1.48x |
| TC-APD Math-2000 | 0.31 | +0.02 | 679.7 | 1.51x | 74.83 | 1.47x |
| TC-APD Math-5000 | 0.29 | +0.00 | 681.3 | 1.50x | 74.44 | 1.48x |

Secondary metrics: `word_count` 807.6300/810.8800, `distinct_4` 0.7129/0.7093, `repetition_8` 0.2365/0.2429, `sentence_repetition` 0.2080/0.2247 (Standard / TC-APD Math-5000).

## Generation-length probe

Fixed-threshold probes at a longer generation budget, used to check that
the 256-token budget is not the binding constraint on these tasks.

| Task | Setting | Gen length | Primary metric | Mean NFE | Mean decode time (s) |
|---|---|---:|---:|---:|---:|
| IFEval | Fixed APD 0.90 | 512 | 58.60 | 203.0 | 11.27 |

## Interpretation boundary

Speedups are computed against the Standard run at the same generation length
and are means over per-example decoder statistics, not paired confidence
intervals. XSum ROUGE, WMT BLEU/chrF/TER, and the WritingPrompts repetition
statistics are automatic metrics; the WritingPrompts numbers measure
degeneration, not human-judged writing quality.

## Files

- `ood_main_table.csv`: compact Standard / Fixed 0.90 / Math-500 / Math-5000 table.
- `ood_calibration_size_sweep.csv`: calibration-size sweep per task.
- `ood_metrics_long.csv`: every reported metric with lm-eval standard errors.
- `ood_efficiency.csv`: NFE, decode time, and speedups for every run.
- `ood_generation_length_probe.csv`: longer-generation probe runs.
- `summary.json`: machine-readable dump of all joined rows.
