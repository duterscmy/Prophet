# Author notes for `iclr2027_rebuttal_responses.md` — do not post

Items that need an author decision before the reply goes on OpenReview.

## 1. Reviewer jgET question numbering

Our two sets of notes disagree about jgET's last two questions. The earlier
`iclr2027_review_rebuttal.md` records them as *Q3 = combination with online
dynamic threshold methods* and *Q4 = long-context / creative trade-off*; the
newer grouping assumes the opposite. The reply therefore labels these blocks
"W3 & Q3" and "Q4" following the newer grouping, but the headings also name the
topic, so either numbering reads correctly. **Please check the forum once and fix
the two numbers if needed** — a wrong question number is the kind of detail a
reviewer notices.

More generally, consider heading each block with the topic plus a short quote of
the reviewer's own wording rather than relying on W/Q numbers.

## 2. Corrected number in the calibration-size table

The draft table had HumanEval = 37.8 and Avg = 47.42 in the **2,000**-trajectory
row. Both the size-ablation run (seed 2027) and the handoff table give
HumanEval = **35.98**, so the macro average is **46.81**, not 47.42. The 37.8
appears to have been copied from the 5,000 row. The reply uses the corrected
values. Source: `iclr27_rebuttal_leonardo/analysis/multisize_seed_sensitivity/multiseed_per_run.csv`.

`SpeedUp` in that table is kept at the draft's definition, Standard macro NFE /
setting macro NFE (256 / Avg NFE), which reproduces the draft column exactly. Note
that `analysis/multisize_seed_sensitivity/` uses the other convention (mean of
per-task ratios, e.g. 3.61 instead of 3.57 at size 100). Both are stated in their
own documents; just do not mix them inside one table.

## 3. Calibration-source disclosure needs your confirmation

The reply to 4QWr W1 & Q3 states that the Minerva-Math source for LLaDA and the
mixed-domain source for Dream were chosen **after** exploratory comparisons on
the final evaluation benchmarks, and that calibration uses training-split
trajectories only. This follows the handoff instruction to disclose
test-informed selection honestly, but only you can confirm it is what actually
happened. **Please verify both sentences before posting**; if the choice was in
fact made before looking at benchmark results, this paragraph should be rewritten
rather than softened.

## 4. SOAR reference

The answer to jgET Q4 says the paper "already includes a plug-and-play TC-APD +
SOAR composition". The repository has `eval_gsm8k_dynamic_threshold_soar.from_math.sh`
and the README lists a TC-APD + SOAR row, but there is no verified result table
in `iclr27_rebuttal_leonardo/analysis/`. **Add the paper's section/table pointer**
(e.g. "Table 6" / "Appendix C") so the reviewer can find it, or soften to "is
straightforward to compose with" if the paper does not report it.

## 5. Multi-seed table to be extended

The reply reports three-subset `Mean ± Std` for calibration sizes **100 and 500**
only, which are complete across all three benchmarks. Sizes 200/1,000/2,000/5,000
are complete for MATH500 and HumanEval; their GSM8K seed-2028/2029 runs are in
Slurm array `64033383`. A watcher regenerates
`analysis/multisize_seed_sensitivity/` automatically when the array finishes —
after that, the full six-size table can replace the two-size table in the reply
and go into the appendix.

Do **not** present size 5,000 as evidence of sampling robustness: the trajectory
pool is itself 5,000, so the three subsets coincide and the tables are identical
(Jaccard 100%, |Δτ| = 0). The report flags this automatically.

## 6. Deliberately missing runs — do not resubmit

- WMT16 en-ro, length 512, fixed threshold (cancelled by request).
- P2.2 configuration transfer, GSM8K length 512, Standard (cancelled by request).

Both are marked as such in the generated reports, and the affected speedup cells
are left empty rather than compared across configurations.
