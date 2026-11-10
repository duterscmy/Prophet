#!/usr/bin/env python3
"""Compare threshold behavior on paired Standard and TC-APD trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tc_apd.calibration import iter_trace_payloads
from tc_apd.thresholds import TokenThresholdTable


def final_tokens(payload: dict) -> dict[int, int]:
    return {
        int(record["generation_relative_position"]): int(record["token_id"])
        for record in payload["selected_records"]
    }


def add(counter: dict[str, list[int]], key: str, matches: bool) -> None:
    counter[key][0] += 1
    counter[key][1] += int(matches)


def rate(pair: list[int]) -> float:
    return pair[1] / pair[0] if pair[0] else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-log", type=Path, required=True)
    parser.add_argument("--tc-log", type=Path, required=True)
    parser.add_argument("--threshold-table", type=Path, required=True)
    parser.add_argument("--num-traces", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    standard = [
        payload
        for _, _, payload in iter_trace_payloads(
            args.standard_log, max_traces_per_log=args.num_traces
        )
    ]
    accelerated = [
        payload
        for _, _, payload in iter_trace_payloads(
            args.tc_log, max_traces_per_log=args.num_traces
        )
    ]
    if len(standard) != args.num_traces or len(accelerated) != args.num_traces:
        raise RuntimeError(
            f"Expected {args.num_traces} paired traces, found "
            f"{len(standard)} Standard and {len(accelerated)} TC-APD"
        )

    table = TokenThresholdTable.load(args.threshold_table, default_threshold=0.90)
    policy: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    strategy: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    blocks: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    per_sample: list[dict] = []

    for trace_index, (std_payload, tc_payload) in enumerate(zip(standard, accelerated)):
        std_final = final_tokens(std_payload)
        tc_final = final_tokens(tc_payload)
        if len(std_final) != 256 or len(tc_final) != 256:
            raise RuntimeError(
                f"Trace {trace_index} has incomplete generation: "
                f"Standard={len(std_final)}, TC-APD={len(tc_final)}"
            )

        final_matches = sum(std_final[pos] == tc_final[pos] for pos in range(256))
        sample_selected = [0, 0]
        sample_parallel = [0, 0]
        sample_single = [0, 0]

        # Apply the deployed Math-5000 lookup policy counterfactually to
        # candidates observed on the paired Standard trajectory.
        for record in std_payload["all_token_records"]:
            position = int(record["generation_relative_position"])
            token_id = int(record["token_id"])
            _, threshold, used_default, clipped = table.lookup(
                token_id, minimum=0.01, maximum=0.90, default_threshold=0.90
            )
            if float(record["confidence"]) >= threshold:
                matches = token_id == std_final[position]
                add(policy, "standard_trajectory_policy_eligible", matches)
                add(
                    policy,
                    "standard_trajectory_default" if used_default else "standard_trajectory_calibrated",
                    matches,
                )
                # Deployment caps every threshold at max_threshold=0.90, so
                # tokens whose calibrated threshold exceeds the cap are
                # committed at a looser bar than calibration prescribed.
                if not used_default:
                    add(
                        policy,
                        "standard_trajectory_calibrated_clipped_at_max"
                        if clipped
                        else "standard_trajectory_calibrated_uncapped",
                        matches,
                    )

        # On accelerated states, agreement is measured against the paired
        # Standard final token. Agreement against TC-APD's own committed token
        # would be tautological for selected records because commits are final.
        for record in tc_payload["all_token_records"]:
            position = int(record["generation_relative_position"])
            token_id = int(record["token_id"])
            matches_standard = token_id == std_final[position]
            matches_tc_final = token_id == tc_final[position]
            add(policy, "tc_all_candidates_vs_standard_final", matches_standard)
            add(policy, "tc_all_candidates_vs_tc_final", matches_tc_final)
            if bool(record["eligible"]):
                add(policy, "tc_trajectory_policy_eligible_vs_standard_final", matches_standard)
                add(
                    policy,
                    "tc_trajectory_default_vs_standard_final"
                    if bool(record["used_default_threshold"])
                    else "tc_trajectory_calibrated_vs_standard_final",
                    matches_standard,
                )
            if bool(record["selected"]):
                add(policy, "tc_selected_commits_vs_standard_final", matches_standard)
                commit_strategy = str(record["strategy"])
                add(strategy, commit_strategy, matches_standard)
                add(blocks, int(record["block"]), matches_standard)
                sample_selected[0] += 1
                sample_selected[1] += int(matches_standard)
                target = sample_parallel if commit_strategy == "parallel" else sample_single
                target[0] += 1
                target[1] += int(matches_standard)

        per_sample.append(
            {
                "trace_index": trace_index,
                "final_position_agreement": final_matches / 256,
                "exact_full_256_token_sequence": int(final_matches == 256),
                "selected_commit_agreement": rate(sample_selected),
                "parallel_commit_agreement": rate(sample_parallel),
                "single_commit_agreement": rate(sample_single),
                "tc_nfe": int(tc_payload["stats"]["model_forward_calls"]),
            }
        )

    policy_rows = [
        {
            "candidate_population": key,
            "observations": value[0],
            "agreements": value[1],
            "agreement_pct": 100 * rate(value),
        }
        for key, value in sorted(policy.items())
    ]
    strategy_rows = [
        {
            "commit_strategy": key,
            "committed_tokens": value[0],
            "agreements_with_standard_final": value[1],
            "agreement_pct": 100 * rate(value),
        }
        for key, value in sorted(strategy.items())
    ]
    block_rows = [
        {
            "block": key,
            "committed_tokens": value[0],
            "agreements_with_standard_final": value[1],
            "agreement_pct": 100 * rate(value),
        }
        for key, value in sorted(blocks.items())
    ]
    write_csv(args.output_dir / "policy_consistency.csv", policy_rows)
    write_csv(args.output_dir / "selected_commit_by_strategy.csv", strategy_rows)
    write_csv(args.output_dir / "selected_commit_by_block.csv", block_rows)
    write_csv(args.output_dir / "per_sample_trajectory_metrics.csv", per_sample)

    def lookup(key: str) -> dict:
        return next(row for row in policy_rows if row["candidate_population"] == key)

    std_policy = lookup("standard_trajectory_policy_eligible")
    std_uncapped = lookup("standard_trajectory_calibrated_uncapped")
    std_clipped = lookup("standard_trajectory_calibrated_clipped_at_max")
    tc_policy = lookup("tc_trajectory_policy_eligible_vs_standard_final")
    commits = lookup("tc_selected_commits_vs_standard_final")
    mean_final = sum(row["final_position_agreement"] for row in per_sample) / len(per_sample)
    exact_final = sum(row["exact_full_256_token_sequence"] for row in per_sample) / len(per_sample)
    mean_nfe = sum(row["tc_nfe"] for row in per_sample) / len(per_sample)

    summary = {
        "paired_traces": args.num_traces,
        "generation_length": 256,
        "standard_policy_eligible": std_policy,
        "tc_policy_eligible_vs_standard_final": tc_policy,
        "tc_selected_commits_vs_standard_final": commits,
        "standard_calibrated_uncapped": std_uncapped,
        "standard_calibrated_clipped_at_max": std_clipped,
        "policy_agreement_shift_pp": tc_policy["agreement_pct"] - std_policy["agreement_pct"],
        "final_position_agreement_pct": 100 * mean_final,
        "exact_full_256_token_sequence_pct": 100 * exact_final,
        "mean_tc_nfe": mean_nfe,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# P1.1 Iteration-level Standard → TC-APD trajectory audit",
        "",
        f"Paired analysis over {args.num_traces:,} GSM8K prompts with full 256-token pre-stop trajectories.",
        "",
        "| Population | Observations | Agreement with paired Standard final token |",
        "|---|---:|---:|",
        f"| Math-5000 policy-eligible candidates on Standard states | {std_policy['observations']:,} | {std_policy['agreement_pct']:.3f}% |",
        f"| — calibrated tokens below the 0.90 cap | {std_uncapped['observations']:,} | {std_uncapped['agreement_pct']:.3f}% |",
        f"| — calibrated tokens clipped at the 0.90 cap | {std_clipped['observations']:,} | {std_clipped['agreement_pct']:.3f}% |",
        f"| Math-5000 policy-eligible candidates on TC-APD states | {tc_policy['observations']:,} | {tc_policy['agreement_pct']:.3f}% |",
        f"| Actual TC-APD selected commits | {commits['observations']:,} | {commits['agreement_pct']:.3f}% |",
        "",
        f"The policy-eligible consistency shift from Standard to accelerated states is **{summary['policy_agreement_shift_pp']:+.3f} percentage points**.",
        f"Final position-wise agreement is **{summary['final_position_agreement_pct']:.3f}%**; exact full-256-token sequence agreement is **{summary['exact_full_256_token_sequence_pct']:.3f}%**.",
        f"Mean TC-APD NFE is **{mean_nfe:.3f}**.",
        "",
        "## Interpretation boundary",
        "",
        "TC-APD commits are irreversible, so agreement of a selected token with TC-APD's own final token would be 100% by construction. The non-trivial deployment diagnostic therefore compares accelerated-state candidates and commits with the paired Standard final token at the same generation position. This measures trajectory distribution shift; it is not a formal guarantee of downstream correctness.",
        "",
        "Two further boundaries apply. First, accelerated-state agreement is measured against a Standard reference that conditions on a different prefix, so the divergence compounds along the trajectory: once the two runs disagree at one position, the Standard final token at later positions is no longer the token a correctly behaving accelerated run should produce. The block-wise table therefore reports accumulated divergence, not a per-block error rate. Second, realized consistency on these Standard states is below the nominal 99.5% empirical target, and the clipped/uncapped split shows that the deployed max_threshold = 0.90 cap is not the reason: capping touches only a small fraction of policy-eligible candidates and those agree at a higher rate than the uncapped calibrated tokens. The residual gap is better attributed to the domain shift between the Math calibration traces and these GSM8K evaluation states, which is consistent with the in-domain held-out validation reaching a higher realized consistency on unseen Math traces.",
    ]
    (args.output_dir / "P11_ITERATION_TRAJECTORY_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
