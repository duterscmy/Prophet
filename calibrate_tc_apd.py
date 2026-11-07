#!/usr/bin/env python3
"""Build a TC-APD token-threshold lookup table from standard logs."""

from __future__ import annotations

import argparse
from pathlib import Path

from tc_apd.calibration import (
    build_grid_thresholds,
    collect_final_consistency,
    save_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a TC-APD lookup table. Multiple input logs are pooled before "
            "calibration; calibrate sources separately when source-specific "
            "tables are required."
        )
    )
    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="standard-decoding logs containing all_token_records",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output token-id-to-threshold JSON used at inference",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        help="audit CSV (default: the JSON path with a .csv suffix)",
    )
    parser.add_argument("--rho", type=float, default=0.995)
    parser.add_argument("--min-token-observations", type=int, default=200)
    parser.add_argument("--min-accepted-observations", type=int, default=100)
    parser.add_argument("--threshold-start", type=float, default=1.0)
    parser.add_argument("--threshold-end", type=float, default=0.01)
    parser.add_argument("--grid-step", type=float, default=0.005)
    parser.add_argument(
        "--max-traces-per-log",
        type=int,
        help="optional per-log limit for smoke tests; omit for full calibration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations = collect_final_consistency(
        args.logs, max_traces_per_log=args.max_traces_per_log
    )
    thresholds, rows = build_grid_thresholds(
        observations,
        target_precision=args.rho,
        min_count=args.min_token_observations,
        min_accepted=args.min_accepted_observations,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        threshold_step=args.grid_step,
    )
    stats_output = args.stats_output or args.output.with_suffix(".csv")
    save_calibration(
        thresholds,
        rows,
        json_path=args.output,
        csv_path=stats_output,
    )

    num_records = sum(len(records) for records in observations.values())
    num_without_threshold = sum(row.status != "ok" for row in rows)
    print("====== TC-APD Calibration ======")
    print(f"Input logs: {len(args.logs)}")
    for path in args.logs:
        print(f"  - {path}")
    print(f"Candidate records: {num_records}")
    print(f"Unique token ids: {len(observations)}")
    print(f"Eligible token ids: {len(rows)}")
    print(f"Calibrated token ids: {len(thresholds)}")
    print(f"Eligible token ids using fallback: {num_without_threshold}")
    print(f"Lookup JSON: {args.output}")
    print(f"Audit CSV: {stats_output}")


if __name__ == "__main__":
    main()
