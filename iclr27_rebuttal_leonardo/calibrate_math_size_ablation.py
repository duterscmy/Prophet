#!/usr/bin/env python3
"""Build nested single-seed TC-APD tables for calibration-size ablations."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tc_apd.calibration import (
    build_grid_thresholds,
    iter_final_consistency_records,
    save_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sizes", default="100,200,500,1000,2000,5000")
    parser.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help=(
            "Number of source traces from which subsets are sampled. "
            "Defaults to the largest requested size."
        ),
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--rho", type=float, default=0.995)
    parser.add_argument("--min-count", type=int, default=200)
    parser.add_argument("--min-accepted", type=int, default=100)
    parser.add_argument("--threshold-start", type=float, default=1.0)
    parser.add_argument("--threshold-end", type=float, default=0.01)
    parser.add_argument("--grid-step", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = sorted({int(value) for value in args.sizes.split(",")})
    if not sizes or sizes[0] <= 0:
        raise ValueError("sizes must contain positive integers")

    pool_size = args.pool_size or sizes[-1]
    if pool_size < sizes[-1]:
        raise ValueError("pool-size must be at least the largest requested size")
    shuffled_indices = list(range(pool_size))
    random.Random(args.seed).shuffle(shuffled_indices)
    rank_by_trace = {
        trace_index: rank for rank, trace_index in enumerate(shuffled_indices)
    }

    observations = {size: defaultdict(list) for size in sizes}
    seen_trace_indices: set[int] = set()
    record_counts = {size: 0 for size in sizes}

    for record in iter_final_consistency_records(
        args.log, max_traces_per_log=pool_size
    ):
        trace_index = int(record["problem_idx"])
        seen_trace_indices.add(trace_index)
        rank = rank_by_trace[trace_index]
        observation = (
            float(record["confidence"]),
            int(bool(record["final_consistent"])),
        )
        token_id = int(record["token_id"])
        for size in sizes:
            if rank < size:
                observations[size][token_id].append(observation)
                record_counts[size] += 1

    if len(seen_trace_indices) != pool_size:
        raise RuntimeError(
            f"Expected {pool_size} traces but found {len(seen_trace_indices)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_log": str(args.log),
        "seed": args.seed,
        "pool_size": pool_size,
        "sizes": sizes,
        "rho": args.rho,
        "min_count": args.min_count,
        "min_accepted": args.min_accepted,
        "threshold_start": args.threshold_start,
        "threshold_end": args.threshold_end,
        "grid_step": args.grid_step,
        "tables": {},
    }

    for size in sizes:
        thresholds, rows = build_grid_thresholds(
            observations[size],
            target_precision=args.rho,
            min_count=args.min_count,
            min_accepted=args.min_accepted,
            threshold_start=args.threshold_start,
            threshold_end=args.threshold_end,
            threshold_step=args.grid_step,
        )
        stem = (
            f"math_n{size}_seed{args.seed}_token_threshold_grid_p99.5_"
            f"mincount{args.min_count}_minaccepted{args.min_accepted}"
        )
        json_path = args.output_dir / f"{stem}.json"
        csv_path = args.output_dir / f"{stem}.csv"
        save_calibration(
            thresholds,
            rows,
            json_path=json_path,
            csv_path=csv_path,
        )
        manifest["tables"][str(size)] = {
            "json": str(json_path),
            "csv": str(csv_path),
            "candidate_records": record_counts[size],
            "observed_token_ids": len(observations[size]),
            "eligible_token_ids": len(rows),
            "calibrated_token_ids": len(thresholds),
            "fallback_token_ids": sum(row.status != "ok" for row in rows),
        }
        print(json.dumps({"size": size, **manifest["tables"][str(size)]}))

    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
