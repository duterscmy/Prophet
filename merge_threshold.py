#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def load_threshold_json(path: str) -> Dict[str, float]:
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path_obj, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Threshold file must be a JSON object: {path}")

    threshold_dict = {}
    for k, v in data.items():
        try:
            token_id = str(k)
            threshold = float(v)
        except Exception as e:
            raise ValueError(
                f"Invalid threshold entry in {path}: key={k}, value={v}"
            ) from e

        threshold_dict[token_id] = threshold

    return threshold_dict


def merge_thresholds_max(threshold_files) -> Dict[str, float]:
    merged = {}

    stats = []
    for file_path in threshold_files:
        threshold_dict = load_threshold_json(file_path)
        stats.append((file_path, len(threshold_dict)))

        for token_id, threshold in threshold_dict.items():
            if token_id not in merged:
                merged[token_id] = threshold
            else:
                merged[token_id] = max(merged[token_id], threshold)

    return merged, stats


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple token-threshold JSON files. "
            "For overlapping token ids, keep the maximum threshold."
        )
    )

    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "Input threshold JSON files followed by output JSON path. "
            "Example: python merge_threshold.py gsm8k.json math.json ace.json output.json"
        ),
    )

    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Default: 2.",
    )

    parser.add_argument(
        "--sort_keys",
        action="store_true",
        help="Sort token ids numerically if possible.",
    )

    args = parser.parse_args()

    if len(args.paths) < 3:
        raise ValueError(
            "Please provide at least two input JSON files and one output JSON file.\n"
            "Example: python merge_threshold.py json1 json2 output_json"
        )

    input_files = args.paths[:-1]
    output_file = args.paths[-1]

    merged, stats = merge_thresholds_max(input_files)

    if args.sort_keys:
        merged = dict(
            sorted(
                merged.items(),
                key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]),
            )
        )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=args.indent, ensure_ascii=False)

    print("====== Merge Threshold Statistics ======")
    print(f"Number of input files: {len(input_files)}")

    total_entries = 0
    for file_path, count in stats:
        print(f"{file_path}: {count} tokens")
        total_entries += count

    print("----------------------------------------")
    print(f"Total input entries: {total_entries}")
    print(f"Merged unique tokens: {len(merged)}")
    print(f"Output file: {output_file}")
    print("Merge rule: overlapping tokens use maximum threshold")


if __name__ == "__main__":
    main()