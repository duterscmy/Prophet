#!/usr/bin/env python3
"""Create the fixed XSum-500 test subset used in rebuttal experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max-document-tokens", type=int, default=3600)
    args = parser.parse_args()

    source = load_dataset("EdinburghNLP/xsum", split="test")
    indices = sorted(random.Random(args.seed).sample(range(len(source)), args.size))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    records = []
    truncated = 0
    original_lengths = []
    stored_lengths = []
    for source_index in indices:
        row = source[source_index]
        token_ids = tokenizer.encode(row["document"], add_special_tokens=False)
        original_lengths.append(len(token_ids))
        if len(token_ids) > args.max_document_tokens:
            token_ids = token_ids[: args.max_document_tokens]
            document = tokenizer.decode(token_ids, skip_special_tokens=True)
            truncated += 1
        else:
            document = row["document"]
        stored_lengths.append(len(token_ids))
        records.append({
            "id": row["id"],
            "source_index": source_index,
            "document": document,
            "summary": row["summary"],
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "test.jsonl"
    Dataset.from_list(records).to_json(data_path, orient="records", lines=True)
    id_digest = hashlib.sha256("\n".join(row["id"] for row in records).encode()).hexdigest()
    manifest = {
        "source": "EdinburghNLP/xsum",
        "source_split": "test",
        "source_size": len(source),
        "subset_size": args.size,
        "sampling": "uniform_without_replacement_then_source_order",
        "seed": args.seed,
        "max_document_tokens": args.max_document_tokens,
        "truncated_documents": truncated,
        "max_original_document_tokens": max(original_lengths),
        "max_stored_document_tokens": max(stored_lengths),
        "mean_original_document_tokens": sum(original_lengths) / len(original_lengths),
        "selected_id_sha256": id_digest,
        "data_path": str(data_path),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
