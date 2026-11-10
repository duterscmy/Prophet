#!/usr/bin/env python3
"""Make a downloaded LLaDA GPTQ checkpoint self-contained for offline jobs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    args = parser.parse_args()

    for name in ("configuration_llada.py", "modeling_llada.py"):
        shutil.copy2(args.base_model / name, args.checkpoint / name)

    config_path = args.checkpoint / "config.json"
    backup_path = args.checkpoint / "config.hub.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)
    config["auto_map"] = {
        key: value.split("--", 1)[-1]
        for key, value in config.get("auto_map", {}).items()
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared offline checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
