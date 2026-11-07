"""Loading and querying token-specific confidence thresholds."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TokenThresholdTable:
    """A vocabulary-aligned token threshold lookup table.

    Both the repository's legacy flat JSON format and a metadata-wrapped format
    are accepted::

        {"13": 0.72, "42": 0.81}

        {
          "thresholds": {"13": 0.72, "42": 0.81},
          "default_threshold": 0.9,
          "metadata": {"rho": 0.995}
        }
    """

    thresholds: Mapping[int, float]
    default_threshold: float = 0.9
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        default_threshold: float | None = None,
    ) -> "TokenThresholdTable":
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Token threshold table not found: {source}")

        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"Threshold table must be a JSON object: {source}")

        if "thresholds" in payload:
            raw_thresholds = payload["thresholds"]
            table_default = payload.get("default_threshold", 0.9)
            metadata = payload.get("metadata", {})
        else:
            raw_thresholds = payload
            table_default = 0.9
            metadata = {}

        if not isinstance(raw_thresholds, dict):
            raise ValueError(f"'thresholds' must be a JSON object: {source}")

        thresholds: dict[int, float] = {}
        for raw_token_id, raw_threshold in raw_thresholds.items():
            try:
                token_id = int(raw_token_id)
                threshold = float(raw_threshold)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid threshold entry {raw_token_id!r}: {raw_threshold!r}"
                ) from error
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    f"Threshold for token {token_id} must be in [0, 1], got {threshold}"
                )
            thresholds[token_id] = threshold

        resolved_default = (
            float(table_default)
            if default_threshold is None
            else float(default_threshold)
        )
        if not 0.0 <= resolved_default <= 1.0:
            raise ValueError("default_threshold must be in [0, 1]")

        return cls(
            thresholds=thresholds,
            default_threshold=resolved_default,
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    def lookup(
        self,
        token_id: int,
        *,
        minimum: float,
        maximum: float,
        default_threshold: float | None = None,
    ) -> tuple[float, float, bool, bool]:
        """Return raw/applied thresholds and fallback/clipping indicators."""
        token_id = int(token_id)
        used_default = token_id not in self.thresholds
        raw = float(
            self.thresholds.get(
                token_id,
                (
                    self.default_threshold
                    if default_threshold is None
                    else default_threshold
                ),
            )
        )
        applied = max(float(minimum), min(float(maximum), raw))
        return raw, applied, used_default, abs(raw - applied) > 1e-12
