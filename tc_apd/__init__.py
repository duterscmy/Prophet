"""Token-Level Calibrated Adaptive Parallel Decoding.

Decoder objects are imported lazily so offline calibration does not require a
PyTorch installation.
"""

from __future__ import annotations

from typing import Any


__all__ = ["DecodeConfig", "DecodeResult", "TokenThresholdTable", "decode"]


def __getattr__(name: str) -> Any:
    if name in {"DecodeConfig", "DecodeResult", "decode"}:
        from .decoder import DecodeConfig, DecodeResult, decode

        return {
            "DecodeConfig": DecodeConfig,
            "DecodeResult": DecodeResult,
            "decode": decode,
        }[name]
    if name == "TokenThresholdTable":
        from .thresholds import TokenThresholdTable

        return TokenThresholdTable
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
