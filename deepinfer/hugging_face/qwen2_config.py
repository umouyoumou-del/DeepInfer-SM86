"""Qwen2/Qwen2.5 config compatibility wrapper.

The actual normalization lives in :mod:`deepinfer.model_spec`; this module
keeps the historical ``build_qwen2_model_config`` import path stable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deepinfer.model_spec import build_model_config


def build_qwen2_model_config(hf: dict[str, Any], hf_path: Path) -> Any:
    if hf.get("model_type") != "qwen2":
        raise ValueError(
            f"Expected model_type='qwen2' in {hf_path}, got {hf.get('model_type')!r}"
        )
    return build_model_config(hf, hf_path)


__all__ = ["build_qwen2_model_config"]
