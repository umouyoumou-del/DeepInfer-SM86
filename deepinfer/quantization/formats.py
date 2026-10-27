"""Detection and metadata for quantized checkpoint formats.

The SM86 runtime consumes one format: the layered W4A16 checkpoint produced by
``scripts/export_w4_checkpoint.py``.  This module identifies common community
formats that can be converted into it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class UnsupportedQuantFormatError(ValueError):
    """Raised when a checkpoint format cannot be converted by this release."""


@dataclass(frozen=True, slots=True)
class SourceFormatInfo:
    format: str
    path: Path
    bits: int | None = None
    group_size: int | None = None
    quant_method: str = ""
    symmetric: bool | None = None
    desc_act: bool | None = None
    layout: str = ""


_KNOWN_METHODS = {
    "gptq": "gptq",
    "awq": "awq",
    "gguf": "gguf",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _find_gguf_file(path: Path) -> Path | None:
    if path.is_file() and path.suffix.lower() == ".gguf":
        return path
    if path.is_dir():
        candidates = sorted(path.glob("*.gguf"))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise UnsupportedQuantFormatError(
                "split GGUF checkpoints are not supported yet; merge the shards "
                "into one .gguf file first"
            )
    return None


def _safetensors_has_quant_keys(path: Path) -> bool:
    try:
        from safetensors import safe_open
    except ImportError:
        return False
    candidates = sorted(path.glob("*.safetensors"))
    for candidate in candidates[:4]:
        try:
            with safe_open(str(candidate), framework="pt", device="cpu") as tensors:
                tensor_names = set(tensors.keys())
                if any(key.endswith(".qweight") for key in tensor_names):
                    return True
        except Exception:  # noqa: BLE001, S112 - malformed shards are handled later
            continue
    return False


def detect_source_format(path: str | Path) -> SourceFormatInfo:
    """Identify GPTQ, AWQ or GGUF and return quantization metadata."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"quantized checkpoint not found: {root}")

    gguf_file = _find_gguf_file(root)
    if gguf_file is not None:
        return SourceFormatInfo(format="gguf", path=gguf_file, quant_method="gguf")

    if not root.is_dir():
        raise UnsupportedQuantFormatError(
            f"unsupported checkpoint path {root}; expected an HF/GPTQ/AWQ "
            "directory or a .gguf file"
        )

    manifest = _read_json(root / "manifest.json")
    manifest_format = str(manifest.get("format", ""))
    if manifest_format.startswith("deepinfer_w4a16"):
        raise UnsupportedQuantFormatError(
            "the source is already a DeepInfer-SM86 W4A16 checkpoint"
        )

    config = _read_json(root / "config.json")
    quantize_config = _read_json(root / "quantize_config.json")
    quant_cfg = config.get("quantization_config")
    if not isinstance(quant_cfg, dict):
        quant_cfg = quantize_config
    method = str(
        quant_cfg.get("quant_method")
        or quant_cfg.get("method")
        or quant_cfg.get("quant_type")
        or ""
    ).lower()
    if method in ("auto-gptq", "gptqmodel"):
        method = "gptq"
    if method == "autoawq":
        method = "awq"

    # Some GPTQ exports omit quantization_config and only keep packed tensors.
    if method not in _KNOWN_METHODS and _safetensors_has_quant_keys(root):
        method = "gptq"

    if method not in ("gptq", "awq"):
        raise UnsupportedQuantFormatError(
            f"unsupported quantization format in {root}: "
            f"quant_method={method or '<missing>'!r}; supported: GPTQ, AWQ, GGUF"
        )

    bits = quant_cfg.get("bits", quant_cfg.get("w_bit", quant_cfg.get("bit")))
    group_size = quant_cfg.get("group_size", quant_cfg.get("q_group_size"))
    if group_size is not None and int(group_size) < 1:
        raise UnsupportedQuantFormatError(
            f"per-channel group_size={group_size!r} is not supported; "
            "use a grouped GPTQ/AWQ checkpoint"
        )
    if bits is not None:
        if method == "gptq" and int(bits) not in (2, 4, 8):
            raise UnsupportedQuantFormatError(
                f"GPTQ bits={bits!r} is not supported; supported: 2, 4, 8"
            )
        if method == "awq" and int(bits) != 4:
            raise UnsupportedQuantFormatError(
                f"AWQ bits={bits!r} is not supported; supported: 4"
            )
    symmetric = quant_cfg.get("sym", quant_cfg.get("symmetric"))
    desc_act = quant_cfg.get("desc_act", quant_cfg.get("act_order"))
    layout = ""
    if method == "awq":
        layout = str(
            quant_cfg.get("version") or quant_cfg.get("linear_version") or "gemm"
        ).lower()
        if layout not in ("gemm", "autoawq"):
            raise UnsupportedQuantFormatError(
                f"AWQ layout {layout!r} is not supported; pack the checkpoint "
                "with the AutoAWQ GEMM layout"
            )
    elif method == "gptq":
        layout = str(
            quant_cfg.get("checkpoint_format") or quant_cfg.get("format") or "gptq"
        ).lower()
        if layout not in ("gptq", "auto-gptq", "autogptq"):
            raise UnsupportedQuantFormatError(
                f"GPTQ checkpoint layout {layout!r} is not supported; supported "
                "layout is the standard GPTQ int32 qweight layout"
            )
    return SourceFormatInfo(
        format=method,
        path=root,
        bits=int(bits) if bits is not None else None,
        group_size=int(group_size) if group_size is not None else None,
        quant_method=method,
        symmetric=bool(symmetric) if symmetric is not None else None,
        desc_act=bool(desc_act) if desc_act is not None else None,
        layout=layout,
    )


def supported_formats() -> list[str]:
    return ["gptq", "awq", "gguf"]


__all__ = [
    "SourceFormatInfo",
    "UnsupportedQuantFormatError",
    "detect_source_format",
    "supported_formats",
]
