"""Static capability probe for model directories and quantized checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepinfer.quantization.formats import (
    UnsupportedQuantFormatError,
    detect_source_format,
)

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _find_gguf(path: Path) -> Path | None:
    if path.is_file() and path.suffix.lower() == ".gguf":
        return path
    if path.is_dir():
        candidates = sorted(path.glob("*.gguf"))
        if len(candidates) == 1:
            return candidates[0]
    return None


def probe_model(path: str | Path) -> dict[str, Any]:
    """Return a JSON-serializable support report without allocating weights."""
    root = Path(path)
    report: dict[str, Any] = {
        "path": str(root),
        "exists": root.exists(),
        "supported": False,
        "runtime_ready": False,
        "reason": "",
    }
    if not root.exists():
        report["reason"] = "path does not exist"
        return report

    gguf_file = _find_gguf(root)
    if gguf_file is not None:
        report.update(
            {
                "format": "gguf",
                "gguf_file": str(gguf_file),
                "supported": True,
                "runtime_ready": False,
                "reason": "GGUF must be converted to layered W4A16 before serving",
                "conversion_required": True,
            }
        )
        return report
    if not root.is_dir():
        report["reason"] = "expected an HF model directory or a single .gguf file"
        return report

    config = _read_json(root / "config.json")
    if not config:
        report["reason"] = "missing or invalid config.json"
        return report
    architecture = str((config.get("architectures") or [""])[0])
    model_type = str(config.get("model_type", ""))
    report.update(
        {
            "architecture": architecture,
            "model_type": model_type,
            "quantization": config.get("quantization_config"),
        }
    )

    manifest = _read_json(root / "manifest.json")
    manifest_format = str(manifest.get("format", ""))
    tokenizer_files = [name for name in _TOKENIZER_FILES if (root / name).is_file()]
    tokenizer_config = _read_json(root / "tokenizer_config.json")
    chat_template = tokenizer_config.get("chat_template")
    report["tokenizer_files"] = tokenizer_files
    report["has_chat_template"] = bool(chat_template)

    if manifest_format.startswith("deepinfer_w4a16"):
        report.update(
            {
                "format": "deepinfer_w4a16",
                "supported": True,
                "runtime_ready": True,
                "reason": "native SM86 layered W4A16 checkpoint",
                "manifest": manifest,
            }
        )
        return report

    try:
        info = detect_source_format(root)
        report["format"] = info.format
        report["quantization"] = {
            "format": info.format,
            "bits": info.bits,
            "group_size": info.group_size,
            "layout": info.layout,
            "desc_act": info.desc_act,
        }
        report["conversion_required"] = True
        report["reason"] = (
            f"detected {info.format}; convert it to layered W4A16 before serving"
        )
    except UnsupportedQuantFormatError:
        report["format"] = "hf_dense"
        report["reason"] = "HF dense weights; no W4A16 manifest found"

    try:
        from deepinfer.hugging_face import load_hf_model_config

        model_cfg = load_hf_model_config(root / "config.json")
        gpt = model_cfg.gpt
        report.update(
            {
                "supported": True,
                "model_family": model_cfg.model_type,
                "hidden_size": int(gpt.hidden_size),
                "num_hidden_layers": int(gpt.num_hidden_layers),
                "num_attention_heads": int(gpt.num_attention_heads),
                "num_key_value_heads": int(gpt.num_key_value_heads),
                "head_dim": int(gpt.head_dim),
                "use_sliding_window": bool(gpt.use_sliding_window),
                "sliding_window": gpt.sliding_window,
                "rope_scaling": gpt.rope_scaling,
            }
        )
        if not report.get("conversion_required"):
            report["reason"] = (
                "config is supported by the source model executor; serving needs "
                "a layered W4A16 conversion or a configured dense backend"
            )
    except Exception as exc:  # noqa: BLE001 - probe must explain rejection
        report["supported"] = False
        report["reason"] = str(exc)
        report["error_type"] = type(exc).__name__
    return report


__all__ = ["probe_model"]
