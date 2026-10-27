"""Convert supported community checkpoints into the SM86 W4A16 layout."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from deepinfer.quantization.formats import (
    SourceFormatInfo,
    detect_source_format,
)
from deepinfer.quantization.gguf_source import GGUFCheckpointSource
from deepinfer.quantization.pack import pack_symmetric_int4
from deepinfer.quantization.sources import (
    SafetensorsCheckpointSource,
    SourceCheckpointError,
)

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.json",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        if output.is_file():
            raise ValueError(f"output path is a file: {output}")
        if any(output.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"output directory is not empty: {output}; pass --overwrite"
                )
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def _open_source(info: SourceFormatInfo) -> Any:
    if info.format == "gguf":
        return GGUFCheckpointSource(info)
    return SafetensorsCheckpointSource(info)


def _pack_into(
    payload: dict[str, torch.Tensor],
    name: str,
    weight: torch.Tensor,
    *,
    group_size: int,
) -> None:
    packed, scales = pack_symmetric_int4(weight, group_size=group_size)
    payload[f"{name}.packed"] = packed
    payload[f"{name}.scales"] = scales


def _validate_weight(
    name: str,
    weight: torch.Tensor,
    expected: tuple[int, int],
) -> torch.Tensor:
    if weight.ndim != 2 or tuple(weight.shape) != expected:
        raise SourceCheckpointError(
            f"{name} shape {tuple(weight.shape)} does not match expected {expected}"
        )
    return weight

def _copy_tokenizer(
    source: Any,
    output: Path,
    *,
    tokenizer: str | Path | None,
) -> tuple[list[str], bool]:
    root: Path | None = None
    if tokenizer is not None:
        root = Path(tokenizer)
    else:
        candidate = getattr(source, "tokenizer_root", None)
        if candidate is not None:
            root = Path(candidate)
    copied: list[str] = []
    if root is None or not root.exists():
        print(
            "[convert] warning: no tokenizer directory supplied; weights/config "
            "will be written but AutoTokenizer will need a separate tokenizer path",
            flush=True,
        )
        return copied, False
    for name in _TOKENIZER_FILES:
        src = root / name
        if src.is_file():
            shutil.copy2(src, output / name)
            copied.append(name)
    chat_template = getattr(source, "chat_template", None)
    tokenizer_config_path = output / "tokenizer_config.json"
    if chat_template and tokenizer_config_path.is_file():
        config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        if not config.get("chat_template"):
            config["chat_template"] = chat_template
            _write_json(tokenizer_config_path, config)
            if "tokenizer_config.json" not in copied:
                copied.append("tokenizer_config.json")
    if not copied:
        print(
            f"[convert] warning: no tokenizer files found under {root}; "
            "AutoTokenizer will need a separate tokenizer path",
            flush=True,
        )
        return copied, False
    return copied, True


def _copy_generation_config(source: Any, config: dict[str, Any], output: Path) -> None:
    payload = getattr(source, "generation_config", None)
    if not payload:
        payload = {
            key: config[key]
            for key in ("bos_token_id", "eos_token_id", "pad_token_id")
            if config.get(key) is not None
        }
    if payload:
        _write_json(output / "generation_config.json", payload)


def _qkv_biases(
    layer: Any,
    config: dict[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    hidden_size = int(config["hidden_size"])
    num_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    q_bias = layer.q_bias
    k_bias = layer.k_bias
    v_bias = layer.v_bias
    present = any(bias is not None for bias in (q_bias, k_bias, v_bias))
    attention_bias = bool(config.get("attention_bias", False)) or present
    config["attention_bias"] = attention_bias
    if not attention_bias:
        return None, None, None
    if q_bias is None:
        q_bias = torch.zeros(num_heads * head_dim, dtype=torch.float32)
    if k_bias is None:
        k_bias = torch.zeros(kv_heads * head_dim, dtype=torch.float32)
    if v_bias is None:
        v_bias = torch.zeros(kv_heads * head_dim, dtype=torch.float32)
    for name, bias, expected in (
        ("q_proj.bias", q_bias, num_heads * head_dim),
        ("k_proj.bias", k_bias, kv_heads * head_dim),
        ("v_proj.bias", v_bias, kv_heads * head_dim),
    ):
        if bias.ndim != 1 or int(bias.numel()) != expected:
            raise SourceCheckpointError(
                f"attention {name} shape {tuple(bias.shape)} does not match "
                f"({expected},); hidden={hidden_size}"
            )
    return q_bias, k_bias, v_bias

def convert_checkpoint(
    source_path: str | Path,
    output_path: str | Path,
    *,
    group_size: int = 128,
    tokenizer: str | Path | None = None,
    overwrite: bool = False,
    source_format: str | None = None,
) -> dict[str, Any]:
    """Convert GPTQ, AWQ or single-file GGUF to SM86 W4A16 layers."""
    info = detect_source_format(source_path)
    if source_format is not None and source_format.lower() != info.format:
        raise ValueError(
            f"--format={source_format!r} does not match detected format {info.format!r}"
        )
    if group_size < 1:
        raise ValueError("group_size must be positive")
    output = Path(output_path)
    source_resolved = Path(source_path).resolve()
    output_resolved = output.resolve()
    if source_resolved.is_relative_to(output_resolved):
        raise ValueError(
            "output directory must not be the source path or one of its "
            "ancestors; refusing to risk deleting the source checkpoint"
        )
    _prepare_output(output, overwrite)
    source = _open_source(info)
    try:
        config = source.target_config()
        embedding = source.load_embedding().to(torch.float32)
        if embedding.ndim != 2:
            raise SourceCheckpointError(
                f"embedding tensor must be 2-D, got {tuple(embedding.shape)}"
            )
        vocab_size, hidden_size = (int(value) for value in embedding.shape)
        config["vocab_size"] = vocab_size
        config["hidden_size"] = hidden_size
        num_heads = int(config["num_attention_heads"])
        num_kv_heads = int(config.get("num_key_value_heads", num_heads))
        head_dim = int(config.get("head_dim") or hidden_size // num_heads)
        intermediate_size = int(config["intermediate_size"])
        config["head_dim"] = head_dim
        config["num_key_value_heads"] = num_kv_heads

        final_norm = source.load_final_norm().to(torch.float32).reshape(-1)
        if int(final_norm.numel()) != hidden_size:
            raise SourceCheckpointError(
                f"final norm has {final_norm.numel()} values, expected {hidden_size}"
            )
        save_file(
            {"weight": final_norm.to(torch.float16).contiguous()},
            str(output / "norm.safetensors"),
        )
        save_file(
            {"weight": embedding.to(torch.float16).contiguous()},
            str(output / "embed.safetensors"),
        )

        lm_head = source.load_lm_head()
        if lm_head is None:
            config["tie_word_embeddings"] = True
        else:
            lm_head = lm_head.to(torch.float32)
            if tuple(lm_head.shape) != (vocab_size, hidden_size):
                raise SourceCheckpointError(
                    f"lm_head shape {tuple(lm_head.shape)} does not match "
                    f"{(vocab_size, hidden_size)}"
                )
            config["tie_word_embeddings"] = False
            save_file(
                {"weight": lm_head.to(torch.float16).contiguous()},
                str(output / "lm_head.safetensors"),
            )
            gib = lm_head.numel() * 2 / 1024**3
            if gib > 0.5:
                print(
                    f"[convert] note: dense lm_head is {gib:.2f} GiB in FP16; "
                    "the current W4 target format keeps lm_head dense",
                    flush=True,
                )
            del lm_head

        q_out = num_heads * head_dim
        kv_out = num_kv_heads * head_dim
        expected = {
            "q_proj": (q_out, hidden_size),
            "k_proj": (kv_out, hidden_size),
            "v_proj": (kv_out, hidden_size),
            "o_proj": (hidden_size, q_out),
            "gate_proj": (intermediate_size, hidden_size),
            "up_proj": (intermediate_size, hidden_size),
            "down_proj": (hidden_size, intermediate_size),
        }
        layer_files: list[str] = []
        for layer_index in range(source.num_layers):
            layer = source.load_layer(layer_index)
            payload: dict[str, torch.Tensor] = {}
            for name, weight, shape in (
                ("self_attn.q_proj", layer.q_weight, expected["q_proj"]),
                ("self_attn.k_proj", layer.k_weight, expected["k_proj"]),
                ("self_attn.v_proj", layer.v_weight, expected["v_proj"]),
                ("self_attn.o_proj", layer.o_weight, expected["o_proj"]),
                ("mlp.gate_proj", layer.gate_weight, expected["gate_proj"]),
                ("mlp.up_proj", layer.up_weight, expected["up_proj"]),
                ("mlp.down_proj", layer.down_weight, expected["down_proj"]),
            ):
                _pack_into(
                    payload,
                    name,
                    _validate_weight(name, weight, shape),
                    group_size=group_size,
                )
                del weight

            q_bias, k_bias, v_bias = _qkv_biases(layer, config)
            if q_bias is not None:
                payload["self_attn.q_proj.bias"] = q_bias.to(torch.float16)
                payload["self_attn.k_proj.bias"] = k_bias.to(torch.float16)
                payload["self_attn.v_proj.bias"] = v_bias.to(torch.float16)

            input_norm = layer.input_norm.to(torch.float32).reshape(-1)
            post_norm = layer.post_attention_norm.to(torch.float32).reshape(-1)
            for label, norm in (("input", input_norm), ("post_attention", post_norm)):
                if int(norm.numel()) != hidden_size:
                    raise SourceCheckpointError(
                        f"layer {layer_index} {label} norm has "
                        f"{norm.numel()} values, expected {hidden_size}"
                    )
            payload["input_layernorm.weight"] = input_norm.to(torch.float16)
            payload["post_attention_layernorm.weight"] = post_norm.to(torch.float16)
            if config.get("model_type") == "gemma2":
                if (
                    layer.pre_feedforward_norm is None
                    or layer.post_feedforward_norm is None
                ):
                    raise SourceCheckpointError(
                        f"layer {layer_index} is missing Gemma 2 sandwich norms"
                    )
                payload["pre_feedforward_layernorm.weight"] = (
                    layer.pre_feedforward_norm.to(torch.float32)
                    .reshape(-1)
                    .to(torch.float16)
                )
                payload["post_feedforward_layernorm.weight"] = (
                    layer.post_feedforward_norm.to(torch.float32)
                    .reshape(-1)
                    .to(torch.float16)
                )

            layer_file = f"layer_{layer_index:02d}.safetensors"
            save_file(payload, str(output / layer_file))
            layer_files.append(layer_file)
            print(
                f"[convert] layer {layer_index + 1}/{source.num_layers} -> {layer_file}",
                flush=True,
            )
            del payload, layer

        config.setdefault("architectures", [source.architecture])
        config["architectures"] = [source.architecture]
        config["model_type"] = str(config.get("model_type") or info.format)
        config["torch_dtype"] = "float16"
        _write_json(output / "config.json", config)
        manifest = {
            "format": "deepinfer_w4a16_sm86_v1",
            "backend": "w4a16",
            "architecture": source.architecture,
            "model_type": config["model_type"],
            "group_size": group_size,
            "num_layers": len(layer_files),
            "tie_word_embeddings": bool(config.get("tie_word_embeddings")),
            "layer_files": layer_files,
            "source_format": info.format,
            "source_path": str(Path(source_path).resolve()),
            "source_quantization": {
                "bits": info.bits,
                "group_size": info.group_size,
                "quant_method": info.quant_method,
                "symmetric": info.symmetric,
                "desc_act": info.desc_act,
                "layout": info.layout,
            },
            "lm_head_format": "dense_fp16"
            if not config.get("tie_word_embeddings")
            else "tied",
        }
        _write_json(output / "manifest.json", manifest)
        copied, tokenizer_ready = _copy_tokenizer(
            source, output, tokenizer=tokenizer
        )
        _copy_generation_config(source, config, output)
        summary = {
            **manifest,
            "output_path": str(output.resolve()),
            "tokenizer_files": copied,
            "tokenizer_ready": tokenizer_ready,
        }
        print(
            f"[convert] done: {len(layer_files)} layers -> {output} "
            f"(format={info.format}, group_size={group_size})",
            flush=True,
        )
        return summary
    finally:
        closer = getattr(source, "close", None)
        if closer is not None:
            closer()


__all__ = ["convert_checkpoint"]
