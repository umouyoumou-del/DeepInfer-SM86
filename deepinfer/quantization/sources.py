"""Common source-model abstractions and HF safetensors readers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from deepinfer.quantization.dequant import dequantize_awq, dequantize_gptq
from deepinfer.quantization.formats import SourceFormatInfo

_SUPPORTED_ARCHITECTURES = {
    "Qwen2ForCausalLM",
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
}
_MODEL_TYPE_TO_ARCH = {
    "qwen2": "Qwen2ForCausalLM",
    "llama": "LlamaForCausalLM",
    "mistral": "MistralForCausalLM",
    "gemma": "GemmaForCausalLM",
    "gemma2": "Gemma2ForCausalLM",
}


class SourceCheckpointError(RuntimeError):
    """Raised when a source checkpoint is missing or inconsistent."""


@dataclass
class LayerWeights:
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    v_weight: torch.Tensor
    o_weight: torch.Tensor
    gate_weight: torch.Tensor
    up_weight: torch.Tensor
    down_weight: torch.Tensor
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    q_bias: torch.Tensor | None = None
    k_bias: torch.Tensor | None = None
    v_bias: torch.Tensor | None = None
    pre_feedforward_norm: torch.Tensor | None = None
    post_feedforward_norm: torch.Tensor | None = None


class SafetensorsCheckpointSource:
    """Lazy reader for GPTQ/AWQ checkpoints stored as HF safetensors shards."""

    def __init__(self, format_info: SourceFormatInfo) -> None:
        self.format_info = format_info
        self.root = format_info.path
        config_path = self.root / "config.json"
        if not config_path.is_file():
            raise SourceCheckpointError(f"missing config.json in {self.root}")
        self.config: dict[str, Any] = json.loads(
            config_path.read_text(encoding="utf-8")
        )
        architecture = (self.config.get("architectures") or [""])[0]
        if not architecture:
            architecture = _MODEL_TYPE_TO_ARCH.get(
                str(self.config.get("model_type", "")), ""
            )
        if architecture not in _SUPPORTED_ARCHITECTURES:
            raise SourceCheckpointError(
                f"unsupported architecture {architecture!r}; supported: "
                f"{sorted(_SUPPORTED_ARCHITECTURES)}"
            )
        self.architecture = architecture
        self._files_by_key: dict[str, Path] = {}
        self._file_keys: dict[Path, set[str]] = {}
        self._build_file_index()
        self.tie_word_embeddings = self._resolve_tie_embeddings()

    def _build_file_index(self) -> None:
        index_path = self.root / "model.safetensors.index.json"
        files: list[Path]
        weight_map: dict[str, str] = {}
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            value = index.get("weight_map")
            if isinstance(value, dict):
                weight_map = {str(key): str(path) for key, path in value.items()}
            files = sorted({self.root / name for name in weight_map.values()})
        else:
            files = sorted(self.root.glob("*.safetensors"))
        if not files:
            raise SourceCheckpointError(f"no safetensors shards found in {self.root}")
        from safetensors import safe_open

        for path in files:
            if not path.is_file():
                raise SourceCheckpointError(f"missing safetensors shard: {path}")
            with safe_open(str(path), framework="pt", device="cpu") as tensors:
                keys = set(tensors.keys())
            self._file_keys[path] = keys
            for key in keys:
                existing = self._files_by_key.get(key)
                if existing is not None and existing != path:
                    raise SourceCheckpointError(
                        f"tensor {key!r} appears in both {existing} and {path}"
                    )
                self._files_by_key[key] = path

    def _resolve_tie_embeddings(self) -> bool:
        configured = self.config.get("tie_word_embeddings")
        if configured is not None:
            return bool(configured)
        has_head = "lm_head.weight" in self._files_by_key or (
            "lm_head.qweight" in self._files_by_key
        )
        return not has_head

    def has(self, name: str) -> bool:
        return name in self._files_by_key

    def get(self, name: str) -> torch.Tensor:
        path = self._files_by_key.get(name)
        if path is None:
            raise SourceCheckpointError(f"missing source tensor: {name}")
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            return tensors.get_tensor(name)

    def _dequantize_base(self, base: str) -> torch.Tensor:
        qweight_name = f"{base}.qweight"
        if qweight_name not in self._files_by_key:
            weight_name = f"{base}.weight"
            if weight_name in self._files_by_key:
                return self.get(weight_name).to(torch.float32)
            raise SourceCheckpointError(f"missing linear weight: {base}")
        scales = self.get(f"{base}.scales")
        qzeros = (
            self.get(f"{base}.qzeros") if self.has(f"{base}.qzeros") else None
        )
        g_idx = self.get(f"{base}.g_idx") if self.has(f"{base}.g_idx") else None
        bits = int(self.format_info.bits or 4)
        group_size = int(self.format_info.group_size or 128)
        qweight = self.get(qweight_name)
        if self.format_info.format == "gptq":
            if self.format_info.desc_act and g_idx is None:
                raise SourceCheckpointError(
                    f"GPTQ layer {base!r} is desc_act=true but has no g_idx; "
                    "cannot reconstruct activation-order groups safely"
                )
            return dequantize_gptq(
                qweight,
                scales,
                qzeros,
                bits=bits,
                group_size=group_size,
                g_idx=g_idx,
            )
        if self.format_info.format == "awq":
            return dequantize_awq(
                qweight,
                scales,
                qzeros,
                bits=bits,
                group_size=group_size,
                g_idx=g_idx,
            )
        raise SourceCheckpointError(
            f"unsupported HF-quantized format: {self.format_info.format}"
        )

    def load_dense(self, name: str) -> torch.Tensor:
        if self.has(name):
            return self.get(name).to(torch.float32)
        base = name.removesuffix(".weight")
        if self.has(f"{base}.qweight"):
            raise SourceCheckpointError(
                f"quantized embedding/norm tensor {base!r} is not supported; "
                "this importer expects a dense embedding/norm tensor"
            )
        raise SourceCheckpointError(f"missing source tensor: {name}")

    def load_embedding(self) -> torch.Tensor:
        return self.load_dense("model.embed_tokens.weight")

    def load_final_norm(self) -> torch.Tensor:
        return self.load_dense("model.norm.weight")

    def load_lm_head(self) -> torch.Tensor | None:
        if self.tie_word_embeddings:
            if self.has("lm_head.weight") or self.has("lm_head.qweight"):
                self.tie_word_embeddings = False
            else:
                return None
        return self._dequantize_base("lm_head")

    def load_layer(self, layer_index: int) -> LayerWeights:
        base = f"model.layers.{layer_index}"
        q_base = f"{base}.self_attn.q_proj"
        k_base = f"{base}.self_attn.k_proj"
        v_base = f"{base}.self_attn.v_proj"
        o_base = f"{base}.self_attn.o_proj"
        gate_base = f"{base}.mlp.gate_proj"
        up_base = f"{base}.mlp.up_proj"
        down_base = f"{base}.mlp.down_proj"

        for unsupported in (
            f"{o_base}.bias",
            f"{gate_base}.bias",
            f"{up_base}.bias",
            f"{down_base}.bias",
        ):
            if self.has(unsupported):
                raise SourceCheckpointError(
                    f"bias {unsupported!r} cannot be represented by the current "
                    "SM86 W4A16 model executor"
                )
        layer = LayerWeights(
            q_weight=self._dequantize_base(q_base),
            k_weight=self._dequantize_base(k_base),
            v_weight=self._dequantize_base(v_base),
            o_weight=self._dequantize_base(o_base),
            gate_weight=self._dequantize_base(gate_base),
            up_weight=self._dequantize_base(up_base),
            down_weight=self._dequantize_base(down_base),
            input_norm=self.load_dense(f"{base}.input_layernorm.weight"),
            post_attention_norm=self.load_dense(
                f"{base}.post_attention_layernorm.weight"
            ),
            q_bias=self.get(f"{q_base}.bias") if self.has(f"{q_base}.bias") else None,
            k_bias=self.get(f"{k_base}.bias") if self.has(f"{k_base}.bias") else None,
            v_bias=self.get(f"{v_base}.bias") if self.has(f"{v_base}.bias") else None,
        )
        if self.has(f"{base}.pre_feedforward_layernorm.weight"):
            layer.pre_feedforward_norm = self.load_dense(
                f"{base}.pre_feedforward_layernorm.weight"
            )
        if self.has(f"{base}.post_feedforward_layernorm.weight"):
            layer.post_feedforward_norm = self.load_dense(
                f"{base}.post_feedforward_layernorm.weight"
            )
        return layer

    @property
    def num_layers(self) -> int:
        return int(self.config["num_hidden_layers"])

    def target_config(self) -> dict[str, Any]:
        config = dict(self.config)
        config.pop("quantization_config", None)
        config.pop("quantization", None)
        config["torch_dtype"] = "float16"
        config["tie_word_embeddings"] = bool(self.tie_word_embeddings)
        config["architectures"] = [self.architecture]
        return config

    @property
    def generation_config(self) -> dict[str, Any] | None:
        path = self.root / "generation_config.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @property
    def tokenizer_root(self) -> Path:
        return self.root

    def close(self) -> None:
        return None


__all__ = [
    "LayerWeights",
    "SafetensorsCheckpointSource",
    "SourceCheckpointError",
]
