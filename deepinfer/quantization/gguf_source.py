"""GGUF source adapter for the SM86 W4A16 converter.

GGUF is a container format: the tensor payload can use many ggml quant types
(Q4_0, Q5_K, Q6_K, ...).  We delegate block decoding to the pure-Python
``gguf`` package and then re-quantize the resulting dense weights into the
SM86 W4A16 layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepinfer.quantization.formats import SourceFormatInfo
from deepinfer.quantization.sources import (
    _MODEL_TYPE_TO_ARCH,
    LayerWeights,
    SourceCheckpointError,
)

_GGUF_ARCH_TO_MODEL_TYPE = {
    "qwen2": "qwen2",
    "llama": "llama",
    "mistral": "mistral",
    "gemma": "gemma",
    "gemma2": "gemma2",
}


def _as_int(value: Any, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise SourceCheckpointError("missing required GGUF integer metadata")
        return int(default)
    if isinstance(value, (list, tuple)):
        if not value:
            return _as_int(default)
        value = value[0]
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (list, tuple)):
        if not value:
            return float(default)
        value = value[0]
    return float(value)


class GGUFCheckpointSource:
    """Read a single-file GGUF model and expose dense layer tensors lazily."""

    def __init__(self, format_info: SourceFormatInfo) -> None:
        try:
            import gguf
            from gguf import GGUFReader
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise SourceCheckpointError(
                "GGUF import requires the pure-Python 'gguf' package; "
                "install it with `pip install gguf`"
            ) from exc
        self.gguf = gguf
        self.format_info = format_info
        self.path = format_info.path
        self.reader = GGUFReader(str(self.path))
        self._tensors = {tensor.name: tensor for tensor in self.reader.tensors}
        arch = str(self._meta("general.architecture", "")).lower()
        if arch not in _GGUF_ARCH_TO_MODEL_TYPE:
            raise SourceCheckpointError(
                f"unsupported GGUF architecture {arch!r}; supported dense "
                f"families: {sorted(_GGUF_ARCH_TO_MODEL_TYPE)}"
            )
        self.gguf_arch = arch
        sliding_window = self._meta(f"{arch}.attention.sliding_window")
        model_type = _GGUF_ARCH_TO_MODEL_TYPE[arch]
        # Many Mistral GGUFs use the llama architecture and mark SWA only via
        # the sliding-window metadata.  Preserve the stronger behavior.
        if arch == "llama" and sliding_window is not None:
            model_type = "mistral"
        self.model_type = model_type
        self.architecture = _MODEL_TYPE_TO_ARCH[model_type]
        self.tie_word_embeddings = "output.weight" not in self._tensors
        self.metadata = self._collect_metadata()
        self.config = self._build_config()

    def _meta(self, key: str, default: Any = None) -> Any:
        field = self.reader.get_field(key)
        if field is None:
            return default
        return field.contents()

    def _logical_shape(self, name: str) -> tuple[int, ...]:
        tensor = self._tensors.get(name)
        if tensor is None:
            raise SourceCheckpointError(f"missing GGUF tensor: {name}")
        return tuple(int(value) for value in reversed(tensor.shape.tolist()))

    def _tensor(self, name: str) -> torch.Tensor:
        tensor = self._tensors.get(name)
        if tensor is None:
            raise SourceCheckpointError(f"missing GGUF tensor: {name}")
        try:
            decoded = self.gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        except Exception as exc:
            raise SourceCheckpointError(
                f"cannot dequantize GGUF tensor {name!r} "
                f"({tensor.tensor_type.name}): {exc}"
            ) from exc
        array = np.asarray(decoded)
        if array.dtype == np.uint16:
            array = array.view(np.float16).astype(np.float32)
        elif array.dtype == np.uint32:
            array = array.view(np.float32)
        if not np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32)
        return torch.tensor(array, dtype=torch.float32)

    def _collect_metadata(self) -> dict[str, Any]:
        arch = self.gguf_arch
        embedding_shape = self._logical_shape("token_embd.weight")
        hidden_size = _as_int(
            self._meta(f"{arch}.embedding_length"), embedding_shape[1]
        )
        heads = _as_int(self._meta(f"{arch}.attention.head_count"), 0)
        if heads < 1:
            raise SourceCheckpointError("GGUF metadata lacks attention.head_count")
        kv_heads = _as_int(self._meta(f"{arch}.attention.head_count_kv"), heads)
        head_dim = _as_int(
            self._meta(f"{arch}.attention.key_length"), hidden_size // heads
        )
        intermediate = self._meta(f"{arch}.feed_forward_length")
        if intermediate is None and "blk.0.ffn_gate.weight" in self._tensors:
            intermediate = self._logical_shape("blk.0.ffn_gate.weight")[0]
        if intermediate is None:
            raise SourceCheckpointError("GGUF metadata lacks feed_forward_length")
        num_layers = _as_int(self._meta(f"{arch}.block_count"), 0)
        if num_layers < 1:
            indices = {
                int(name.split(".")[1])
                for name in self._tensors
                if name.startswith("blk.") and name.split(".")[1].isdigit()
            }
            num_layers = max(indices, default=-1) + 1
        vocab_meta = self._meta(f"{arch}.vocab_size")
        vocab_size = _as_int(vocab_meta, embedding_shape[0])
        if vocab_meta is None:
            tokens = self._meta("tokenizer.ggml.tokens")
            if isinstance(tokens, list) and tokens:
                vocab_size = max(vocab_size, len(tokens))
        return {
            "hidden_size": hidden_size,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "intermediate_size": _as_int(intermediate),
            "num_hidden_layers": num_layers,
            "vocab_size": vocab_size,
            "context_length": _as_int(
                self._meta(f"{arch}.context_length"), 4096
            ),
            "rms_norm_eps": _as_float(
                self._meta(f"{arch}.attention.layer_norm_rms_epsilon"), 1e-6
            ),
            "rope_theta": _as_float(self._meta(f"{arch}.rope.freq_base"), 10000.0),
            "sliding_window": self._meta(f"{arch}.attention.sliding_window"),
            "attn_logit_softcapping": self._meta(
                f"{arch}.attn_logit_softcapping"
            ),
            "final_logit_softcapping": self._meta(
                f"{arch}.final_logit_softcapping"
            ),
            "attention_scale": self._meta(f"{arch}.attention.scale"),
            "bos_token_id": self._meta("tokenizer.ggml.bos_token_id"),
            "eos_token_id": self._meta("tokenizer.ggml.eos_token_id"),
            "pad_token_id": self._meta("tokenizer.ggml.padding_token_id"),
            "chat_template": self._meta("tokenizer.chat_template"),
            "has_q_bias": "blk.0.attn_q.bias" in self._tensors,
            "has_qkv_bias": "blk.0.attn_qkv.bias" in self._tensors,
        }

    def _build_config(self) -> dict[str, Any]:
        meta = self.metadata
        architecture = self.architecture
        model_type = self.model_type
        hidden_act = (
            "gelu_pytorch_tanh"
            if model_type in ("gemma", "gemma2")
            else "silu"
        )
        config: dict[str, Any] = {
            "architectures": [architecture],
            "model_type": model_type,
            "hidden_size": meta["hidden_size"],
            "intermediate_size": meta["intermediate_size"],
            "num_hidden_layers": meta["num_hidden_layers"],
            "num_attention_heads": meta["num_attention_heads"],
            "num_key_value_heads": meta["num_key_value_heads"],
            "head_dim": meta["head_dim"],
            "vocab_size": meta["vocab_size"],
            "max_position_embeddings": meta["context_length"],
            "rms_norm_eps": meta["rms_norm_eps"],
            "rope_theta": meta["rope_theta"],
            "attention_bias": bool(meta["has_q_bias"] or meta["has_qkv_bias"]),
            "attention_dropout": 0.0,
            "hidden_act": hidden_act,
            "torch_dtype": "float16",
            "tie_word_embeddings": bool(self.tie_word_embeddings),
            "bos_token_id": meta["bos_token_id"],
            "eos_token_id": meta["eos_token_id"],
            "pad_token_id": meta["pad_token_id"],
        }
        if model_type in ("gemma", "gemma2"):
            config["hidden_activation"] = hidden_act
        if meta["sliding_window"] is not None:
            config["sliding_window"] = _as_int(meta["sliding_window"])
            config["use_sliding_window"] = True
        else:
            config["use_sliding_window"] = False
        if model_type == "gemma2":
            config["attn_logit_softcapping"] = _as_float(
                meta["attn_logit_softcapping"], 50.0
            )
            config["final_logit_softcapping"] = _as_float(
                meta["final_logit_softcapping"], 30.0
            )
            if meta["attention_scale"] is not None:
                scale = _as_float(meta["attention_scale"], meta["head_dim"] ** -0.5)
                config["query_pre_attn_scalar"] = 1.0 / (scale * scale)
            else:
                config["query_pre_attn_scalar"] = meta["head_dim"]
        return config

    def _optional_tensor(self, name: str) -> torch.Tensor | None:
        return self._tensor(name) if name in self._tensors else None

    def _load_qkv(
        self, layer_index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix = f"blk.{layer_index}"
        names = (
            f"{prefix}.attn_q.weight",
            f"{prefix}.attn_k.weight",
            f"{prefix}.attn_v.weight",
        )
        if all(name in self._tensors for name in names):
            return tuple(self._tensor(name) for name in names)  # type: ignore[return-value]
        fused_name = f"{prefix}.attn_qkv.weight"
        if fused_name not in self._tensors:
            raise SourceCheckpointError(
                f"layer {layer_index} has neither separate q/k/v tensors nor "
                f"{fused_name}"
            )
        fused = self._tensor(fused_name)
        heads = int(self.config["num_attention_heads"])
        kv_heads = int(self.config["num_key_value_heads"])
        head_dim = int(self.config["head_dim"])
        q_rows = heads * head_dim
        kv_rows = kv_heads * head_dim
        expected = q_rows + 2 * kv_rows
        if int(fused.shape[0]) != expected:
            raise SourceCheckpointError(
                f"fused QKV shape {tuple(fused.shape)} does not match {expected} rows"
            )
        return fused[:q_rows], fused[q_rows : q_rows + kv_rows], fused[
            q_rows + kv_rows :
        ]

    def _load_qkv_bias(
        self, layer_index: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        prefix = f"blk.{layer_index}"
        fused = self._optional_tensor(f"{prefix}.attn_qkv.bias")
        if fused is not None:
            heads = int(self.config["num_attention_heads"])
            kv_heads = int(self.config["num_key_value_heads"])
            head_dim = int(self.config["head_dim"])
            q_rows = heads * head_dim
            kv_rows = kv_heads * head_dim
            return fused[:q_rows], fused[q_rows : q_rows + kv_rows], fused[
                q_rows + kv_rows :
            ]
        return (
            self._optional_tensor(f"{prefix}.attn_q.bias"),
            self._optional_tensor(f"{prefix}.attn_k.bias"),
            self._optional_tensor(f"{prefix}.attn_v.bias"),
        )

    def _load_gate_up(
        self, layer_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = f"blk.{layer_index}"
        gate_name = f"{prefix}.ffn_gate.weight"
        up_name = f"{prefix}.ffn_up.weight"
        if gate_name in self._tensors and up_name in self._tensors:
            return self._tensor(gate_name), self._tensor(up_name)
        fused_name = f"{prefix}.ffn_gate_up.weight"
        if fused_name not in self._tensors:
            raise SourceCheckpointError(
                f"layer {layer_index} has no ffn_gate/ffn_up tensors"
            )
        fused = self._tensor(fused_name)
        if int(fused.shape[0]) % 2:
            raise SourceCheckpointError(
                f"fused gate/up tensor has odd output size {fused.shape[0]}"
            )
        half = int(fused.shape[0]) // 2
        return fused[:half], fused[half:]

    def load_embedding(self) -> torch.Tensor:
        return self._tensor("token_embd.weight")

    def load_final_norm(self) -> torch.Tensor:
        return self._tensor("output_norm.weight")

    def load_lm_head(self) -> torch.Tensor | None:
        if self.tie_word_embeddings:
            return None
        return self._tensor("output.weight")

    def load_layer(self, layer_index: int) -> LayerWeights:
        prefix = f"blk.{layer_index}"
        q_weight, k_weight, v_weight = self._load_qkv(layer_index)
        gate_weight, up_weight = self._load_gate_up(layer_index)
        q_bias, k_bias, v_bias = self._load_qkv_bias(layer_index)
        input_norm = self._tensor(f"{prefix}.attn_norm.weight")
        post_attention_name = (
            f"{prefix}.post_attention_norm.weight"
            if self.model_type == "gemma2"
            else f"{prefix}.ffn_norm.weight"
        )
        post_attention_norm = self._tensor(post_attention_name)
        pre_feedforward_norm = None
        post_feedforward_norm = None
        if self.model_type == "gemma2":
            pre_feedforward_norm = self._tensor(f"{prefix}.ffn_norm.weight")
            post_feedforward_norm = self._tensor(
                f"{prefix}.post_ffw_norm.weight"
            )
        return LayerWeights(
            q_weight=q_weight,
            k_weight=k_weight,
            v_weight=v_weight,
            o_weight=self._tensor(f"{prefix}.attn_output.weight"),
            gate_weight=gate_weight,
            up_weight=up_weight,
            down_weight=self._tensor(f"{prefix}.ffn_down.weight"),
            input_norm=input_norm.reshape(-1),
            post_attention_norm=post_attention_norm.reshape(-1),
            q_bias=q_bias.reshape(-1) if q_bias is not None else None,
            k_bias=k_bias.reshape(-1) if k_bias is not None else None,
            v_bias=v_bias.reshape(-1) if v_bias is not None else None,
            pre_feedforward_norm=(
                pre_feedforward_norm.reshape(-1)
                if pre_feedforward_norm is not None
                else None
            ),
            post_feedforward_norm=(
                post_feedforward_norm.reshape(-1)
                if post_feedforward_norm is not None
                else None
            ),
        )

    @property
    def num_layers(self) -> int:
        return int(self.config["num_hidden_layers"])

    def target_config(self) -> dict[str, Any]:
        return dict(self.config)

    @property
    def generation_config(self) -> dict[str, Any] | None:
        return {
            key: self.config[key]
            for key in ("bos_token_id", "eos_token_id", "pad_token_id")
            if self.config.get(key) is not None
        }

    @property
    def tokenizer_root(self) -> Path:
        return self.path.parent

    @property
    def chat_template(self) -> str | None:
        value = self.metadata.get("chat_template")
        return str(value) if value else None

    def close(self) -> None:
        self._tensors.clear()
        # GGUFReader owns a numpy memmap; dropping it releases the file handle
        # on Windows before the temporary directory is cleaned up.
        self.reader = None


__all__ = ["GGUFCheckpointSource"]
