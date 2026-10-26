"""Model-family abstraction for the SM86 source executor.

The executor should consume a normalized :class:`ModelSpec`, not branch on
Qwen-specific HF config dictionaries.  Architecture adapters translate HF
configs into this one shape; execution code only reads the shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from deepinfer.attrdict import AttrDict

_DTYPE_ALIASES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _dtype(value: Any) -> torch.dtype:
    try:
        return _DTYPE_ALIASES[str(value or "bfloat16").lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported model dtype: {value!r}") from exc


def _single_eos(value: Any) -> int | None:
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else None
    return None if value is None else int(value)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Architecture-independent description consumed by the executor."""

    family: str
    architecture: str
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool
    attention_dropout: float
    hidden_act: str
    mlp_type: str
    tie_word_embeddings: bool
    dtype: torch.dtype = torch.float16
    rope_scaling: dict[str, Any] | None = None
    sliding_window: int | None = None
    use_sliding_window: bool = False
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    adapter_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def validate(self) -> None:
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim; "
                f"got {self.hidden_size} != {self.num_attention_heads} * {self.head_dim}"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.hidden_act not in ("silu", "gelu", "gelu_new"):
            raise ValueError(f"unsupported activation: {self.hidden_act!r}")
        if self.mlp_type not in ("swiglu",):
            raise ValueError(f"unsupported MLP type: {self.mlp_type!r}")
        if self.use_sliding_window:
            raise NotImplementedError("sliding-window attention is not implemented")
        if self.rope_scaling:
            raise NotImplementedError(
                f"rope_scaling is not implemented: {self.rope_scaling!r}"
            )

    def to_gpt_config(self) -> AttrDict:
        """Fields the generic decoder modules consume at runtime."""
        cfg = AttrDict(
            family=self.family,
            model_type=self.model_type,
            attention_type="gqa_full",
            full_attention=True,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            num_key_value_groups=self.num_key_value_groups,
            head_dim=self.head_dim,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            tie_word_embeddings=self.tie_word_embeddings,
            attention_bias=self.attention_bias,
            attention_dropout=self.attention_dropout,
            hidden_act=self.hidden_act,
            activation_type="swiglu",
            window_size=(-1, -1),
            use_sliding_window=False,
            sliding_window=self.sliding_window,
            max_window_layers=self.num_hidden_layers,
            num_nextn_predict_layers=0,
            moe=None,
            engram=None,
            dflash_cfg=None,
            use_fp4_sparse_kv=False,
            fp8_attn=False,
            fp8_mlp=False,
            mxfp8=False,
            hc_enabled=False,
            hc_mult=1,
            hc_shifted=False,
            n_prefill_layer=self.num_hidden_layers,
            attn_module_list=[],
        )
        cfg.update(self.extra)
        return cfg

    def to_model_config(self) -> AttrDict:
        return AttrDict(
            architecture=self.architecture,
            model_type=self.model_type,
            gpt=self.to_gpt_config(),
            dtype=self.dtype,
            max_position_embeddings=self.max_position_embeddings,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_nextn_predict_layers=0,
            sentence_level_mask=AttrDict(
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
            ),
            full_attention=True,
            adapter_path=self.adapter_path,
        )


class ArchitectureAdapter:
    """Translate one HF architecture into :class:`ModelSpec`."""

    family: str = ""
    model_types: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()
    attention_bias_default: bool = False

    def parse(self, hf: dict[str, Any], hf_path: Path) -> ModelSpec:
        return self._base_spec(hf, hf_path)

    def _base_spec(self, hf: dict[str, Any], hf_path: Path) -> ModelSpec:
        model_type = str(hf.get("model_type", ""))
        architectures = hf.get("architectures") or []
        if model_type not in self.model_types:
            raise ValueError(
                f"{hf_path}: expected model_type in {self.model_types}, got {model_type!r}"
            )
        if architectures and architectures != list(self.architectures):
            raise ValueError(
                f"{hf_path}: expected architectures {self.architectures}, got {architectures!r}"
            )

        hidden_size = int(hf["hidden_size"])
        heads = int(hf["num_attention_heads"])
        kv_heads = int(hf.get("num_key_value_heads", heads))
        head_dim = int(hf.get("head_dim") or hidden_size // heads)
        hidden_act = str(hf.get("hidden_act", "silu"))
        if hidden_act not in ("silu", "gelu", "gelu_new"):
            raise ValueError(f"unsupported hidden_act: {hidden_act!r}")
        rope_scaling = hf.get("rope_scaling")
        use_sliding_window = bool(hf.get("use_sliding_window", False))
        sliding_window = hf.get("sliding_window")
        spec = ModelSpec(
            family=self.family,
            architecture=str(architectures[0])
            if architectures
            else self.architectures[0],
            model_type=model_type,
            hidden_size=hidden_size,
            intermediate_size=int(hf["intermediate_size"]),
            num_hidden_layers=int(hf["num_hidden_layers"]),
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            vocab_size=int(hf["vocab_size"]),
            max_position_embeddings=int(hf["max_position_embeddings"]),
            rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
            rope_theta=float(hf.get("rope_theta", 10000.0)),
            attention_bias=bool(hf.get("attention_bias", self.attention_bias_default)),
            attention_dropout=float(hf.get("attention_dropout", 0.0)),
            hidden_act=hidden_act,
            mlp_type="swiglu",
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            dtype=_dtype(hf.get("torch_dtype", hf.get("dtype", "bfloat16"))),
            rope_scaling=rope_scaling,
            sliding_window=int(sliding_window) if sliding_window else None,
            use_sliding_window=use_sliding_window,
            bos_token_id=hf.get("bos_token_id"),
            eos_token_id=_single_eos(hf.get("eos_token_id")),
            adapter_path=str(hf.get("lora_adapter_path") or ""),
        )
        spec.validate()
        return spec


class Qwen2Adapter(ArchitectureAdapter):
    family = "qwen2"
    model_types = ("qwen2",)
    architectures = ("Qwen2ForCausalLM",)
    attention_bias_default = True


class LlamaAdapter(ArchitectureAdapter):
    family = "llama"
    model_types = ("llama",)
    architectures = ("LlamaForCausalLM",)
    attention_bias_default = False


_ADAPTERS = (Qwen2Adapter(), LlamaAdapter())


def get_adapter(
    hf: dict[str, Any], hf_path: Path | str = "<config>"
) -> ArchitectureAdapter:
    model_type = str(hf.get("model_type", ""))
    arch = (hf.get("architectures") or [""])[0]
    for adapter in _ADAPTERS:
        if model_type in adapter.model_types or arch in adapter.architectures:
            return adapter
    supported = sorted({f"{a.family}:{a.model_types[0]}" for a in _ADAPTERS})
    raise ValueError(
        f"unsupported HF architecture {arch!r} (model_type={model_type!r}); supported: {supported}"
    )


def supported_families() -> list[str]:
    return [adapter.family for adapter in _ADAPTERS]


def build_model_config(
    hf: dict[str, Any], hf_path: Path | str = "<config>"
) -> AttrDict:
    return get_adapter(hf, hf_path).parse(hf, Path(hf_path)).to_model_config()


def load_model_spec(hf_path: Path | str) -> ModelSpec:
    path = Path(hf_path)
    hf = json.loads(path.read_text(encoding="utf-8"))
    return get_adapter(hf, path).parse(hf, path)


__all__ = [
    "ArchitectureAdapter",
    "LlamaAdapter",
    "ModelSpec",
    "Qwen2Adapter",
    "build_model_config",
    "get_adapter",
    "load_model_spec",
    "supported_families",
]
