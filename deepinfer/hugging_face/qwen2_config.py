"""Parse Qwen2/Qwen2.5 HF configs into DeepInfer's model config shape.

This is the source-level bridge for the native Qwen2 model executor.  It does
not depend on the DeepSeek V4.1 MoE/MLA config schema: Qwen2 is a dense GQA
decoder and therefore gets its own explicit fields and full-attention cache
requirements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from deepinfer import AttrDict


def _torch_dtype(value: Any) -> torch.dtype:
    name = str(value or "bfloat16").lower()
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Qwen2 dtype: {value!r}") from exc


def build_qwen2_model_config(hf: dict[str, Any], hf_path: Path) -> AttrDict:
    """Build a DeepInfer ``model_cfg`` for a Qwen2/Qwen2.5 checkpoint."""

    if hf.get("model_type") != "qwen2":
        raise ValueError(
            f"Expected model_type='qwen2' in {hf_path}, got {hf.get('model_type')!r}"
        )

    architectures = hf.get("architectures")
    if architectures and architectures != ["Qwen2ForCausalLM"]:
        raise ValueError(f"Unsupported Qwen2 architecture list: {architectures!r}")

    hidden_size = int(hf["hidden_size"])
    num_attention_heads = int(hf["num_attention_heads"])
    num_key_value_heads = int(hf.get("num_key_value_heads", num_attention_heads))
    head_dim = int(hf.get("head_dim") or hidden_size // num_attention_heads)
    if head_dim * num_attention_heads != hidden_size:
        raise ValueError(
            "Qwen2 source integration currently requires hidden_size == "
            f"num_attention_heads * head_dim, got {hidden_size=}, {num_attention_heads=}, {head_dim=}"
        )
    if num_attention_heads % num_key_value_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    num_hidden_layers = int(hf["num_hidden_layers"])
    max_position_embeddings = int(hf["max_position_embeddings"])
    rope_scaling = hf.get("rope_scaling")
    use_sliding_window = bool(hf.get("use_sliding_window", False))
    sliding_window = hf.get("sliding_window")
    if use_sliding_window:
        raise NotImplementedError(
            "Qwen2 sliding-window attention is not implemented in the native model path yet"
        )
    if rope_scaling:
        raise NotImplementedError(
            f"Qwen2 rope_scaling is not implemented in the native model path yet: {rope_scaling!r}"
        )
    hidden_act = str(hf.get("hidden_act", "silu"))
    if hidden_act != "silu":
        raise ValueError(
            f"Qwen2 native path currently supports hidden_act='silu', got {hidden_act!r}"
        )
    window_size = (-1, -1)

    # Qwen2 attention always uses q/k/v biases and an unbiased output projection.
    attention_bias = True
    gpt = AttrDict(
        model_type="qwen2",
        attention_type="gqa_full",
        full_attention=not use_sliding_window,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        intermediate_size=int(hf["intermediate_size"]),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_key_value_groups=num_attention_heads // num_key_value_heads,
        head_dim=head_dim,
        vocab_size=int(hf["vocab_size"]),
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=float(hf.get("rms_norm_eps", 1e-6)),
        rope_theta=float(hf.get("rope_theta", 10000.0)),
        rope_scaling=rope_scaling,
        tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
        attention_bias=attention_bias,
        attention_dropout=float(hf.get("attention_dropout", 0.0)),
        hidden_act=hidden_act,
        activation_type="swiglu",
        window_size=window_size,
        use_sliding_window=use_sliding_window,
        sliding_window=sliding_window,
        max_window_layers=int(hf.get("max_window_layers", num_hidden_layers)),
        num_nextn_predict_layers=0,
        # Fields expected by generic runtime code; Qwen2 does not use MoE,
        # engram, DFlash, compressed KV, or the DeepSeek HC topology.
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
        n_prefill_layer=num_hidden_layers,
        attn_module_list=[],
    )

    eos_token_id = hf.get("eos_token_id")
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0] if eos_token_id else None

    return AttrDict(
        architecture="Qwen2ForCausalLM",
        model_type="qwen2",
        gpt=gpt,
        dtype=_torch_dtype(hf.get("torch_dtype", hf.get("dtype", "bfloat16"))),
        max_position_embeddings=max_position_embeddings,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_nextn_predict_layers=0,
        sentence_level_mask=AttrDict(
            bos_token_id=hf.get("bos_token_id"),
            eos_token_id=eos_token_id,
        ),
        full_attention=not use_sliding_window,
        adapter_path=str(hf.get("lora_adapter_path") or ""),
    )


__all__ = ["build_qwen2_model_config"]
