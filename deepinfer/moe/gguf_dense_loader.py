"""Load a dense (non-MoE) transformer straight from a GGUF file.

The MoE runtime already reads packed expert slices out of GGUF.  Dense
checkpoints are still forced through a GPTQ/AWQ/GGUF -> W4A16 conversion step,
which is the main friction for trying a new model.  This module removes that
step for Qwen2/Llama/Mistral-style decoder layouts: it maps llama.cpp tensor
names onto this repo's module names, dequantizes whole tensors once, and hands
back a plain state dict.

Scope: dense tensors only.  MoE tensors (``*_exps``) are rejected with a clear
error rather than silently skipped.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# llama.cpp name -> this repo's parameter name, for the parts that are static.
STATIC_TENSORS = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output_norm.bias": "model.norm.bias",
    "output.weight": "lm_head.weight",
    "output.bias": "lm_head.bias",
}

# blk.<N>.<gguf suffix> -> model.layers.<N>.<repo suffix>
BLOCK_TENSORS = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_norm.bias": "input_layernorm.bias",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.bias": "post_attention_layernorm.bias",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_output.bias": "self_attn.o_proj.bias",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

_BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")
_MOE_MARKERS = ("_exps", "ffn_gate_inp", "ffn_gate_exps", "ffn_down_exps")


class UnsupportedGGUFLayout(RuntimeError):
    """Raised when a GGUF file is not a plain dense decoder."""


def map_tensor_name(name: str) -> str | None:
    """Map one llama.cpp tensor name to this repo's name.

    Returns ``None`` for tensors that have no counterpart (for example the
    rotary embedding frequency table, which is computed, not loaded).
    Raises :class:`UnsupportedGGUFLayout` for MoE tensors.
    """

    if any(marker in name for marker in _MOE_MARKERS):
        raise UnsupportedGGUFLayout(
            f"{name!r} looks like mixture-of-experts weights; "
            "use the streaming MoE runtime instead of the dense loader"
        )
    static = STATIC_TENSORS.get(name)
    if static is not None:
        return static
    match = _BLOCK_RE.match(name)
    if match is None:
        return None
    layer, suffix = int(match.group(1)), match.group(2)
    mapped = BLOCK_TENSORS.get(suffix)
    if mapped is None:
        return None
    return f"model.layers.{layer}.{mapped}"


def build_dense_state_dict(
    index: Any,
    *,
    num_layers: int | None = None,
    dtype: Any = None,
) -> dict[str, Any]:
    """Read and dequantize every dense tensor into one state dict.

    ``index`` only needs ``tensor_names()`` and ``read_tensor(name)``, so a real
    :class:`~deepinfer.moe.gguf_index.GGUFIndex` and a test double are both
    acceptable.
    """

    state: dict[str, Any] = {}
    seen_layers: set[int] = set()
    for name in index.tensor_names():
        mapped = map_tensor_name(name)
        if mapped is None:
            continue
        tensor = index.read_tensor(name)
        if dtype is not None:
            tensor = tensor.astype(dtype, copy=False)
        state[mapped] = tensor
        match = _BLOCK_RE.match(name)
        if match is not None:
            seen_layers.add(int(match.group(1)))

    if num_layers is not None:
        missing = sorted(set(range(num_layers)) - seen_layers)
        if missing:
            raise UnsupportedGGUFLayout(
                "GGUF file is missing tensors for layer(s): "
                + ", ".join(str(layer) for layer in missing[:8])
                + (" ..." if len(missing) > 8 else "")
            )
    if "model.embed_tokens.weight" not in state:
        raise UnsupportedGGUFLayout("GGUF file has no token_embd.weight")
    return state


def infer_layer_count(index: Any) -> int:
    layers: set[int] = set()
    for name in index.tensor_names():
        match = _BLOCK_RE.match(name)
        if match is not None:
            layers.add(int(match.group(1)))
    if not layers:
        raise UnsupportedGGUFLayout("GGUF file contains no blk.<N> tensors")
    return max(layers) + 1


__all__ = [
    "BLOCK_TENSORS",
    "STATIC_TENSORS",
    "UnsupportedGGUFLayout",
    "build_dense_state_dict",
    "infer_layer_count",
    "map_tensor_name",
]
