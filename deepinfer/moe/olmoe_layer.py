"""Assemble one OLMoE decoder layer in plain PyTorch.

This exists to validate the wiring - norm placement, RoPE, per-head QK-norm,
residual order, the absence of a shared expert - against a reference
implementation before any paged/quantised path is involved.
"""

from __future__ import annotations

import math

import torch

from deepinfer.moe.olmoe_mlp import moe_forward_batched
from deepinfer.moe.olmoe_packed import moe_forward_packed


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    return (hidden.float() * torch.rsqrt(variance + eps) * weight.float()).to(
        hidden.dtype
    )


_ROPE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def build_rope(
    positions: torch.Tensor, head_dim: int, base: float, *, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE tables for the given positions, cached per shape/device/dtype.

    Computed on the CPU and moved once.  Building a GPU scalar from a Python
    float (``torch.tensor(base, device=...)``) or calling ``.cpu()`` on a CUDA
    tensor forces a synchronisation, and this used to run once per layer - 16
    syncs at about 4 ms each, 43% of a forward pass.
    """

    tokens = int(positions.numel())
    start = int(positions.flatten()[0]) if tokens else 0
    key = (start, tokens, head_dim, float(base), str(device), str(dtype))
    cached = _ROPE_CACHE.get(key)
    if cached is not None:
        return cached

    half = head_dim // 2
    exponents = torch.arange(half, dtype=torch.float32) * (2.0 / head_dim)
    inverse = torch.pow(float(base), -exponents)
    indices = torch.arange(start, start + tokens, dtype=torch.float32).unsqueeze(1)
    angles = indices * inverse.unsqueeze(0)
    tables = (
        torch.cos(angles).to(device=device, dtype=dtype),
        torch.sin(angles).to(device=device, dtype=dtype),
    )
    _ROPE_CACHE[key] = tables
    return tables


def apply_rope(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``tensor`` is ``[batch, heads, tokens, head_dim]``."""

    half = tensor.shape[-1] // 2
    first, second = tensor[..., :half], tensor[..., half:]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)


def decoder_layer_forward(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    num_heads: int,
    eps: float,
    rope_base: float,
    top_k: int,
    attention_mask: torch.Tensor | None = None,
    timings: dict[str, float] | None = None,
    cache: "OLMoEKVKache | None" = None,
    layer_index: int = 0,
) -> torch.Tensor:
    """One OLMoE layer: attn(norm) + MLP(norm), each with a residual."""

    batch, tokens, hidden_size = hidden.shape
    head_dim = hidden_size // num_heads

    marks: list[tuple[str, torch.cuda.Event]] = []

    def _stamp(name: str) -> None:
        if timings is None:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        marks.append((name, event))

    _stamp("start")
    residual = hidden
    normed = rms_norm(hidden, weights["attn_norm"], eps)
    _stamp("attn_norm")
    query = normed @ weights["q_proj"].t()
    key = normed @ weights["k_proj"].t()
    value = (normed @ weights["v_proj"].t()).view(batch, tokens, num_heads, head_dim)
    # OLMoE's q_norm / k_norm are RMSNorm over the *whole* hidden vector,
    # applied to the projection output before it is split into heads.  They are
    # not per-head normalisers: transformers builds them as OlmoeRMSNorm(hidden)
    # and calls q_norm(q_proj(x)).  Verified against OlmoeDecoderLayer in
    # tests/moe/test_olmoe_layer.py.
    query = rms_norm(query, weights["attn_q_norm"], eps).view(
        batch, tokens, num_heads, head_dim
    )
    key = rms_norm(key, weights["attn_k_norm"], eps).view(
        batch, tokens, num_heads, head_dim
    )

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # absolute positions: with a cache the new token continues from where the
    # prefix ended, otherwise every decode step would rotate as position 0
    offset = 0 if cache is None else cache.length(layer_index)
    cos, sin = build_rope(
        torch.arange(offset, offset + tokens, device=hidden.device), head_dim,
        rope_base, device=hidden.device, dtype=hidden.dtype,
    )
    query = apply_rope(query, cos, sin)
    key = apply_rope(key, cos, sin)

    _stamp("qkv_qknorm_rope")
    if cache is not None:
        key_all, value_all = cache.append(layer_index, key, value)
        if key_all.shape[2] == query.shape[2]:
            # whole sequence in one go (prefill): ordinary causal attention
            context = torch.nn.functional.scaled_dot_product_attention(
                query, key_all, value_all, is_causal=True
            )
        else:
            # decode: one new query token against the cached prefix, no mask
            context = torch.nn.functional.scaled_dot_product_attention(
                query, key_all, value_all
            )
    else:
        context = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, is_causal=attention_mask is None
        )
    _stamp("attention")
    context = context.transpose(1, 2).reshape(batch, tokens, hidden_size)
    hidden = residual + context @ weights["o_proj"].t()
    _stamp("o_proj")

    residual = hidden
    normed = rms_norm(hidden, weights["ffn_norm"], eps)
    flat = normed.reshape(batch * tokens, hidden_size)
    use_packed = "packed" in weights and flat.shape[0] == 1
    if use_packed:
        # packed branch: no gather, no dtype conversion - the kernels read the
        # Q4_K/Q6_K bytes directly through per-expert pointers
        experts = moe_forward_packed(
            flat,
            weights["router"].t(),
            weights["packed"],
            top_k=top_k,
            gate_out=weights["packed_gate_out"],
            down_out=hidden_size,
            gate_k=hidden_size,
            down_k=weights["packed_gate_out"],
        ).reshape(batch, tokens, hidden_size)
    else:
        experts = moe_forward_batched(
            flat,
            weights["router"].t(),
            weights["expert_gate"],
            weights["expert_up"],
            weights["expert_down"],
            top_k=top_k,
        ).reshape(batch, tokens, hidden_size)
    _stamp("moe")
    out = residual + experts
    if timings is not None and marks:
        # elapsed_time needs both events completed, so sync first
        torch.cuda.synchronize()
        for index in range(1, len(marks)):
            name = marks[index][0]
            elapsed = marks[index - 1][1].elapsed_time(marks[index][1])
            timings[name] = timings.get(name, 0.0) + elapsed
    return out


class OLMoEKVKache:
    """Per-layer key/value cache so decode does not recompute the prefix.

    Without this, every generated token reprocessed the entire sequence: step 1
    saw 6 tokens, step 12 saw 17, which is why end-to-end throughput degraded
    with length while the single-token cost stayed at 19.7 ms.
    """

    def __init__(self) -> None:
        self._keys: dict[int, torch.Tensor] = {}
        self._values: dict[int, torch.Tensor] = {}

    def append(
        self, layer_index: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached_key = self._keys.get(layer_index)
        if cached_key is None:
            self._keys[layer_index] = key
            self._values[layer_index] = value
            return key, value
        keys = torch.cat([cached_key, key], dim=2)
        values = torch.cat([self._values[layer_index], value], dim=2)
        self._keys[layer_index] = keys
        self._values[layer_index] = values
        return keys, values

    def length(self, layer_index: int = 0) -> int:
        cached = self._keys.get(layer_index)
        return 0 if cached is None else int(cached.shape[2])

    def reset(self) -> None:
        self._keys.clear()
        self._values.clear()
