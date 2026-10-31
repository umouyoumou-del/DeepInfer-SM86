"""Assemble one OLMoE decoder layer in plain PyTorch.

This exists to validate the wiring - norm placement, RoPE, per-head QK-norm,
residual order, the absence of a shared expert - against a reference
implementation before any paged/quantised path is involved.
"""

from __future__ import annotations

import math

import torch

from deepinfer.moe.olmoe_mlp import moe_forward


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    return (hidden.float() * torch.rsqrt(variance + eps) * weight.float()).to(
        hidden.dtype
    )


def build_rope(
    positions: torch.Tensor, head_dim: int, base: float, *, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard rotate-half RoPE tables for the given positions."""

    half = head_dim // 2
    exponent = torch.arange(half, device=device, dtype=torch.float32) * 2.0 / head_dim
    inverse = 1.0 / torch.pow(torch.tensor(base, device=device), exponent)
    angles = positions.float().unsqueeze(1) * inverse.unsqueeze(0)
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


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
) -> torch.Tensor:
    """One OLMoE layer: attn(norm) + MLP(norm), each with a residual."""

    batch, tokens, hidden_size = hidden.shape
    head_dim = hidden_size // num_heads

    residual = hidden
    normed = rms_norm(hidden, weights["attn_norm"], eps)
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

    cos, sin = build_rope(
        torch.arange(tokens, device=hidden.device), head_dim, rope_base,
        device=hidden.device, dtype=hidden.dtype,
    )
    query = apply_rope(query, cos, sin)
    key = apply_rope(key, cos, sin)

    context = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, is_causal=attention_mask is None
    )
    context = context.transpose(1, 2).reshape(batch, tokens, hidden_size)
    hidden = residual + context @ weights["o_proj"].t()

    residual = hidden
    normed = rms_norm(hidden, weights["ffn_norm"], eps)
    flat = normed.reshape(batch * tokens, hidden_size)
    experts = moe_forward(
        flat,
        weights["router"].t(),
        weights["expert_gate"],
        weights["expert_up"],
        weights["expert_down"],
        top_k=top_k,
    ).reshape(batch, tokens, hidden_size)
    return residual + experts
