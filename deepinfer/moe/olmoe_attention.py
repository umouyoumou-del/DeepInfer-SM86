"""Attention helpers used by the OLMoE port.

OLMoE is plain multi-head attention (``head_count_kv == head_count == 16``,
head_dim 128).

**Correction (2026-10-31):** this module originally claimed OLMoE normalises
queries and keys *per head*.  It does not.  transformers builds
``q_norm = OlmoeRMSNorm(hidden_size)`` and calls ``q_norm(q_proj(x))`` on the
full projected vector, before it is split into heads, and before RoPE.
``per_head_rms_norm`` below is therefore *not* what OLMoE uses and is kept only
as a general utility; see ``olmoe_layer.py`` for the real path, which is
verified against ``OlmoeDecoderLayer`` in ``tests/moe/test_olmoe_layer.py``.
"""

from __future__ import annotations

import torch


def per_head_rms_norm(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """RMSNorm each head separately, then scale per head-dimension.

    ``tensor`` is ``[..., num_heads * head_dim]`` and ``weight`` is
    ``[num_heads * head_dim]``: OLMoE's ``attn_q_norm`` / ``attn_k_norm`` are
    2048 wide for 16 heads of 128, so every head carries its own 128 scales
    rather than sharing one vector across heads.
    """

    if tensor.shape[-1] != num_heads * head_dim:
        raise ValueError(
            f"expected last dim {num_heads * head_dim}, got {tensor.shape[-1]}"
        )
    if weight.shape[-1] != num_heads * head_dim:
        raise ValueError(
            f"expected weight of {num_heads * head_dim}, got {weight.shape[-1]}"
        )
    leading = tensor.shape[:-1]
    heads = tensor.reshape(*leading, num_heads, head_dim).float()
    variance = heads.pow(2).mean(dim=-1, keepdim=True)
    normalised = heads * torch.rsqrt(variance + eps)
    scaled = normalised * weight.reshape(num_heads, head_dim).float()
    return scaled.reshape(*leading, num_heads * head_dim).to(tensor.dtype)


def reference_per_head_rms_norm(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Deliberately naive version used by the tests."""

    out = torch.empty_like(tensor, dtype=torch.float32)
    for row_index in range(tensor.shape[0]):
        for head in range(num_heads):
            start = head * head_dim
            chunk = tensor[row_index, start : start + head_dim].float()
            rms = torch.sqrt(chunk.pow(2).mean() + eps)
            out[row_index, start : start + head_dim] = (
                chunk / rms * weight[start : start + head_dim].float()
            )
    return out.to(tensor.dtype)
