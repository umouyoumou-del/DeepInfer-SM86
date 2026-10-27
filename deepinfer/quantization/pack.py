"""Packing helpers for the native SM86 W4A16 tensor layout."""

from __future__ import annotations

import torch


def pack_symmetric_int4(
    weight: torch.Tensor,
    *,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[out, in]`` FP weights into SM86 packed int4 + scales.

    The native GEMV kernel uses signed symmetric int4 values in ``[-8, 7]``,
    two nibbles per byte, with one FP16 scale per output row and 128-input group.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got {tuple(weight.shape)}")
    out_features, in_features = (int(value) for value in weight.shape)
    if group_size < 1:
        raise ValueError("group_size must be positive")
    if in_features % group_size:
        raise ValueError(
            f"in_features={in_features} must be divisible by group_size={group_size}"
        )
    if in_features % 2:
        raise ValueError("native int4 packing requires an even input dimension")

    source = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    groups = source.view(out_features, in_features // group_size, group_size)
    scales = groups.abs().amax(dim=-1).clamp_min(1e-8) / 7.0
    quantized = torch.clamp(torch.round(groups / scales[..., None]), -8, 7).to(
        torch.int8
    )
    flat = quantized.view(out_features, in_features)
    low = (flat[:, 0::2] & 0x0F).to(torch.uint8)
    high = ((flat[:, 1::2] & 0x0F) << 4).to(torch.uint8)
    packed = (low | high).contiguous()
    return packed, scales.to(torch.float16).contiguous()


def unpack_symmetric_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int = 128,
) -> torch.Tensor:
    """Reference inverse of :func:`pack_symmetric_int4` for validation."""
    if packed.ndim != 2:
        raise ValueError("packed weights must be 2-D")
    out_features = int(packed.shape[0])
    in_features = int(packed.shape[1]) * 2
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    values = torch.empty(
        (out_features, in_features), dtype=torch.float32, device=packed.device
    )
    values[:, 0::2] = low.float()
    values[:, 1::2] = high.float()
    groups = values.view(out_features, in_features // group_size, group_size)
    return (groups * scales.float()[..., None]).reshape(out_features, in_features)


__all__ = ["pack_symmetric_int4", "unpack_symmetric_int4"]
