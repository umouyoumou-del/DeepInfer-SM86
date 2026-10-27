"""Dequantizers for the packed GPTQ and AWQ tensor layouts.

Both formats pack multiple quantized values into int32 words, but with
different orientation and ordering.  These functions intentionally use only
PyTorch CPU operations so a conversion can run on machines without CUDA.
"""

from __future__ import annotations

import torch

AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def _mask(bits: int) -> int:
    return (1 << bits) - 1


def _unpack_last_dim(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack int32 words along the last dimension."""
    if bits not in (2, 4, 8):
        raise ValueError(f"unsupported packed bit width: {bits}")
    shifts = torch.arange(0, 32, bits, dtype=torch.int64)
    values = (
        packed.to(torch.int64).unsqueeze(-1) >> shifts
    ) & _mask(bits)
    return values.reshape(*packed.shape[:-1], packed.shape[-1] * (32 // bits))


def _unpack_first_dim(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack int32 words along the first dimension (GPTQ qweight layout)."""
    if bits not in (2, 4, 8):
        raise ValueError(f"unsupported packed bit width: {bits}")
    shifts = torch.arange(0, 32, bits, dtype=torch.int64)
    values = (
        packed.to(torch.int64).unsqueeze(1) >> shifts.view(1, -1, 1)
    ) & _mask(bits)
    return values.reshape(packed.shape[0] * (32 // bits), packed.shape[1])


def dequantize_gptq(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    *,
    bits: int,
    group_size: int,
    g_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``[out_features, in_features]`` from a GPTQ packed layer."""
    if qweight.ndim != 2 or scales.ndim != 2:
        raise ValueError("GPTQ qweight/scales must be 2-D tensors")
    in_features = int(qweight.shape[0]) * (32 // bits)
    out_features = int(qweight.shape[1])
    groups = (in_features + group_size - 1) // group_size

    values = _unpack_first_dim(qweight, bits).to(torch.float32)
    if values.shape != (in_features, out_features):
        raise ValueError(
            f"GPTQ unpack produced {tuple(values.shape)}, expected "
            f"{(in_features, out_features)}"
        )

    if qzeros is None:
        zeros = torch.full(
            (groups, out_features), float(1 << (bits - 1)), dtype=torch.float32
        )
    else:
        if qzeros.ndim != 2:
            raise ValueError("GPTQ qzeros must be a 2-D tensor")
        zeros = (_unpack_last_dim(qzeros, bits).to(torch.float32) + 1.0)
        if zeros.shape != (int(qzeros.shape[0]), out_features):
            raise ValueError(
                f"GPTQ qzeros unpack produced {tuple(zeros.shape)}, expected "
                f"{(int(qzeros.shape[0]), out_features)}"
            )
        groups = int(qzeros.shape[0])

    if scales.shape != (groups, out_features):
        raise ValueError(
            f"GPTQ scales shape {tuple(scales.shape)} does not match "
            f"{(groups, out_features)}"
        )
    if g_idx is None:
        g_idx = torch.arange(in_features, dtype=torch.long) // int(group_size)
    else:
        g_idx = g_idx.reshape(-1).to(torch.long)
        if g_idx.numel() != in_features:
            raise ValueError(
                f"GPTQ g_idx has {g_idx.numel()} entries, expected {in_features}"
            )
        if int(g_idx.min()) < 0 or int(g_idx.max()) >= groups:
            raise ValueError("GPTQ g_idx contains an out-of-range group index")

    weights = scales.float()[g_idx] * (values - zeros.float()[g_idx])
    return weights.transpose(0, 1).contiguous()


def _reverse_awq_order(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] % 8:
        raise ValueError("AWQ output dimension must be divisible by 8")
    view = values.reshape(*values.shape[:-1], values.shape[-1] // 8, 8)
    order = torch.tensor(AWQ_REVERSE_ORDER, dtype=torch.long, device=values.device)
    return view.index_select(-1, order).reshape(*values.shape)


def dequantize_awq(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None,
    *,
    bits: int = 4,
    group_size: int = 128,
    g_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``[out_features, in_features]`` from an AutoAWQ GEMM layer."""
    if bits != 4:
        raise NotImplementedError(
            f"the AutoAWQ GEMM importer currently supports 4-bit only, got {bits}"
        )
    if qweight.ndim != 2 or scales.ndim != 2:
        raise ValueError("AWQ qweight/scales must be 2-D tensors")
    in_features = int(qweight.shape[0])
    out_features = int(qweight.shape[1]) * (32 // bits)

    values = _reverse_awq_order(
        _unpack_last_dim(qweight, bits).to(torch.float32)
    )
    if values.shape != (in_features, out_features):
        raise ValueError(
            f"AWQ unpack produced {tuple(values.shape)}, expected "
            f"{(in_features, out_features)}"
        )

    if qzeros is None:
        zeros = torch.full(
            ((in_features + group_size - 1) // group_size, out_features),
            8.0,
            dtype=torch.float32,
        )
    else:
        zeros = _reverse_awq_order(
            _unpack_last_dim(qzeros, bits).to(torch.float32)
        )
        expected = (int(qzeros.shape[0]), out_features)
        if zeros.shape != expected:
            raise ValueError(
                f"AWQ qzeros unpack produced {tuple(zeros.shape)}, expected {expected}"
            )
    groups = int(zeros.shape[0])
    if scales.shape != (groups, out_features):
        raise ValueError(
            f"AWQ scales shape {tuple(scales.shape)} does not match "
            f"{(groups, out_features)}"
        )
    if g_idx is None:
        g_idx = torch.arange(in_features, dtype=torch.long) // int(group_size)
    else:
        g_idx = g_idx.reshape(-1).to(torch.long)
        if g_idx.numel() != in_features:
            raise ValueError(
                f"AWQ g_idx has {g_idx.numel()} entries, expected {in_features}"
            )
        if int(g_idx.min()) < 0 or int(g_idx.max()) >= groups:
            raise ValueError("AWQ g_idx contains an out-of-range group index")

    weights = (values - zeros.float()[g_idx]) * scales.float()[g_idx]
    return weights.transpose(0, 1).contiguous()


__all__ = [
    "dequantize_awq",
    "dequantize_gptq",
]
