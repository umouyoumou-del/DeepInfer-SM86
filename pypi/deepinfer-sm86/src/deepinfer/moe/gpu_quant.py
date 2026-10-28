"""GPU-side dequantization for the GGUF formats used by DeepSeek-V2-Lite.

DeepSeek-V2-Lite Q4_K_M uses Q4_K for the routed gate/up projections and Q8_0
for the routed down projection.  Keeping those tensors packed while they cross
PCIe is essential: an expert is only ~6.3 MiB packed but ~17.3 MiB in FP16.
The kernels here are implemented with vectorized PyTorch ops so they work on
SM86 without NVRTC or a C++ toolchain.
"""

from __future__ import annotations

import os
from typing import Any

import torch

try:
    import cupy as cp
except Exception:  # pragma: no cover - exercised only on CPU-only hosts
    cp = None


_CUDA_DEQUANT_SOURCE = r"""
#include <cuda_fp16.h>

extern "C" __global__ void dequant_q4_k(
    const unsigned char* __restrict__ packed,
    __half* __restrict__ out)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const unsigned char* block = packed + (size_t)bid * 144;
    const float d = __half2float(*reinterpret_cast<const __half*>(block));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
    const unsigned char* scales = block + 4;
    const unsigned char* qs = block + 16;

    const int sub = tid >> 5;
    const int pos = tid & 31;
    unsigned int scale;
    unsigned int minimum;
    if (sub < 4) {
        scale = scales[sub] & 0x3F;
        minimum = scales[sub + 4] & 0x3F;
    } else {
        scale = (scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 2) & 0x30);
        minimum = (scales[sub + 4] >> 4) | ((scales[sub] >> 2) & 0x30);
    }
    const unsigned char packed_byte = qs[(sub >> 1) * 32 + pos];
    const int q = (sub & 1) ? (packed_byte >> 4) : (packed_byte & 0x0F);
    const float value = d * (float)scale * (float)q - dmin * (float)minimum;
    out[(size_t)bid * 256 + tid] = __float2half_rn(value);
}

extern "C" __global__ void dequant_q5_0(
    const unsigned char* __restrict__ packed,
    __half* __restrict__ out)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const unsigned char* block = packed + (size_t)bid * 22;
    const float d = __half2float(*reinterpret_cast<const __half*>(block));
    const unsigned int qh =
        (unsigned int)block[2] |
        ((unsigned int)block[3] << 8) |
        ((unsigned int)block[4] << 16) |
        ((unsigned int)block[5] << 24);
    const unsigned char qs = block[6 + (tid & 15)];
    const int ql = (tid < 16) ? (qs & 0x0F) : (qs >> 4);
    const int high = (qh >> tid) & 1;
    const int q = ql | (high << 4);
    out[(size_t)bid * 32 + tid] = __float2half_rn(d * (float)(q - 16));
}

extern "C" __global__ void dequant_q8_0(
    const unsigned char* __restrict__ packed,
    __half* __restrict__ out)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const unsigned char* block = packed + (size_t)bid * 34;
    const float d = __half2float(*reinterpret_cast<const __half*>(block));
    const int q = (int)((signed char)block[2 + tid]);
    out[(size_t)bid * 32 + tid] = __float2half_rn(d * (float)q);
}
"""

_RAW_KERNELS: dict[str, Any] = {}
_CUDA_DEQUANT_DISABLED = os.environ.get("DEEPINFER_DISABLE_CUDA_DEQUANT") == "1"


def _to_cupy(tensor: torch.Tensor) -> Any:
    """Zero-copy CuPy view; the PyTorch tensor stays the owner."""

    if cp is None:
        raise RuntimeError("CuPy is unavailable")
    contiguous = tensor.contiguous()
    dtype = {
        torch.uint8: cp.uint8,
        torch.int32: cp.int32,
        torch.int64: cp.int64,
        torch.float16: cp.float16,
        torch.float32: cp.float32,
    }[contiguous.dtype]
    memory = cp.cuda.UnownedMemory(
        contiguous.data_ptr(),
        contiguous.numel() * contiguous.element_size(),
        owner=contiguous,
        device_id=contiguous.device.index or 0,
    )
    return cp.ndarray(
        contiguous.shape,
        dtype=dtype,
        memptr=cp.cuda.MemoryPointer(memory, 0),
    )


def _raw_kernel(name: str) -> Any:
    if cp is None:
        raise RuntimeError("CuPy is unavailable")
    kernel = _RAW_KERNELS.get(name)
    if kernel is None:
        kernel = cp.RawKernel(
            _CUDA_DEQUANT_SOURCE,
            name,
            options=("--std=c++14", "--use_fast_math"),
        )
        _RAW_KERNELS[name] = kernel
    return kernel


def _dequantize_cuda_kernel(
    packed: torch.Tensor,
    quant_type: str,
    *,
    out_features: int,
    in_features: int,
) -> torch.Tensor | None:
    if _CUDA_DEQUANT_DISABLED or cp is None or not packed.is_cuda:
        return None
    name = quant_type.upper()
    if name == "Q4_K":
        block_size, block_bytes = 256, 144
        threads = 256
        kernel_name = "dequant_q4_k"
    elif name == "Q5_0":
        block_size, block_bytes = 32, 22
        threads = 32
        kernel_name = "dequant_q5_0"
    elif name == "Q8_0":
        block_size, block_bytes = 32, 34
        threads = 32
        kernel_name = "dequant_q8_0"
    else:
        return None
    expected_elements = out_features * in_features
    if packed.numel() % block_bytes:
        raise ValueError(f"{name} input is not block aligned")
    n_blocks = packed.numel() // block_bytes
    if n_blocks * block_size != expected_elements:
        raise ValueError(
            f"{name} block count mismatch: {n_blocks * block_size} values, "
            f"expected {expected_elements}"
        )
    output = torch.empty(
        (out_features, in_features),
        dtype=torch.float16,
        device=packed.device,
    )
    kernel = _raw_kernel(kernel_name)
    kernel(
        (n_blocks,),
        (threads,),
        (_to_cupy(packed), _to_cupy(output)),
    )
    return output



def _fp16_from_le_bytes(values: torch.Tensor) -> torch.Tensor:
    """Reinterpret the last dimension of two little-endian bytes as FP16."""

    if values.shape[-1] != 2:
        raise ValueError("expected two bytes per FP16 value")
    low = values[..., 0].to(torch.int16)
    high = values[..., 1].to(torch.int16)
    packed = low | (high << 8)
    return packed.contiguous().view(torch.float16)


def dequantize_q8_0(
    packed: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize a GGML Q8_0 matrix on its current device.

    Q8_0 is 32 values per block: one FP16 scale followed by 32 signed int8
    values.  ``packed`` must have shape ``(out_features,
    in_features // 32 * 34)`` or an equivalent flattened byte count.
    """

    block_size, type_size = 32, 34
    expected_elements = out_features * in_features
    expected_bytes = expected_elements // block_size * type_size
    if packed.numel() != expected_bytes:
        raise ValueError(
            f"Q8_0 byte count mismatch: got {packed.numel()}, "
            f"expected {expected_bytes} for ({out_features}, {in_features})"
        )
    blocks = packed.reshape(-1, type_size)
    scale = _fp16_from_le_bytes(blocks[:, 0:2]).to(torch.float32)
    quantized = blocks[:, 2:34].view(torch.int8).to(torch.float32)
    values = scale.unsqueeze(-1) * quantized
    return values.reshape(out_features, in_features).to(dtype)


def _q4_k_scales_and_mins(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack Q4_K's 6-bit scales and mins.

    ``scales`` is a uint8 tensor with shape ``(blocks, 12)``.  The layout is
    documented in ``gguf.quants.Q4_K.get_scale_min``.  The returned tensors
    have shape ``(blocks, 8)``.
    """

    scales = scales.to(torch.int32)
    d = scales[:, 0:4]
    m = scales[:, 4:8]
    m_d = scales[:, 8:12]
    scale = torch.cat(
        [d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)],
        dim=-1,
    )
    minimum = torch.cat(
        [m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)],
        dim=-1,
    )
    return scale, minimum


def dequantize_q4_k(
    packed: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize a GGML Q4_K matrix on its current device.

    Each 256-value super-block is 144 bytes: two FP16 scales (``d`` and
    ``dmin``), 12 packed scale/min bytes, and 128 nibble bytes.  This is the
    hot path for routed experts, so temporaries stay in FP32 and only the
    final expert matrix is converted to the requested dtype.
    """

    block_size, type_size = 256, 144
    expected_elements = out_features * in_features
    expected_bytes = expected_elements // block_size * type_size
    if packed.numel() != expected_bytes:
        raise ValueError(
            f"Q4_K byte count mismatch: got {packed.numel()}, "
            f"expected {expected_bytes} for ({out_features}, {in_features})"
        )
    blocks = packed.reshape(-1, type_size)
    d = _fp16_from_le_bytes(blocks[:, 0:2]).to(torch.float32)
    dmin = _fp16_from_le_bytes(blocks[:, 2:4]).to(torch.float32)
    scale, minimum = _q4_k_scales_and_mins(blocks[:, 4:16])
    qs = blocks[:, 16:144].to(torch.int32).reshape(-1, 4, 32)

    low = qs & 0x0F
    high = (qs >> 4) & 0x0F
    scale_f = scale.to(torch.float32).unsqueeze(-1)
    minimum_f = minimum.to(torch.float32).unsqueeze(-1)
    d_col = d.unsqueeze(-1)
    dmin_col = dmin.unsqueeze(-1)

    # Sub-block 2*i comes from the low nibble, 2*i+1 from the high nibble.
    # Each sub-block covers 32 contiguous input values.
    sub_blocks = torch.empty(
        (blocks.shape[0], 8, 32),
        device=packed.device,
        dtype=torch.float32,
    )
    for index in range(4):
        sub_blocks[:, index * 2, :] = (
            d_col * scale_f[:, index * 2, :] * low[:, index, :].to(torch.float32)
            - dmin_col * minimum_f[:, index * 2, :]
        )
        sub_blocks[:, index * 2 + 1, :] = (
            d_col * scale_f[:, index * 2 + 1, :] * high[:, index, :].to(torch.float32)
            - dmin_col * minimum_f[:, index * 2 + 1, :]
        )
    return sub_blocks.reshape(out_features, in_features).to(dtype)


def dequantize_q5_0(
    packed: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize a GGML Q5_0 matrix on its current device.

    Q5_0 is 32 values per 22-byte block: FP16 scale, a little-endian 32-bit
    high-bit mask, and 16 bytes of packed low nibbles.
    """

    block_size, type_size = 32, 22
    expected_elements = out_features * in_features
    expected_bytes = expected_elements // block_size * type_size
    if packed.numel() != expected_bytes:
        raise ValueError(
            f"Q5_0 byte count mismatch: got {packed.numel()}, "
            f"expected {expected_bytes} for ({out_features}, {in_features})"
        )
    blocks = packed.reshape(-1, type_size)
    scale = _fp16_from_le_bytes(blocks[:, 0:2]).to(torch.float32)
    qh_bytes = blocks[:, 2:6].to(torch.int64)
    high_mask = (
        qh_bytes[:, 0]
        | (qh_bytes[:, 1] << 8)
        | (qh_bytes[:, 2] << 16)
        | (qh_bytes[:, 3] << 24)
    )
    shifts = torch.arange(32, device=packed.device, dtype=torch.int64)
    high_bits = ((high_mask.unsqueeze(-1) >> shifts) & 1).to(torch.int32)
    packed_lows = blocks[:, 6:22].to(torch.int32)
    low_values = (packed_lows & 0x0F) | (high_bits[:, 0:16] << 4)
    high_values = (packed_lows >> 4) | (high_bits[:, 16:32] << 4)
    values = torch.cat([low_values, high_values], dim=-1) - 16
    values = scale.unsqueeze(-1) * values.to(torch.float32)
    return values.reshape(out_features, in_features).to(dtype)


def dequantize_matrix(
    packed: torch.Tensor,
    quant_type: str,
    *,
    out_features: int,
    in_features: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dispatch a packed expert matrix to the SM86 dequantizer."""

    name = quant_type.upper()
    if dtype == torch.float16:
        accelerated = _dequantize_cuda_kernel(
            packed,
            name,
            out_features=out_features,
            in_features=in_features,
        )
        if accelerated is not None:
            return accelerated
    if name == "Q4_K":
        return dequantize_q4_k(
            packed,
            out_features=out_features,
            in_features=in_features,
            dtype=dtype,
        )
    if name == "Q8_0":
        return dequantize_q8_0(
            packed,
            out_features=out_features,
            in_features=in_features,
            dtype=dtype,
        )
    if name == "Q5_0":
        return dequantize_q5_0(
            packed,
            out_features=out_features,
            in_features=in_features,
            dtype=dtype,
        )
    raise NotImplementedError(f"GPU dequantization for {quant_type} is not implemented")
