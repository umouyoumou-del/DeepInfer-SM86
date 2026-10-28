"""Native paged MLA attention for INT8/FP8 KV storage.

The pool stores K/V padded to the MLA head width (192).  INT8 uses one FP16
scale per (block, head, token) for K and V.  FP8 uses direct E4M3/E5M2 bytes.
This kernel reads the quantized block table directly; it never materializes a
dense K/V buffer.
"""

from __future__ import annotations

from typing import Any

import torch

from deepinfer.moe.gpu_quant import _to_cupy

try:
    import cupy as cp
except Exception:  # pragma: no cover - CPU-only hosts
    cp = None


_QUANT_KERNEL: Any = None

_QUANT_SOURCE = r"""
#include <cuda_fp16.h>

#define DP_MLA_D 192
#define DP_MLA_V 128
#define DP_MLA_WARPS 4

__device__ __forceinline__ float dp_fp8_e4m3(unsigned char bits) {
    const float sign = (bits & 0x80u) ? -1.0f : 1.0f;
    const int exp = (bits >> 3) & 0x0f;
    const int mant = bits & 0x07;
    if (exp == 0) {
        return sign * ((float)mant / 8.0f) * (1.0f / 64.0f);
    }
    if (exp == 15 && mant == 7) {
        return nanf("");
    }
    return sign * (1.0f + (float)mant / 8.0f) * exp2f((float)exp - 7.0f);
}

__device__ __forceinline__ float dp_fp8_e5m2(unsigned char bits) {
    const float sign = (bits & 0x80u) ? -1.0f : 1.0f;
    const int exp = (bits >> 2) & 0x1f;
    const int mant = bits & 0x03;
    if (exp == 0) {
        return sign * ((float)mant / 4.0f) * (1.0f / 16384.0f);
    }
    if (exp == 31) {
        return mant == 0 ? sign * (1.0f / 0.0f) : nanf("");
    }
    return sign * (1.0f + (float)mant / 4.0f) * exp2f((float)exp - 15.0f);
}

__device__ __forceinline__ float dp_quant_value(
    unsigned char bits, int mode)
{
    if (mode == 0) {
        return (float)(signed char)bits;
    }
    if (mode == 1) {
        return dp_fp8_e4m3(bits);
    }
    return dp_fp8_e5m2(bits);
}

extern "C" __global__ void dp_paged_mla_attend_quant(
    const __half* __restrict__ q,
    const unsigned char* __restrict__ k_pool,
    const unsigned char* __restrict__ v_pool,
    const __half* __restrict__ k_scale,
    const __half* __restrict__ v_scale,
    const long long* __restrict__ block_table,
    const long long* __restrict__ valid_lengths,
    const long long* __restrict__ positions,
    __half* __restrict__ out,
    int total_warps,
    int heads,
    int q_len,
    int block_size,
    int max_blocks,
    int pool_dim,
    int key_dim,
    int value_dim,
    int mode,
    float scale)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int index = blockIdx.x * DP_MLA_WARPS + warp;
    if (index >= total_warps) return;
    const int qi = index % q_len;
    const int head = (index / q_len) % heads;
    const int batch = index / (q_len * heads);
    const long long valid = valid_lengths[batch];
    const long long qpos = positions[(long long)batch * q_len + qi];
    const long long limit = valid < qpos + 1 ? valid : qpos + 1;
    __half* op =
        out + (((long long)batch * heads + head) * q_len + qi) * DP_MLA_V;
    const __half* qp =
        q + (((long long)batch * heads + head) * q_len + qi) * pool_dim;
    if (limit <= 0) {
        for (int d = lane; d < DP_MLA_V; d += 32) {
            op[d] = __float2half_rn(0.0f);
        }
        return;
    }

    float running_max = -1.0e30f;
    float running_sum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (long long token = 0; token < limit; ++token) {
        const long long block_index = token / block_size;
        const long long offset = token - block_index * block_size;
        const long long block =
            block_table[(long long)batch * max_blocks + block_index];
        if (block < 0) continue;
        const long long pool_offset =
            ((block * heads + head) * block_size + offset);
        const unsigned char* kp = k_pool + pool_offset * pool_dim;
        float dot = 0.0f;
        for (int d = lane; d < key_dim; d += 32) {
            dot += __half2float(qp[d]) * dp_quant_value(kp[d], mode);
        }
        for (int delta = 16; delta > 0; delta >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, delta);
        }
        if (mode == 0) {
            dot *= __half2float(k_scale[pool_offset]);
        }
        const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;
        const float new_max = fmaxf(running_max, score);
        const float alpha = __expf(running_max - new_max);
        const float probability = __expf(score - new_max);
        running_sum = running_sum * alpha + probability;
        const unsigned char* vp = v_pool + pool_offset * pool_dim;
        const float value_scale =
            mode == 0 ? __half2float(v_scale[pool_offset]) : 1.0f;
        for (int d = lane; d < value_dim; d += 32) {
            const float value = dp_quant_value(vp[d], mode) * value_scale;
            if (d < 32) {
                acc0 = acc0 * alpha + probability * value;
            } else if (d < 64) {
                acc1 = acc1 * alpha + probability * value;
            } else if (d < 96) {
                acc2 = acc2 * alpha + probability * value;
            } else {
                acc3 = acc3 * alpha + probability * value;
            }
        }
        running_max = new_max;
    }
    const float inv = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    for (int d = lane; d < DP_MLA_V; d += 32) {
        float value;
        if (d < 32) {
            value = acc0;
        } else if (d < 64) {
            value = acc1;
        } else if (d < 96) {
            value = acc2;
        } else {
            value = acc3;
        }
        op[d] = __float2half_rn(value * inv);
    }
}
"""


def _kernel() -> Any:
    global _QUANT_KERNEL
    if cp is None:
        raise RuntimeError("CuPy is unavailable")
    if _QUANT_KERNEL is None:
        _QUANT_KERNEL = cp.RawKernel(
            _QUANT_SOURCE,
            "dp_paged_mla_attend_quant",
            options=("-std=c++17",),
        )
    return _QUANT_KERNEL


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        value
        for value in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
        )
        if value is not None
    }


def cuda_paged_mla_quant_supported(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
) -> bool:
    quantized = key_pool.dtype == torch.int8 or _is_fp8_dtype(key_pool.dtype)
    return (
        cp is not None
        and query.is_cuda
        and query.dtype == torch.float16
        and quantized
        and value_pool.dtype == key_pool.dtype
        and query.shape[-1] == 192
        and key_pool.shape[-1] == 192
        and value_pool.shape[-1] == 192
    )


def paged_mla_attend_quantized(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_table: torch.Tensor,
    valid_lengths: torch.Tensor,
    positions: torch.Tensor,
    *,
    block_size: int,
    max_blocks: int,
    scale: float,
    key_scale_pool: torch.Tensor | None = None,
    value_scale_pool: torch.Tensor | None = None,
    key_dim: int = 192,
    value_dim: int = 128,
) -> torch.Tensor:
    """Paged MLA over INT8/FP8 storage without dense dequantization."""

    if not cuda_paged_mla_quant_supported(query, key_pool, value_pool):
        raise RuntimeError("quantized paged MLA requires INT8/FP8 pools")
    if key_pool.dtype == torch.int8:
        if key_scale_pool is None or value_scale_pool is None:
            raise ValueError("INT8 paged MLA requires key/value scale pools")
        mode = 0
    elif key_pool.dtype == getattr(torch, "float8_e4m3fn", None):
        mode = 1
    else:
        mode = 2
    query = query.contiguous()
    raw_key = key_pool.contiguous().view(torch.uint8)
    raw_value = value_pool.contiguous().view(torch.uint8)
    block_table = block_table.to(torch.int64).contiguous()
    valid_lengths = valid_lengths.to(torch.int64).contiguous()
    positions = positions.to(torch.int64).contiguous()
    if key_scale_pool is None:
        key_scale_pool = torch.ones(1, dtype=torch.float16, device=query.device)
    if value_scale_pool is None:
        value_scale_pool = torch.ones(1, dtype=torch.float16, device=query.device)
    batch, heads, q_len, pool_dim = query.shape
    output = torch.empty(
        (batch, heads, q_len, value_dim),
        dtype=torch.float16,
        device=query.device,
    )
    total_warps = batch * heads * q_len
    _kernel()(
        ((total_warps + 3) // 4,),
        (128,),
        (
            _to_cupy(query),
            _to_cupy(raw_key),
            _to_cupy(raw_value),
            _to_cupy(key_scale_pool.contiguous()),
            _to_cupy(value_scale_pool.contiguous()),
            _to_cupy(block_table),
            _to_cupy(valid_lengths),
            _to_cupy(positions),
            _to_cupy(output),
            int(total_warps),
            int(heads),
            int(q_len),
            int(block_size),
            int(max_blocks),
            int(pool_dim),
            int(key_dim),
            int(value_dim),
            int(mode),
            cp.float32(scale),
        ),
    )
    return output


__all__ = [
    "cuda_paged_mla_quant_supported",
    "paged_mla_attend_quantized",
]
