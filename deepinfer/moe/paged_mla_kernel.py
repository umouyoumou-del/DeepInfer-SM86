"""Fused FP16 CUDA kernel for DeepSeek MLA paged attention.

The kernel assigns one warp to one (batch, query-head, query-token) tuple.
Each warp walks the block table, computes Q.K over the 192-wide MLA key,
updates an online softmax scalar, and accumulates the first 128 value dims.
No dense K/V buffer is materialized.
"""

from __future__ import annotations

from typing import Any

import torch

from deepinfer.moe.gpu_quant import _to_cupy

try:
    import cupy as cp
except Exception:  # pragma: no cover - CPU-only hosts
    cp = None


_KERNEL: Any = None

_SOURCE = r"""
#include <cuda_fp16.h>

#define DP_MLA_D 192
#define DP_MLA_V 128
#define DP_MLA_WARPS 4

extern "C" __global__ void dp_paged_mla_attend(
    const __half* __restrict__ q,
    const __half* __restrict__ k_pool,
    const __half* __restrict__ v_pool,
    const long long* __restrict__ block_table,
    const long long* __restrict__ valid_lengths,
    const long long* __restrict__ positions,
    __half* __restrict__ out,
    int total_warps,
    int heads,
    int q_len,
    int block_size,
    int max_blocks,
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
    if (limit <= 0) {
        __half* op = out + (((long long)batch * heads + head) * q_len + qi) * DP_MLA_V;
        for (int d = lane; d < DP_MLA_V; d += 32) op[d] = __float2half_rn(0.0f);
        return;
    }

    const __half* qp =
        q + (((long long)batch * heads + head) * q_len + qi) * DP_MLA_D;
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
        const __half* kp =
            k_pool + (((block * heads + head) * block_size + offset) * DP_MLA_D);
        float dot = 0.0f;
        for (int d = lane; d < DP_MLA_D; d += 32) {
            dot += __half2float(qp[d]) * __half2float(kp[d]);
        }
        for (int delta = 16; delta > 0; delta >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, delta);
        }
        const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;
        const float new_max = fmaxf(running_max, score);
        const float alpha = __expf(running_max - new_max);
        const float probability = __expf(score - new_max);
        running_sum = running_sum * alpha + probability;
        const __half* vp =
            v_pool + (((block * heads + head) * block_size + offset) * DP_MLA_D);
        const float p = probability;
        acc0 = acc0 * alpha + p * __half2float(vp[lane]);
        if (lane + 32 < DP_MLA_V) {
            acc1 = acc1 * alpha + p * __half2float(vp[lane + 32]);
        }
        if (lane + 64 < DP_MLA_V) {
            acc2 = acc2 * alpha + p * __half2float(vp[lane + 64]);
        }
        if (lane + 96 < DP_MLA_V) {
            acc3 = acc3 * alpha + p * __half2float(vp[lane + 96]);
        }
        running_max = new_max;
    }

    const float inv = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    __half* op =
        out + (((long long)batch * heads + head) * q_len + qi) * DP_MLA_V;
    op[lane] = __float2half_rn(acc0 * inv);
    if (lane + 32 < DP_MLA_V) op[lane + 32] = __float2half_rn(acc1 * inv);
    if (lane + 64 < DP_MLA_V) op[lane + 64] = __float2half_rn(acc2 * inv);
    if (lane + 96 < DP_MLA_V) op[lane + 96] = __float2half_rn(acc3 * inv);
}
"""


def _kernel() -> Any:
    global _KERNEL
    if cp is None:
        raise RuntimeError("CuPy is unavailable")
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(
            _SOURCE,
            "dp_paged_mla_attend",
            options=("--std=c++14", "--use_fast_math"),
        )
    return _KERNEL


def cuda_paged_mla_supported(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
) -> bool:
    return (
        cp is not None
        and query.is_cuda
        and query.dtype == torch.float16
        and key_pool.dtype == torch.float16
        and value_pool.dtype == torch.float16
        and query.shape[-1] == 192
        and key_pool.shape[-1] == 192
        and value_pool.shape[-1] == 192
    )


def paged_mla_attend(
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
) -> torch.Tensor:
    """Return [B, H, Q, 128] FP16 attention output."""

    if not cuda_paged_mla_supported(query, key_pool, value_pool):
        raise RuntimeError("fused paged MLA kernel requires FP16 MLA tensors")
    if query.ndim != 4:
        raise ValueError("query must be [B,H,Q,192]")
    batch, heads, q_len, _ = query.shape
    if q_len < 1 or heads < 1 or batch < 1:
        raise ValueError("invalid query shape")
    if block_table.ndim != 2 or block_table.shape[0] != batch:
        raise ValueError("block_table must be [B,max_blocks]")
    if valid_lengths.numel() != batch or positions.numel() != batch * q_len:
        raise ValueError("valid_lengths/positions shape mismatch")
    query = query.contiguous()
    key_pool = key_pool.contiguous()
    value_pool = value_pool.contiguous()
    block_table = block_table.to(torch.int64).contiguous()
    valid_lengths = valid_lengths.to(torch.int64).contiguous()
    positions = positions.to(torch.int64).contiguous()
    output = torch.empty(
        (batch, heads, q_len, 128),
        dtype=torch.float16,
        device=query.device,
    )
    total_warps = batch * heads * q_len
    grid = ((total_warps + 3) // 4,)
    _kernel()(
        grid,
        (128,),
        (
            _to_cupy(query),
            _to_cupy(key_pool),
            _to_cupy(value_pool),
            _to_cupy(block_table),
            _to_cupy(valid_lengths),
            _to_cupy(positions),
            _to_cupy(output),
            int(total_warps),
            int(heads),
            int(q_len),
            int(block_size),
            int(max_blocks),
            cp.float32(scale),
        ),
    )
    return output


__all__ = ["cuda_paged_mla_supported", "paged_mla_attend"]
