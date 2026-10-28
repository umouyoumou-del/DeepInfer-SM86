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
        __half* op =
            out + (((long long)batch * heads + head) * q_len + qi) * DP_MLA_V;
        __half2 zero = __float2half2_rn(0.0f);
        *reinterpret_cast<__half2*>(op + lane * 2) = zero;
        *reinterpret_cast<__half2*>(op + 64 + lane * 2) = zero;
        return;
    }

    const __half* qp =
        q + (((long long)batch * heads + head) * q_len + qi) * DP_MLA_D;
    // 192 dims / (32 lanes * 2 half values) = 3 vector loads.
    const float2 qv0 = __half22float2(
        *reinterpret_cast<const __half2*>(qp + lane * 2));
    const float2 qv1 = __half22float2(
        *reinterpret_cast<const __half2*>(qp + 64 + lane * 2));
    const float2 qv2 = __half22float2(
        *reinterpret_cast<const __half2*>(qp + 128 + lane * 2));

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
        const float2 kv0 = __half22float2(
            *reinterpret_cast<const __half2*>(kp + lane * 2));
        const float2 kv1 = __half22float2(
            *reinterpret_cast<const __half2*>(kp + 64 + lane * 2));
        const float2 kv2 = __half22float2(
            *reinterpret_cast<const __half2*>(kp + 128 + lane * 2));
        float dot = qv0.x * kv0.x + qv0.y * kv0.y;
        dot += qv1.x * kv1.x + qv1.y * kv1.y;
        dot += qv2.x * kv2.x + qv2.y * kv2.y;
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
        const float2 vv0 = __half22float2(
            *reinterpret_cast<const __half2*>(vp + lane * 2));
        const float2 vv1 = __half22float2(
            *reinterpret_cast<const __half2*>(vp + 64 + lane * 2));
        acc0 = acc0 * alpha + probability * vv0.x;
        acc1 = acc1 * alpha + probability * vv0.y;
        acc2 = acc2 * alpha + probability * vv1.x;
        acc3 = acc3 * alpha + probability * vv1.y;
        running_max = new_max;
    }

    const float inv = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    __half* op =
        out + (((long long)batch * heads + head) * q_len + qi) * DP_MLA_V;
    *reinterpret_cast<__half2*>(op + lane * 2) =
        __floats2half2_rn(acc0 * inv, acc1 * inv);
    *reinterpret_cast<__half2*>(op + 64 + lane * 2) =
        __floats2half2_rn(acc2 * inv, acc3 * inv);
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


_CTA_SOURCE = r"""
#include <cuda_fp16.h>

#define DP_MLA_D 192
#define DP_MLA_V 128
#define DP_MLA_WARPS 4

extern "C" __global__ void dp_paged_mla_attend_cta(
    const __half* __restrict__ q,
    const __half* __restrict__ k_pool,
    const __half* __restrict__ v_pool,
    const long long* __restrict__ block_table,
    const long long* __restrict__ valid_lengths,
    const long long* __restrict__ positions,
    __half* __restrict__ out,
    int total_queries,
    int heads,
    int q_len,
    int block_size,
    int max_blocks,
    float scale)
{
    const int query_index = blockIdx.x;
    if (query_index >= total_queries) return;
    const int qi = query_index % q_len;
    const int head = (query_index / q_len) % heads;
    const int batch = query_index / (q_len * heads);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tid = threadIdx.x;

    const long long valid = valid_lengths[batch];
    const long long qpos = positions[(long long)batch * q_len + qi];
    const long long limit = valid < qpos + 1 ? valid : qpos + 1;
    const __half* qp =
        q + (((long long)batch * heads + head) * q_len + qi) * DP_MLA_D;

    float q0 = __half2float(qp[lane]);
    float q1 = __half2float(qp[lane + 32]);
    float q2 = __half2float(qp[lane + 64]);
    float q3 = __half2float(qp[lane + 96]);
    float q4 = __half2float(qp[lane + 128]);
    float q5 = __half2float(qp[lane + 160]);

    float running_max = -1.0e30f;
    float running_sum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (long long token = warp; token < limit; token += DP_MLA_WARPS) {
        const long long block_index = token / block_size;
        const long long offset = token - block_index * block_size;
        const long long block =
            block_table[(long long)batch * max_blocks + block_index];
        if (block < 0) continue;
        const __half* kp =
            k_pool + (((block * heads + head) * block_size + offset) * DP_MLA_D);
        float dot = q0 * __half2float(kp[lane]);
        dot += q1 * __half2float(kp[lane + 32]);
        dot += q2 * __half2float(kp[lane + 64]);
        dot += q3 * __half2float(kp[lane + 96]);
        dot += q4 * __half2float(kp[lane + 128]);
        dot += q5 * __half2float(kp[lane + 160]);
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
        acc0 = acc0 * alpha + probability * __half2float(vp[lane]);
        acc1 = acc1 * alpha + probability * __half2float(vp[lane + 32]);
        acc2 = acc2 * alpha + probability * __half2float(vp[lane + 64]);
        acc3 = acc3 * alpha + probability * __half2float(vp[lane + 96]);
        running_max = new_max;
    }

    __shared__ float sh_max[DP_MLA_WARPS];
    __shared__ float sh_sum[DP_MLA_WARPS];
    __shared__ float sh_acc[DP_MLA_WARPS][DP_MLA_V];
    if (lane == 0) {
        sh_max[warp] = running_max;
        sh_sum[warp] = running_sum;
    }
    sh_acc[warp][lane] = acc0;
    sh_acc[warp][lane + 32] = acc1;
    sh_acc[warp][lane + 64] = acc2;
    sh_acc[warp][lane + 96] = acc3;
    __syncthreads();

    float global_max = -1.0e30f;
    for (int w = 0; w < DP_MLA_WARPS; ++w) {
        global_max = fmaxf(global_max, sh_max[w]);
    }
    float denominator = 0.0f;
    for (int w = 0; w < DP_MLA_WARPS; ++w) {
        denominator += sh_sum[w] * __expf(sh_max[w] - global_max);
    }
    float numerator = 0.0f;
    for (int w = 0; w < DP_MLA_WARPS; ++w) {
        numerator += sh_acc[w][tid] * __expf(sh_max[w] - global_max);
    }
    const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    __half* op =
        out + (((long long)batch * heads + head) * q_len + qi) * DP_MLA_V;
    op[tid] = __float2half_rn(numerator * inv);
}
"""

_CTA_KERNEL: Any = None


def _cta_kernel() -> Any:
    global _CTA_KERNEL
    if cp is None:
        raise RuntimeError("CuPy is unavailable")
    if _CTA_KERNEL is None:
        _CTA_KERNEL = cp.RawKernel(
            _CTA_SOURCE,
            "dp_paged_mla_attend_cta",
            options=("--std=c++14", "--use_fast_math"),
        )
    return _CTA_KERNEL


def paged_mla_attend_cta(
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
    """CTA-per-query variant: four warps split the key range."""

    if not cuda_paged_mla_supported(query, key_pool, value_pool):
        raise RuntimeError("fused paged MLA kernel requires FP16 MLA tensors")
    if query.ndim != 4:
        raise ValueError("query must be [B,H,Q,192]")
    batch, heads, q_len, _ = query.shape
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
    total_queries = batch * heads * q_len
    _cta_kernel()(
        (total_queries,),
        (128,),
        (
            _to_cupy(query),
            _to_cupy(key_pool),
            _to_cupy(value_pool),
            _to_cupy(block_table),
            _to_cupy(valid_lengths),
            _to_cupy(positions),
            _to_cupy(output),
            int(total_queries),
            int(heads),
            int(q_len),
            int(block_size),
            int(max_blocks),
            cp.float32(scale),
        ),
    )
    return output


_SCALAR_SOURCE = r"""
#include <cuda_fp16.h>

#define DP_MLA_D 192
#define DP_MLA_V 128
#define DP_MLA_WARPS 4

extern "C" __global__ void dp_paged_mla_attend_scalar(
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
    const __half* qp =
        q + (((long long)batch * heads + head) * q_len + qi) * DP_MLA_D;
    const float q0 = __half2float(qp[lane]);
    const float q1 = __half2float(qp[lane + 32]);
    const float q2 = __half2float(qp[lane + 64]);
    const float q3 = __half2float(qp[lane + 96]);
    const float q4 = __half2float(qp[lane + 128]);
    const float q5 = __half2float(qp[lane + 160]);
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
        float dot = q0 * __half2float(kp[lane]);
        dot += q1 * __half2float(kp[lane + 32]);
        dot += q2 * __half2float(kp[lane + 64]);
        dot += q3 * __half2float(kp[lane + 96]);
        dot += q4 * __half2float(kp[lane + 128]);
        dot += q5 * __half2float(kp[lane + 160]);
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
        acc0 = acc0 * alpha + probability * __half2float(vp[lane]);
        acc1 = acc1 * alpha + probability * __half2float(vp[lane + 32]);
        acc2 = acc2 * alpha + probability * __half2float(vp[lane + 64]);
        acc3 = acc3 * alpha + probability * __half2float(vp[lane + 96]);
        running_max = new_max;
    }
    const float inv = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    __half* op =
        out + (((long long)batch * heads + head) * q_len + qi) * DP_MLA_V;
    op[lane] = __float2half_rn(acc0 * inv);
    op[lane + 32] = __float2half_rn(acc1 * inv);
    op[lane + 64] = __float2half_rn(acc2 * inv);
    op[lane + 96] = __float2half_rn(acc3 * inv);
}
"""

_SCALAR_KERNEL: Any = None


def _scalar_kernel() -> Any:
    global _SCALAR_KERNEL
    if cp is None:
        raise RuntimeError("CuPy is unavailable")
    if _SCALAR_KERNEL is None:
        _SCALAR_KERNEL = cp.RawKernel(
            _SCALAR_SOURCE,
            "dp_paged_mla_attend_scalar",
            options=("--std=c++14", "--use_fast_math"),
        )
    return _SCALAR_KERNEL


def paged_mla_attend_scalar(
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
    if not cuda_paged_mla_supported(query, key_pool, value_pool):
        raise RuntimeError("fused paged MLA kernel requires FP16 MLA tensors")
    if query.ndim != 4:
        raise ValueError("query must be [B,H,Q,192]")
    batch, heads, q_len, _ = query.shape
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
    _scalar_kernel()(
        ((total_warps + 3) // 4,),
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


__all__ = [
    "cuda_paged_mla_supported",
    "paged_mla_attend",
    "paged_mla_attend_cta",
    "paged_mla_attend_scalar",
]
