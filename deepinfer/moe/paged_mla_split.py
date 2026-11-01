"""Split-K paged MLA attention for decode (flash-decoding over the block table).

The single-pass kernels in ``paged_mla_kernel`` / ``paged_mla_quant`` give each
(batch, head, query) one warp that walks every key in order.  A decode step has
one query per head, so the whole GPU runs 16 warps; on a 4080 SUPER a 4K window
took 8.5 ms per layer (about 3 GB/s) and 128K full attention 276 ms.  They also
loop from token 0 and ``continue`` past the skipped middle, so a window still
costs a pass over the whole history.

Here the kept keys of each query - the first ``attention_sink`` and the last
``sliding_window``, or all of them - are cut into ``splits`` slices.  One CTA
(4 warps) per (row, slice) runs the online softmax over its slice with the
query held in registers and writes (max, sum, 128-wide accumulator); a second
kernel merges the slices of each row.  Pools are fp16, INT8 (+ per-token
scales) or FP8 E4M3/E5M2, laid out as in ``PagedKVCacheManager``:
``[blocks, heads, block_size, 192]`` with V in the first 128 channels.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from deepinfer.moe.gpu_quant import _to_cupy

try:
    import cupy as cp
except Exception:  # noqa: BLE001 - CuPy can fail on driver/toolkit errors, not only ImportError
    cp = None

_SOURCE = r"""
#include <cuda_fp16.h>

#define SPLIT_WARPS 4
#define V_DIM 128

__device__ __forceinline__ float sp_fp8_e4m3(unsigned char bits) {
    const float sign = (bits & 0x80u) ? -1.0f : 1.0f;
    const int exp = (bits >> 3) & 0x0f;
    const int mant = bits & 0x07;
    if (exp == 0) return sign * ((float)mant / 8.0f) * (1.0f / 64.0f);
    if (exp == 15 && mant == 7) return nanf("");
    return sign * (1.0f + (float)mant / 8.0f) * exp2f((float)exp - 7.0f);
}

__device__ __forceinline__ float sp_fp8_e5m2(unsigned char bits) {
    const float sign = (bits & 0x80u) ? -1.0f : 1.0f;
    const int exp = (bits >> 2) & 0x1f;
    const int mant = bits & 0x03;
    if (exp == 0) return sign * ((float)mant / 4.0f) * (1.0f / 16384.0f);
    if (exp == 31) return mant == 0 ? sign * (1.0f / 0.0f) : nanf("");
    return sign * (1.0f + (float)mant / 4.0f) * exp2f((float)exp - 15.0f);
}

// mode: 0 int8 (scaled), 1 fp8 e4m3, 2 fp8 e5m2, 3 fp16
__device__ __forceinline__ float sp_load(const void* pool, long long i, int mode) {
    if (mode == 3) return __half2float(((const __half*)pool)[i]);
    const unsigned char bits = ((const unsigned char*)pool)[i];
    if (mode == 0) return (float)(signed char)bits;
    if (mode == 1) return sp_fp8_e4m3(bits);
    return sp_fp8_e5m2(bits);
}

extern "C" __global__ void dp_paged_mla_split_partial(
    const __half* __restrict__ q,
    const void* __restrict__ k_pool,
    const void* __restrict__ v_pool,
    const __half* __restrict__ k_scale,
    const __half* __restrict__ v_scale,
    const long long* __restrict__ block_table,
    const long long* __restrict__ valid_lengths,
    const long long* __restrict__ positions,
    float* __restrict__ part,
    int heads,
    int q_len,
    int block_size,
    int max_blocks,
    int pool_dim,
    int key_dim,
    int mode,
    float scale,
    int window_size,
    int sink_size,
    int splits,
    int chunk)
{
    const int split = blockIdx.x;
    const int row = blockIdx.y;  // ((batch * heads) + head) * q_len + qi
    const int qi = row % q_len;
    const int head = (row / q_len) % heads;
    const int batch = row / (q_len * heads);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const long long valid = valid_lengths[batch];
    const long long qpos = positions[(long long)batch * q_len + qi];
    long long limit = valid < qpos + 1 ? valid : qpos + 1;
    if (limit < 0) limit = 0;
    // kept keys: [0, sink_end) and [win_start, limit)
    long long sink_end = limit;
    long long win_start = limit;
    if (window_size > 0) {
        sink_end = sink_size < limit ? sink_size : limit;
        win_start = limit - window_size;
        if (win_start < sink_end) win_start = sink_end;
    }
    const long long kept = sink_end + (limit - win_start);
    const long long begin = (long long)split * chunk;
    long long end = begin + chunk;
    if (end > kept) end = kept;

    float qv[6];
    const __half* qp = q + (long long)row * pool_dim;
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        const int d = lane + 32 * i;
        qv[i] = d < key_dim ? __half2float(qp[d]) : 0.0f;
    }

    float m = -1.0e30f, s = 0.0f, a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long long idx = begin + warp; idx < end; idx += SPLIT_WARPS) {
        const long long token = idx < sink_end ? idx : win_start + (idx - sink_end);
        const long long bi = token / block_size;
        const long long block = block_table[(long long)batch * max_blocks + bi];
        if (block < 0) continue;
        const long long slot = (block * heads + head) * block_size + (token - bi * block_size);
        const long long base = slot * pool_dim;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            const int d = lane + 32 * i;
            if (d < key_dim) dot += qv[i] * sp_load(k_pool, base + d, mode);
        }
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, delta);
        }
        if (mode == 0) dot *= __half2float(k_scale[slot]);
        const float score = dot * scale;
        const float nm = fmaxf(m, score);
        const float alpha = __expf(m - nm);
        const float p = __expf(score - nm);
        const float vs = mode == 0 ? __half2float(v_scale[slot]) : 1.0f;
        s = s * alpha + p;
        a0 = a0 * alpha + p * vs * sp_load(v_pool, base + lane, mode);
        a1 = a1 * alpha + p * vs * sp_load(v_pool, base + lane + 32, mode);
        a2 = a2 * alpha + p * vs * sp_load(v_pool, base + lane + 64, mode);
        a3 = a3 * alpha + p * vs * sp_load(v_pool, base + lane + 96, mode);
        m = nm;
    }

    __shared__ float sh_m[SPLIT_WARPS];
    __shared__ float sh_s[SPLIT_WARPS];
    __shared__ float sh_a[SPLIT_WARPS][V_DIM];
    if (lane == 0) {
        sh_m[warp] = m;
        sh_s[warp] = s;
    }
    sh_a[warp][lane] = a0;
    sh_a[warp][lane + 32] = a1;
    sh_a[warp][lane + 64] = a2;
    sh_a[warp][lane + 96] = a3;
    __syncthreads();
    float mm = -1.0e30f;
#pragma unroll
    for (int w = 0; w < SPLIT_WARPS; ++w) mm = fmaxf(mm, sh_m[w]);
    float* out = part + ((long long)row * splits + split) * (V_DIM + 2);
    const int t = threadIdx.x;
    float acc = 0.0f;
#pragma unroll
    for (int w = 0; w < SPLIT_WARPS; ++w) acc += sh_a[w][t] * __expf(sh_m[w] - mm);
    out[t] = acc;
    if (t == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int w = 0; w < SPLIT_WARPS; ++w) ss += sh_s[w] * __expf(sh_m[w] - mm);
        out[V_DIM] = mm;
        out[V_DIM + 1] = ss;
    }
}

extern "C" __global__ void dp_paged_mla_split_merge(
    const float* __restrict__ part,
    __half* __restrict__ out,
    int splits)
{
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const float* p = part + (long long)row * splits * (V_DIM + 2);
    float mm = -1.0e30f;
    for (int s = 0; s < splits; ++s) mm = fmaxf(mm, p[s * (V_DIM + 2) + V_DIM]);
    float acc = 0.0f, total = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float* ps = p + s * (V_DIM + 2);
        const float w = __expf(ps[V_DIM] - mm);
        acc += ps[t] * w;
        total += ps[V_DIM + 1] * w;
    }
    out[(long long)row * V_DIM + t] = __float2half_rn(total > 0.0f ? acc / total : 0.0f);
}
"""

_MODULE: Any = None
_SM_COUNT: dict[int, int] = {}
# Beyond this many queries per sequence a prefill chunk already fills the GPU
# with the single-pass kernels.
MAX_QUERIES = 16


def _module() -> Any:
    global _MODULE
    if cp is None:
        raise RuntimeError("CuPy is unavailable")
    if _MODULE is None:
        _MODULE = cp.RawModule(code=_SOURCE, options=("-std=c++17",))
    return _MODULE


def _mode(pool: torch.Tensor) -> int | None:
    if pool.dtype == torch.float16:
        return 3
    if pool.dtype == torch.int8:
        return 0
    if pool.dtype == getattr(torch, "float8_e4m3fn", None):
        return 1
    if pool.dtype == getattr(torch, "float8_e5m2", None):
        return 2
    return None


def paged_mla_split_supported(
    query: torch.Tensor, key_pool: torch.Tensor, value_pool: torch.Tensor
) -> bool:
    return (
        cp is not None
        and query.is_cuda
        and query.dtype == torch.float16
        and query.dim() == 4
        and int(query.shape[2]) <= MAX_QUERIES
        and query.shape[-1] == 192
        and key_pool.shape[-1] == 192
        and value_pool.shape[-1] == 192
        and value_pool.dtype == key_pool.dtype
        and _mode(key_pool) is not None
        and key_pool.is_contiguous()
        and value_pool.is_contiguous()
    )


def _sm_count(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SM_COUNT:
        _SM_COUNT[index] = int(
            torch.cuda.get_device_properties(index).multi_processor_count
        )
    return _SM_COUNT[index]


def split_plan(kept_bound: int, rows: int, sm_count: int) -> tuple[int, int]:
    """(splits, chunk): about 8 CTAs per SM, slices of 32-512 keys."""
    kept_bound = max(1, int(kept_bound))
    wanted = math.ceil(8 * sm_count / max(1, rows))
    splits = max(
        1, min(math.ceil(kept_bound / 32), max(math.ceil(kept_bound / 512), wanted))
    )
    return splits, math.ceil(kept_bound / splits)


def paged_mla_decode_split(
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
    sliding_window: int | None = None,
    attention_sink: int | None = None,
    max_kv_length: int | None = None,
) -> torch.Tensor:
    """Attention of a few queries per sequence over a paged pool; returns ``[batch, heads, q_len, 128]`` fp16.

    ``max_kv_length`` bounds how many keys any query can see (the caller knows the
    longest sequence in the step); without it the block-table width is used.
    """

    if value_dim != 128 or key_dim > 192:
        raise ValueError("split paged MLA expects key_dim <= 192 and value_dim == 128")
    mode = _mode(key_pool)
    if mode is None or not paged_mla_split_supported(query, key_pool, value_pool):
        raise RuntimeError("unsupported inputs for split paged MLA")
    if mode == 0 and (key_scale_pool is None or value_scale_pool is None):
        raise ValueError("INT8 paged MLA requires key/value scale pools")
    window = 0 if sliding_window is None else int(sliding_window)
    sink = 0 if attention_sink is None else int(attention_sink)
    if window < 0 or sink < 0:
        raise ValueError("sliding_window and attention_sink must be non-negative")

    batch, heads, q_len, pool_dim = (int(x) for x in query.shape)
    rows = batch * heads * q_len
    bound = int(max_blocks) * int(block_size)
    if max_kv_length is not None:
        bound = min(bound, int(max_kv_length))
    kept_bound = min(bound, sink + window) if window > 0 else bound
    splits, chunk = split_plan(kept_bound, rows, _sm_count(query.device))

    query = query.contiguous()
    if mode == 3:
        raw_key, raw_value = key_pool, value_pool
    else:
        raw_key, raw_value = key_pool.view(torch.uint8), value_pool.view(torch.uint8)
    dummy = None
    if key_scale_pool is None or value_scale_pool is None:
        dummy = torch.ones(1, dtype=torch.float16, device=query.device)
    k_scale = (key_scale_pool if key_scale_pool is not None else dummy).contiguous()
    v_scale = (value_scale_pool if value_scale_pool is not None else dummy).contiguous()
    table = block_table.to(torch.int64).contiguous()
    lengths = valid_lengths.to(torch.int64).contiguous()
    pos = positions.to(torch.int64).contiguous()
    part = torch.empty((rows, splits, 130), dtype=torch.float32, device=query.device)
    output = torch.empty(
        (batch, heads, q_len, 128), dtype=torch.float16, device=query.device
    )

    module = _module()
    module.get_function("dp_paged_mla_split_partial")(
        (splits, rows),
        (128,),
        (
            _to_cupy(query),
            _to_cupy(raw_key),
            _to_cupy(raw_value),
            _to_cupy(k_scale),
            _to_cupy(v_scale),
            _to_cupy(table),
            _to_cupy(lengths),
            _to_cupy(pos),
            _to_cupy(part),
            heads,
            q_len,
            int(block_size),
            int(table.shape[1]),
            pool_dim,
            int(key_dim),
            mode,
            cp.float32(scale),
            window,
            sink,
            splits,
            chunk,
        ),
    )
    module.get_function("dp_paged_mla_split_merge")(
        (rows,),
        (128,),
        (_to_cupy(part), _to_cupy(output), splits),
    )
    return output


__all__ = [
    "MAX_QUERIES",
    "paged_mla_decode_split",
    "paged_mla_split_supported",
    "split_plan",
]
