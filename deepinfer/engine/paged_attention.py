"""Native SM86 paged GQA decode kernel.

This is the decode-only fast path for paged KV blocks.  It reads the block
table directly, so it does not materialize a dense K/V tensor or call SDPA.
Prefill chunks still use the reference SDPA path.
"""

from __future__ import annotations

from typing import Any

import torch

from deepinfer.engine.kv_cache import PagedLayerKVCache

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional on CPU-only installs
    cp = None


_SM86_PAGED_GQA_KERNEL = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define DP_PAGED_MAX_THREADS 1024

__device__ __forceinline__ float dp_warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

extern "C" __global__ void dp_paged_gqa_decode(
    const __half* __restrict__ query,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const long long* __restrict__ block_table,
    const long long* __restrict__ valid_lengths,
    __half* __restrict__ output,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks,
    int max_length,
    int window_size,
    float scale,
    float softcap,
    int use_softcap)
{
    const int head = blockIdx.x;
    const int batch = blockIdx.y;
    const int tid = threadIdx.x;
    const int kv_group = num_heads / num_kv_heads;
    const int kv_head = head / kv_group;
    const int valid_len = (int)valid_lengths[batch];
    const int start = window_size > 0 ? max(0, valid_len - window_size) : 0;

    __shared__ float scores[DP_PAGED_MAX_THREADS];
    __shared__ float probs[DP_PAGED_MAX_THREADS];
    __shared__ float reduce_buf[DP_PAGED_MAX_THREADS];

    float score = -1.0e30f;
    if (tid < valid_len && tid >= start && tid < max_length) {
        const int block_index = tid / block_size;
        const int offset = tid % block_size;
        const long long block_id = block_table[batch * max_blocks + block_index];
        const size_t kv_offset = (
            ((size_t)block_id * num_kv_heads + kv_head) * block_size + offset
        ) * head_dim;
        const __half* kptr = key_cache + kv_offset;
        float dot = 0.0f;
#pragma unroll 4
        for (int d = 0; d < head_dim; ++d) {
            dot += __half2float(query[
                ((size_t)batch * num_heads + head) * head_dim + d
            ]) * __half2float(kptr[d]);
        }
        score = dot * scale;
        if (use_softcap) {
            score = tanhf(score / softcap) * softcap;
        }
    }
    scores[tid] = score;
    reduce_buf[tid] = score;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce_buf[0];
    const bool valid = (tid < valid_len && tid >= start && tid < max_length);
    const float probability = valid ? __expf(scores[tid] - max_score) : 0.0f;
    probs[tid] = probability;
    reduce_buf[tid] = probability;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / reduce_buf[0];

    if (tid < head_dim) {
        float acc = 0.0f;
        for (int token = start; token < valid_len; ++token) {
            if (token >= max_length) {
                break;
            }
            const int block_index = token / block_size;
            const int offset = token % block_size;
            const long long block_id = block_table[
                batch * max_blocks + block_index
            ];
            const size_t v_offset = (
                ((size_t)block_id * num_kv_heads + kv_head) * block_size + offset
            ) * head_dim + tid;
            acc += probs[token] * inv_sum * __half2float(value_cache[v_offset]);
        }
        output[
            ((size_t)batch * num_heads + head) * head_dim + tid
        ] = __float2half_rn(acc);
    }
}
"""

_SM86_PAGED_GQA_I8_KERNEL = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define DP_PAGED_MAX_THREADS 1024

__device__ __forceinline__ float dp_warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

extern "C" __global__ void dp_paged_gqa_decode_i8(
    const __half* __restrict__ query,
    const signed char* __restrict__ key_cache,
    const signed char* __restrict__ value_cache,
    const __half* __restrict__ key_scales,
    const __half* __restrict__ value_scales,
    const long long* __restrict__ block_table,
    const long long* __restrict__ valid_lengths,
    __half* __restrict__ output,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks,
    int max_length,
    int window_size,
    float scale,
    float softcap,
    int use_softcap)
{
    const int head = blockIdx.x;
    const int batch = blockIdx.y;
    const int tid = threadIdx.x;
    const int kv_group = num_heads / num_kv_heads;
    const int kv_head = head / kv_group;
    const int valid_len = (int)valid_lengths[batch];
    const int start = window_size > 0 ? max(0, valid_len - window_size) : 0;

    __shared__ float scores[DP_PAGED_MAX_THREADS];
    __shared__ float probs[DP_PAGED_MAX_THREADS];
    __shared__ float reduce_buf[DP_PAGED_MAX_THREADS];

    float score = -1.0e30f;
    if (tid < valid_len && tid >= start && tid < max_length) {
        const int block_index = tid / block_size;
        const int offset = tid % block_size;
        const long long block_id = block_table[batch * max_blocks + block_index];
        const size_t kv_scale_offset = (
            (size_t)block_id * num_kv_heads + kv_head
        ) * block_size + offset;
        const size_t kv_offset = (
            ((size_t)block_id * num_kv_heads + kv_head) * block_size + offset
        ) * head_dim;
        const signed char* kptr = key_cache + kv_offset;
        const float k_scale = __half2float(key_scales[kv_scale_offset]);
        float dot = 0.0f;
#pragma unroll 4
        for (int d = 0; d < head_dim; ++d) {
            dot += __half2float(query[
                ((size_t)batch * num_heads + head) * head_dim + d
            ]) * ((float)kptr[d] * k_scale);
        }
        score = dot * scale;
        if (use_softcap) {
            score = tanhf(score / softcap) * softcap;
        }
    }
    scores[tid] = score;
    reduce_buf[tid] = score;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce_buf[0];
    const bool valid = (tid < valid_len && tid >= start && tid < max_length);
    const float probability = valid ? __expf(scores[tid] - max_score) : 0.0f;
    probs[tid] = probability;
    reduce_buf[tid] = probability;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / reduce_buf[0];

    if (tid < head_dim) {
        float acc = 0.0f;
        for (int token = start; token < valid_len; ++token) {
            if (token >= max_length) {
                break;
            }
            const int block_index = token / block_size;
            const int offset = token % block_size;
            const long long block_id = block_table[
                batch * max_blocks + block_index
            ];
            const size_t v_scale_offset = (
                (size_t)block_id * num_kv_heads + kv_head
            ) * block_size + offset;
            const size_t v_offset = (
                ((size_t)block_id * num_kv_heads + kv_head) * block_size + offset
            ) * head_dim + tid;
            acc += probs[token] * inv_sum
                * ((float)value_cache[v_offset]
                   * __half2float(value_scales[v_scale_offset]));
        }
        output[
            ((size_t)batch * num_heads + head) * head_dim + tid
        ] = __float2half_rn(acc);
    }
}
"""

_PAGED_KERNEL: Any = None
_PAGED_I8_KERNEL: Any = None


def _torch_to_cupy(tensor: torch.Tensor) -> Any:
    if cp is None:
        raise RuntimeError("CuPy is unavailable")
    if tensor.device.type != "cuda":
        raise ValueError("native paged attention requires a CUDA tensor")
    contiguous = tensor.contiguous()
    dtype = {
        torch.float16: cp.float16,
        torch.float32: cp.float32,
        torch.int64: cp.int64,
        torch.int32: cp.int32,
        torch.int8: cp.int8,
    }.get(contiguous.dtype)
    if dtype is None:
        raise TypeError(f"unsupported CuPy bridge dtype: {contiguous.dtype}")
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


def _kernel() -> Any:
    global _PAGED_KERNEL
    if cp is None:
        raise RuntimeError("CuPy is required for native paged attention")
    if _PAGED_KERNEL is None:
        _PAGED_KERNEL = cp.RawKernel(
            _SM86_PAGED_GQA_KERNEL,
            "dp_paged_gqa_decode",
            options=("--std=c++14", "--use_fast_math"),
        )
    return _PAGED_KERNEL


def _i8_kernel() -> Any:
    global _PAGED_I8_KERNEL
    if cp is None:
        raise RuntimeError("CuPy is required for native paged attention")
    if _PAGED_I8_KERNEL is None:
        _PAGED_I8_KERNEL = cp.RawKernel(
            _SM86_PAGED_GQA_I8_KERNEL,
            "dp_paged_gqa_decode_i8",
            options=("--std=c++14", "--use_fast_math"),
        )
    return _PAGED_I8_KERNEL


def native_paged_gqa_decode_available() -> bool:
    return cp is not None


def native_paged_gqa_decode(
    query: torch.Tensor,
    cache: PagedLayerKVCache,
    *,
    scale: float,
    sliding_window: int | None,
    softcap: float | None,
) -> torch.Tensor | None:
    """Decode one token per sequence directly from a paged cache.

    Returns ``None`` when the current tensors fall outside the kernel's
    supported FP16/head-dim/batch shape envelope, allowing the caller to use
    the SDPA fallback.
    """
    if cp is None:
        return None
    if query.ndim != 4 or query.shape[2] != 1:
        return None
    if query.dtype != torch.float16 or not query.is_cuda:
        return None
    is_int8 = cache.key_pool.dtype == torch.int8
    if not is_int8 and cache.key_pool.dtype != torch.float16:
        return None
    if cache.value_pool.dtype != cache.key_pool.dtype:
        return None
    if is_int8 and (
        cache.key_scale_pool is None or cache.value_scale_pool is None
    ):
        return None
    batch, heads, _, head_dim = query.shape
    if batch < 1 or heads < 1 or head_dim < 1 or head_dim > 128:
        return None
    max_length = int(cache.max_length)
    if max_length < 1 or max_length > 1024:
        return None
    if cache.block_table.shape[0] != batch:
        return None
    if cache.seq_lens.shape[0] != batch:
        return None
    valid_lengths = cache.seq_lens + int(cache.new_tokens)
    max_blocks = int(cache.block_table.shape[1])
    if max_blocks < 1:
        return None
    threads = 1 << max(0, max(max_length, int(head_dim)) - 1).bit_length()
    threads = min(1024, max(32, threads))
    query_contiguous = query.contiguous()
    key_pool = cache.key_pool.contiguous()
    value_pool = cache.value_pool.contiguous()
    key_scales = (
        cache.key_scale_pool.contiguous() if is_int8 else None
    )
    value_scales = (
        cache.value_scale_pool.contiguous() if is_int8 else None
    )
    block_table = cache.block_table.contiguous()
    lengths = valid_lengths.contiguous()
    output = torch.empty(
        (batch, heads, 1, head_dim),
        dtype=torch.float16,
        device=query.device,
    )
    stream = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    common = (
        _torch_to_cupy(query_contiguous),
        _torch_to_cupy(key_pool),
        _torch_to_cupy(value_pool),
    )
    tail = (
        _torch_to_cupy(block_table),
        _torch_to_cupy(lengths),
        _torch_to_cupy(output),
        int(heads),
        int(cache.key_pool.shape[1]),
        int(head_dim),
        int(cache.block_size),
        max_blocks,
        max_length,
        int(sliding_window or 0),
        cp.float32(scale),
        cp.float32(softcap or 0.0),
        (1 if softcap is not None else 0),
    )
    with stream:
        if is_int8:
            assert key_scales is not None and value_scales is not None
            _i8_kernel()(
                (heads, batch),
                (threads,),
                common
                + (
                    _torch_to_cupy(key_scales),
                    _torch_to_cupy(value_scales),
                )
                + tail,
            )
        else:
            _kernel()(
                (heads, batch),
                (threads,),
                common + tail,
            )
    return output


__all__ = [
    "native_paged_gqa_decode",
    "native_paged_gqa_decode_available",
]
