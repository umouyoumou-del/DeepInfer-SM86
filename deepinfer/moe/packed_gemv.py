"""Single-token expert GEMVs on packed Q4_K / Q6_K rows, one warp per row.

The pointer-grouped kernels in ``ptr_grouped_moe`` decode one weight per call:
every element re-reads its super-block's scales and unpacks a single nibble,
so they run instruction-bound at about 8% of the RTX 3050's bandwidth and take
most of a decode step's GPU time once a CUDA graph has removed the host.

Here a warp owns an output row.  For Q4_K each lane reads 4 contiguous quant
bytes of a super-block - the warp reads its 128 bytes in one coalesced pass -
and decodes the two sub-block scales once; Q6_K's 210-byte blocks are only
2-byte aligned, so it reads ushorts.  The activations sit in shared memory as
float and are read with float4/float2 accesses that do not bank-conflict.

The kernels also absorb the glue around them: they index the full expert
pointer tables with the routed ids themselves (no gather), gate/up read one
shared activation vector (no repeat), and down folds the routing-weighted sum
over experts into its accumulator, writing the layer's MoE output directly.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch

from deepinfer.moe.gpu_quant import _to_cupy

try:
    import cupy as cp
except Exception:  # pragma: no cover - CPU-only hosts
    cp = None

_QTYPES = {"Q4_K": 0, "Q6_K": 1}
_BLOCK_BYTES = {"Q4_K": 144, "Q6_K": 210}
_WARPS = 8
_SHARED_LIMIT = 48 * 1024
_KERNELS: dict[str, Any] = {}
_STREAMS: dict[int, Any] = {}

_SOURCE = r"""
#include <cuda_fp16.h>

__device__ __forceinline__ float pg_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ void pg_q4k_scale_min(
    const unsigned char* scales, int sub, float* scale, float* minimum)
{
    if (sub < 4) {
        *scale = (float)(scales[sub] & 0x3F);
        *minimum = (float)(scales[sub + 4] & 0x3F);
    } else {
        *scale = (float)((scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 2) & 0x30));
        *minimum = (float)((scales[sub + 4] >> 4) | ((scales[sub] >> 2) & 0x30));
    }
}

// Lane L reads quant bytes [4L, 4L+4) of every 256-element super-block: they
// hold elements 4(L%8)..4(L%8)+3 of sub-block 2(L/8) (low nibbles) and of
// sub-block 2(L/8)+1 (high nibbles).
__device__ __forceinline__ float pg_q4k_dot(
    const unsigned char* row, const float* xs, int k_size, int lane)
{
    const int pair = lane >> 3;
    const int pos = (lane & 7) * 4;
    const int s0 = 2 * pair;
    const int s1 = s0 + 1;
    float acc = 0.0f;
    for (int base = 0; base < k_size; base += 256) {
        const unsigned char* block = row + (size_t)(base >> 8) * 144;
        const float d = __half2float(*reinterpret_cast<const __half*>(block));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
        float sc0, m0, sc1, m1;
        pg_q4k_scale_min(block + 4, s0, &sc0, &m0);
        pg_q4k_scale_min(block + 4, s1, &sc1, &m1);
        const unsigned int q = *reinterpret_cast<const unsigned int*>(block + 16 + lane * 4);
        const float4 xa = *reinterpret_cast<const float4*>(xs + base + s0 * 32 + pos);
        const float4 xb = *reinterpret_cast<const float4*>(xs + base + s1 * 32 + pos);
        const float dot0 = (float)(q & 0xF) * xa.x + (float)((q >> 8) & 0xF) * xa.y
                         + (float)((q >> 16) & 0xF) * xa.z + (float)((q >> 24) & 0xF) * xa.w;
        const float dot1 = (float)((q >> 4) & 0xF) * xb.x + (float)((q >> 12) & 0xF) * xb.y
                         + (float)((q >> 20) & 0xF) * xb.z + (float)((q >> 28) & 0xF) * xb.w;
        const float sum0 = xa.x + xa.y + xa.z + xa.w;
        const float sum1 = xb.x + xb.y + xb.z + xb.w;
        acc += d * (sc0 * dot0 + sc1 * dot1) - dmin * (m0 * sum0 + m1 * sum1);
    }
    return acc;
}

// Lane L covers half L/16 of every super-block at positions l = 2(L%16) and
// l+1, each in all four quarters: 8 elements per super-block.
__device__ __forceinline__ float pg_q6k_dot(
    const unsigned char* row, const float* xs, int k_size, int lane)
{
    const int half = lane >> 4;
    const int l = (lane & 15) * 2;
    float acc = 0.0f;
    for (int base = 0; base < k_size; base += 256) {
        const unsigned char* block = row + (size_t)(base >> 8) * 210;
        const unsigned char* ql = block + half * 64;
        const unsigned char* qh = block + 128 + half * 32;
        const signed char* sc = reinterpret_cast<const signed char*>(block + 192) + half * 8;
        const float d = __half2float(*reinterpret_cast<const __half*>(block + 208));
        const unsigned int lo_a = *reinterpret_cast<const unsigned short*>(ql + l);
        const unsigned int lo_b = *reinterpret_cast<const unsigned short*>(ql + l + 32);
        const unsigned int hi = *reinterpret_cast<const unsigned short*>(qh + l);
        const float* xh = xs + base + half * 128 + l;
        float sum = 0.0f;
        #pragma unroll
        for (int quarter = 0; quarter < 4; ++quarter) {
            const unsigned int src = (quarter & 1) ? lo_b : lo_a;
            const int shift = (quarter >> 1) * 4;
            const int q0 = (int)(((src >> shift) & 0xF) | (((hi >> (2 * quarter)) & 3) << 4)) - 32;
            const int q1 = (int)(((src >> (8 + shift)) & 0xF)
                               | (((hi >> (8 + 2 * quarter)) & 3) << 4)) - 32;
            const float2 xv = *reinterpret_cast<const float2*>(xh + 32 * quarter);
            sum += (float)sc[(l >> 4) + quarter * 2] * ((float)q0 * xv.x + (float)q1 * xv.y);
        }
        acc += d * sum;
    }
    return acc;
}

__device__ __forceinline__ float pg_row_dot(
    const unsigned char* row, const float* xs, int k_size, int qtype, int lane)
{
    return qtype == 0 ? pg_q4k_dot(row, xs, k_size, lane) : pg_q6k_dot(row, xs, k_size, lane);
}

// out[s, r] = silu(gate_e[r] . x) * (up_e[r] . x), e = expert_ids[s]
// grid (ceil(out_features / warps), top_k), shared k_size floats
extern "C" __global__ void pg_gate_up_gemv(
    const __half* __restrict__ x,
    const long long* __restrict__ expert_ids,
    const unsigned long long* __restrict__ gate_table,
    const unsigned long long* __restrict__ up_table,
    __half* __restrict__ out,
    int out_features, int k_size, int gate_qtype, int up_qtype,
    int gate_row_bytes, int up_row_bytes)
{
    extern __shared__ float4 shared4[];
    float* xs = reinterpret_cast<float*>(shared4);
    for (int i = threadIdx.x; i < k_size; i += blockDim.x) {
        xs[i] = __half2float(x[i]);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= out_features) return;
    const int slot = blockIdx.y;
    const long long expert = expert_ids[slot];
    const unsigned char* g = reinterpret_cast<const unsigned char*>(gate_table[expert])
                           + (size_t)row * gate_row_bytes;
    const unsigned char* u = reinterpret_cast<const unsigned char*>(up_table[expert])
                           + (size_t)row * up_row_bytes;
    const float gate = pg_warp_sum(pg_row_dot(g, xs, k_size, gate_qtype, lane));
    const float up = pg_warp_sum(pg_row_dot(u, xs, k_size, up_qtype, lane));
    if (lane == 0) {
        out[(size_t)slot * out_features + row] = __float2half_rn(gate / (1.0f + __expf(-gate)) * up);
    }
}

// out[r] = sum_s w[s] * (down_e[r] . act[s]), e = expert_ids[s]
// grid ceil(out_features / warps), shared top_k * k_size floats
extern "C" __global__ void pg_down_gemv_combine(
    const __half* __restrict__ act,
    const long long* __restrict__ expert_ids,
    const float* __restrict__ route_weights,
    const unsigned long long* __restrict__ down_table,
    __half* __restrict__ out,
    int top_k, int out_features, int k_size, int down_qtype, int down_row_bytes)
{
    extern __shared__ float4 shared4[];
    float* xs = reinterpret_cast<float*>(shared4);
    for (int i = threadIdx.x; i < top_k * k_size; i += blockDim.x) {
        xs[i] = __half2float(act[i]);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= out_features) return;
    float total = 0.0f;
    for (int slot = 0; slot < top_k; ++slot) {
        const unsigned char* d = reinterpret_cast<const unsigned char*>(down_table[expert_ids[slot]])
                               + (size_t)row * down_row_bytes;
        total += route_weights[slot] * pg_row_dot(d, xs + (size_t)slot * k_size, k_size,
                                                  down_qtype, lane);
    }
    total = pg_warp_sum(total);
    if (lane == 0) {
        out[row] = __float2half_rn(total);
    }
}
"""


def _kernel(name: str) -> Any:
    kernel = _KERNELS.get(name)
    if kernel is None:
        kernel = cp.RawKernel(_SOURCE, name, options=("--std=c++14", "--use_fast_math"))
        _KERNELS[name] = kernel
    return kernel


def decode_gemv_supported(*qtypes: str, k_size: int = 256, top_k: int = 1) -> bool:
    """Whether these kernels cover the given formats and shapes."""

    return (
        cp is not None
        and torch.cuda.is_available()
        and all(qtype.upper() in _QTYPES for qtype in qtypes)
        and k_size % 256 == 0
        and top_k * k_size * 4 <= _SHARED_LIMIT
    )


def _row_bytes(k_size: int, qtype: str) -> int:
    return k_size // 256 * _BLOCK_BYTES[qtype.upper()]


def _check(row_bytes: int, k_size: int, qtype: str, table: torch.Tensor, ids: torch.Tensor) -> None:
    if row_bytes != _row_bytes(k_size, qtype):
        raise ValueError(f"{qtype} row of {k_size} is {_row_bytes(k_size, qtype)} bytes, got {row_bytes}")
    if table.dtype != torch.int64 or not table.is_cuda:
        raise TypeError("pointer tables must be int64 CUDA tensors")
    if ids.dtype != torch.int64 or not ids.is_cuda or ids.dim() != 1:
        raise TypeError("expert ids must be a 1-D int64 CUDA tensor")


def _launch(name: str, grid, shared: int, args) -> None:
    """Launch on the caller's torch stream, so eager use and graph capture both work.

    torch's default stream is the legacy null stream CuPy launches on anyway;
    any other stream (a graph's capture stream) gets a cached CuPy wrapper.
    """

    handle = torch.cuda.current_stream().cuda_stream
    kernel = _kernel(name)
    if handle == 0:
        kernel(grid, (32 * _WARPS,), args, shared_mem=shared)
        return
    stream = _STREAMS.get(handle)
    if stream is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            stream = _STREAMS[handle] = cp.cuda.ExternalStream(handle)
    with stream:
        kernel(grid, (32 * _WARPS,), args, shared_mem=shared)


def gate_up_gemv(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_table: torch.Tensor,
    up_table: torch.Tensor,
    *,
    out_features: int,
    k_size: int,
    gate_qtype: str,
    up_qtype: str,
    gate_row_bytes: int,
    up_row_bytes: int,
) -> torch.Tensor:
    """``[top_k, out_features]`` = silu(gate . x) * (up . x) for each routed expert.

    ``x`` is the token's ``[k_size]`` fp16 activation, shared by every expert;
    the tables hold one device pointer per expert of the whole layer.
    """

    if x.dtype != torch.float16 or not x.is_cuda or x.numel() != k_size:
        raise TypeError("x must be a CUDA fp16 tensor of k_size elements")
    _check(gate_row_bytes, k_size, gate_qtype, gate_table, expert_ids)
    _check(up_row_bytes, k_size, up_qtype, up_table, expert_ids)
    top_k = expert_ids.numel()
    out = torch.empty((top_k, out_features), dtype=torch.float16, device=x.device)
    _launch(
        "pg_gate_up_gemv",
        ((out_features + _WARPS - 1) // _WARPS, top_k),
        k_size * 4,
        (
            _to_cupy(x), _to_cupy(expert_ids), _to_cupy(gate_table), _to_cupy(up_table),
            _to_cupy(out), out_features, k_size, _QTYPES[gate_qtype.upper()],
            _QTYPES[up_qtype.upper()], gate_row_bytes, up_row_bytes,
        ),
    )
    return out


def down_gemv_combine(
    activation: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    down_table: torch.Tensor,
    *,
    out_features: int,
    k_size: int,
    down_qtype: str,
    down_row_bytes: int,
) -> torch.Tensor:
    """``[out_features]`` = sum over routed experts of weight * (down . activation)."""

    top_k = expert_ids.numel()
    if activation.dtype != torch.float16 or activation.shape != (top_k, k_size):
        raise TypeError("activation must be fp16 [top_k, k_size]")
    if route_weights.dtype != torch.float32 or route_weights.numel() != top_k:
        raise TypeError("route weights must be fp32 [top_k]")
    _check(down_row_bytes, k_size, down_qtype, down_table, expert_ids)
    out = torch.empty(out_features, dtype=torch.float16, device=activation.device)
    _launch(
        "pg_down_gemv_combine",
        ((out_features + _WARPS - 1) // _WARPS,),
        top_k * k_size * 4,
        (
            _to_cupy(activation.contiguous()), _to_cupy(expert_ids),
            _to_cupy(route_weights.contiguous()), _to_cupy(down_table), _to_cupy(out),
            top_k, out_features, k_size, _QTYPES[down_qtype.upper()], down_row_bytes,
        ),
    )
    return out
