"""Fused packed-weight MoE GEMV/GEMM for SM86.

The regular path dequantizes each selected expert into FP16, writes it to HBM,
and then asks cuBLAS to read it back for a grouped matmul.  For a decode token,
that materializes roughly 2.7 GiB of expert weights over all 26 MoE layers.
These kernels consume the GGUF Q4_K/Q5_0/Q8_0 packed rows directly: each thread
block decodes one weight row once and immediately multiplies it by the assigned
token activations.  Only the small activation tensors are written to HBM.
"""

from __future__ import annotations

from typing import Any

import torch

from deepinfer.moe.gpu_quant import _to_cupy

try:
    import cupy as cp
except Exception:  # pragma: no cover - CPU-only hosts
    cp = None


_QTYPE_CODES = {"Q4_K": 0, "Q5_0": 1, "Q8_0": 2, "Q6_K": 3}
_KERNELS: dict[str, Any] = {}

_FUSED_MOE_SOURCE = r"""
#include <cuda_fp16.h>

#define DP_MAX_K_PER_THREAD 16

__device__ __forceinline__ float dp_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float dp_q4_k_value(
    const unsigned char* row,
    int k)
{
    const unsigned char* block = row + (size_t)(k >> 8) * 144;
    const float d = __half2float(*reinterpret_cast<const __half*>(block));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
    const unsigned char* scales = block + 4;
    const int rem = k & 255;
    const int sub = rem >> 5;
    const int pos = rem & 31;
    unsigned int sc;
    unsigned int m;
    if (sub < 4) {
        sc = scales[sub] & 0x3F;
        m = scales[sub + 4] & 0x3F;
    } else {
        sc = (scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 2) & 0x30);
        m = (scales[sub + 4] >> 4) | ((scales[sub] >> 2) & 0x30);
    }
    const unsigned char qb = block[16 + (sub >> 1) * 32 + pos];
    const int q = (sub & 1) ? (qb >> 4) : (qb & 0x0F);
    return d * (float)sc * (float)q - dmin * (float)m;
}

__device__ __forceinline__ float dp_q5_0_value(
    const unsigned char* row,
    int k)
{
    const unsigned char* block = row + (size_t)(k >> 5) * 22;
    const float d = __half2float(*reinterpret_cast<const __half*>(block));
    const int pos = k & 31;
    const unsigned int qh =
        (unsigned int)block[2] |
        ((unsigned int)block[3] << 8) |
        ((unsigned int)block[4] << 16) |
        ((unsigned int)block[5] << 24);
    const unsigned char qs = block[6 + (pos & 15)];
    const int ql = (pos < 16) ? (qs & 0x0F) : (qs >> 4);
    const int q = ql | (((qh >> pos) & 1u) << 4);
    return d * (float)(q - 16);
}

__device__ __forceinline__ float dp_q8_0_value(
    const unsigned char* row,
    int k)
{
    const unsigned char* block = row + (size_t)(k >> 5) * 34;
    const float d = __half2float(*reinterpret_cast<const __half*>(block));
    const int q = (int)((signed char)block[2 + (k & 31)]);
    return d * (float)q;
}

__device__ __forceinline__ float dp_q6_k_value(
    const unsigned char* row,
    int k)
{
    // block_q6_K: ql[128] | qh[64] | int8 scales[16] | fp16 d  => 210 bytes
    const unsigned char* block = row + (size_t)(k >> 8) * 210;
    const int kk = k & 255;
    const int half = kk >> 7;
    const unsigned char* ql = block + half * 64;
    const unsigned char* qh = block + 128 + half * 32;
    const signed char* sc =
        reinterpret_cast<const signed char*>(block + 192) + half * 8;
    const float d = __half2float(*reinterpret_cast<const __half*>(block + 208));
    const int r = kk & 127;
    const int l = r & 31;
    const int quarter = r >> 5;
    int q;
    if (quarter == 0) {
        q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
    } else if (quarter == 1) {
        q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
    } else if (quarter == 2) {
        q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
    } else {
        q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
    }
    const int s = sc[(l >> 4) + quarter * 2];
    return d * (float)s * (float)(q - 32);
}

__device__ __forceinline__ float dp_weight_value(
    const unsigned char* row,
    int k,
    int qtype)
{
    if (qtype == 0) return dp_q4_k_value(row, k);
    if (qtype == 1) return dp_q5_0_value(row, k);
    if (qtype == 3) return dp_q6_k_value(row, k);
    return dp_q8_0_value(row, k);
}

extern "C" __global__ void dp_fused_gate_up_gemm(
    const __half* __restrict__ x,
    const unsigned char* __restrict__ gate,
    const unsigned char* __restrict__ up,
    __half* __restrict__ out,
    int n_tokens,
    int out_features,
    int k_size,
    int gate_qtype,
    int up_qtype,
    int gate_row_bytes,
    int up_row_bytes)
{
    const int row = blockIdx.x;
    if (row >= out_features) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const unsigned char* gate_row =
        gate + (size_t)row * gate_row_bytes;
    const unsigned char* up_row =
        up + (size_t)row * up_row_bytes;

    float gate_weights[DP_MAX_K_PER_THREAD];
    float up_weights[DP_MAX_K_PER_THREAD];
    int weight_count = 0;
    for (int k = tid; k < k_size; k += blockDim.x) {
        gate_weights[weight_count] =
            dp_weight_value(gate_row, k, gate_qtype);
        up_weights[weight_count] =
            dp_weight_value(up_row, k, up_qtype);
        ++weight_count;
    }

    __shared__ float gate_warp_sums[8];
    __shared__ float up_warp_sums[8];

    for (int token = 0; token < n_tokens; ++token) {
        const __half* xt = x + (size_t)token * k_size;
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        int index = 0;
        for (int k = tid; k < k_size; k += blockDim.x) {
            const float xv = __half2float(xt[k]);
            gate_acc += xv * gate_weights[index];
            up_acc += xv * up_weights[index];
            ++index;
        }
        gate_acc = dp_warp_sum(gate_acc);
        up_acc = dp_warp_sum(up_acc);
        if (lane == 0) {
            gate_warp_sums[warp] = gate_acc;
            up_warp_sums[warp] = up_acc;
        }
        __syncthreads();
        if (tid == 0) {
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            for (int w = 0; w < (blockDim.x >> 5); ++w) {
                gate_sum += gate_warp_sums[w];
                up_sum += up_warp_sums[w];
            }
            const float silu = gate_sum / (1.0f + __expf(-gate_sum));
            out[(size_t)token * out_features + row] =
                __float2half_rn(silu * up_sum);
        }
        __syncthreads();
    }
}

extern "C" __global__ void dp_fused_down_gemm(
    const __half* __restrict__ x,
    const unsigned char* __restrict__ down,
    __half* __restrict__ out,
    int n_tokens,
    int out_features,
    int k_size,
    int down_qtype,
    int down_row_bytes)
{
    const int row = blockIdx.x;
    if (row >= out_features) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const unsigned char* down_row =
        down + (size_t)row * down_row_bytes;
    float weights[DP_MAX_K_PER_THREAD];
    int weight_count = 0;
    for (int k = tid; k < k_size; k += blockDim.x) {
        weights[weight_count] = dp_weight_value(down_row, k, down_qtype);
        ++weight_count;
    }

    __shared__ float warp_sums[8];
    for (int token = 0; token < n_tokens; ++token) {
        const __half* xt = x + (size_t)token * k_size;
        float acc = 0.0f;
        int index = 0;
        for (int k = tid; k < k_size; k += blockDim.x) {
            acc += __half2float(xt[k]) * weights[index];
            ++index;
        }
        acc = dp_warp_sum(acc);
        if (lane == 0) {
            warp_sums[warp] = acc;
        }
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < (blockDim.x >> 5); ++w) {
                total += warp_sums[w];
            }
            out[(size_t)token * out_features + row] =
                __float2half_rn(total);
        }
        __syncthreads();
    }
}
"""


def fused_moe_supported() -> bool:
    return cp is not None and torch.cuda.is_available()


def _kernel(name: str) -> Any:
    if cp is None:
        raise RuntimeError("CuPy is unavailable")
    kernel = _KERNELS.get(name)
    if kernel is None:
        kernel = cp.RawKernel(
            _FUSED_MOE_SOURCE,
            name,
            options=("--std=c++14", "--use_fast_math"),
        )
        _KERNELS[name] = kernel
    return kernel


def _validate_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    n_tokens: int,
    out_features: int,
    k_size: int,
    row_bytes: int,
    qtype: str,
) -> None:
    if qtype.upper() not in _QTYPE_CODES:
        raise NotImplementedError(f"fused MoE does not support {qtype}")
    if x.dtype != torch.float16 or x.device.type != "cuda":
        raise TypeError("fused MoE expects a CUDA FP16 activation tensor")
    if weight.dtype != torch.uint8 or weight.device != x.device:
        raise TypeError("fused MoE expects packed uint8 weights on the same GPU")
    if x.numel() != n_tokens * k_size:
        raise ValueError("activation shape does not match n_tokens * k_size")
    if weight.numel() != out_features * row_bytes:
        raise ValueError("packed weight size does not match out_features * row_bytes")
    if k_size > 256 * 16:
        raise NotImplementedError("k_size exceeds the fused kernel register tile")


def fused_gate_up_gemm(
    x: torch.Tensor,
    gate_packed: torch.Tensor,
    up_packed: torch.Tensor,
    *,
    n_tokens: int,
    out_features: int,
    k_size: int,
    gate_qtype: str,
    up_qtype: str,
    gate_row_bytes: int,
    up_row_bytes: int,
) -> torch.Tensor:
    """Return ``silu(x @ gate.T) * (x @ up.T)`` without FP16 weights."""

    if not fused_moe_supported():
        raise RuntimeError("fused MoE requires CUDA and CuPy")
    _validate_inputs(
        x,
        gate_packed,
        n_tokens,
        out_features,
        k_size,
        gate_row_bytes,
        gate_qtype,
    )
    _validate_inputs(
        x,
        up_packed,
        n_tokens,
        out_features,
        k_size,
        up_row_bytes,
        up_qtype,
    )
    out = torch.empty(
        (n_tokens, out_features), dtype=torch.float16, device=x.device
    )
    _kernel("dp_fused_gate_up_gemm")(
        (out_features,),
        (256,),
        (
            _to_cupy(x),
            _to_cupy(gate_packed),
            _to_cupy(up_packed),
            _to_cupy(out),
            n_tokens,
            out_features,
            k_size,
            _QTYPE_CODES[gate_qtype.upper()],
            _QTYPE_CODES[up_qtype.upper()],
            gate_row_bytes,
            up_row_bytes,
        ),
    )
    return out


def fused_down_gemm(
    x: torch.Tensor,
    down_packed: torch.Tensor,
    *,
    n_tokens: int,
    out_features: int,
    k_size: int,
    down_qtype: str,
    down_row_bytes: int,
) -> torch.Tensor:
    """Return ``x @ down.T`` from packed down-projection weights."""

    if not fused_moe_supported():
        raise RuntimeError("fused MoE requires CUDA and CuPy")
    _validate_inputs(
        x,
        down_packed,
        n_tokens,
        out_features,
        k_size,
        down_row_bytes,
        down_qtype,
    )
    out = torch.empty(
        (n_tokens, out_features), dtype=torch.float16, device=x.device
    )
    _kernel("dp_fused_down_gemm")(
        (out_features,),
        (256,),
        (
            _to_cupy(x),
            _to_cupy(down_packed),
            _to_cupy(out),
            n_tokens,
            out_features,
            k_size,
            _QTYPE_CODES[down_qtype.upper()],
            down_row_bytes,
        ),
    )
    return out
