"""Pointer-array grouped MoE kernels for a GPU packed-expert cache.

Each entry of the pointer arrays points at one expert's packed matrix.  This
lets a caller mix cached GPU experts with freshly staged experts without
copying them into one contiguous packed tensor.
"""

from __future__ import annotations

from typing import Any

import torch

from deepinfer.moe.fused_moe import (
    _FUSED_MOE_SOURCE,
    _QTYPE_CODES,
    _to_cupy,
    fused_moe_supported,
)

try:
    import cupy as cp
except Exception:  # pragma: no cover - CPU-only hosts
    cp = None


_KERNELS: dict[str, Any] = {}

_PTR_GROUPED_SOURCE = _FUSED_MOE_SOURCE + r"""
extern "C" __global__ void dp_ptr_grouped_gate_up_gemm(
    const __half* __restrict__ x,
    const unsigned long long* __restrict__ gate_ptrs,
    const unsigned long long* __restrict__ up_ptrs,
    __half* __restrict__ out,
    const int* __restrict__ counts,
    int max_count,
    int out_features,
    int k_size,
    int gate_qtype,
    int up_qtype,
    int gate_row_bytes,
    int up_row_bytes)
{
    const int row = blockIdx.x;
    const int expert = blockIdx.y;
    const int n_tokens = counts[expert];
    if (n_tokens <= 0) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const unsigned char* gate_row =
        reinterpret_cast<const unsigned char*>(gate_ptrs[expert])
        + (size_t)row * gate_row_bytes;
    const unsigned char* up_row =
        reinterpret_cast<const unsigned char*>(up_ptrs[expert])
        + (size_t)row * up_row_bytes;

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

    const __half* xbase = x + (size_t)expert * max_count * k_size;
    __half* outbase = out + (size_t)expert * max_count * out_features;
    __shared__ float gate_warp_sums[8];
    __shared__ float up_warp_sums[8];

    for (int token = 0; token < n_tokens; ++token) {
        const __half* xt = xbase + (size_t)token * k_size;
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
            outbase[(size_t)token * out_features + row] =
                __float2half_rn(silu * up_sum);
        }
        __syncthreads();
    }
}

extern "C" __global__ void dp_ptr_grouped_down_gemm(
    const __half* __restrict__ x,
    const unsigned long long* __restrict__ down_ptrs,
    __half* __restrict__ out,
    const int* __restrict__ counts,
    int max_count,
    int out_features,
    int k_size,
    int down_qtype,
    int down_row_bytes)
{
    const int row = blockIdx.x;
    const int expert = blockIdx.y;
    const int n_tokens = counts[expert];
    if (n_tokens <= 0) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const unsigned char* down_row =
        reinterpret_cast<const unsigned char*>(down_ptrs[expert])
        + (size_t)row * down_row_bytes;

    float weights[DP_MAX_K_PER_THREAD];
    int weight_count = 0;
    for (int k = tid; k < k_size; k += blockDim.x) {
        weights[weight_count] = dp_weight_value(down_row, k, down_qtype);
        ++weight_count;
    }

    const __half* xbase = x + (size_t)expert * max_count * k_size;
    __half* outbase = out + (size_t)expert * max_count * out_features;
    __shared__ float warp_sums[8];
    for (int token = 0; token < n_tokens; ++token) {
        const __half* xt = xbase + (size_t)token * k_size;
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
            outbase[(size_t)token * out_features + row] =
                __float2half_rn(total);
        }
        __syncthreads();
    }
}
"""


def _kernel(name: str) -> Any:
    if cp is None:
        raise RuntimeError("CuPy is unavailable")
    kernel = _KERNELS.get(name)
    if kernel is None:
        kernel = cp.RawKernel(
            _PTR_GROUPED_SOURCE,
            name,
            options=("--std=c++14", "--use_fast_math"),
        )
        _KERNELS[name] = kernel
    return kernel


def _validate(
    x: torch.Tensor,
    pointers: torch.Tensor,
    counts: torch.Tensor,
    *,
    n_experts: int,
    max_count: int,
    out_features: int,
    k_size: int,
    qtype: str,
) -> None:
    if qtype.upper() not in _QTYPE_CODES:
        raise NotImplementedError(f"pointer grouped MoE does not support {qtype}")
    if x.dtype != torch.float16 or x.device.type != "cuda":
        raise TypeError("pointer grouped MoE expects CUDA FP16 activations")
    if pointers.dtype != torch.int64 or pointers.device != x.device:
        raise TypeError("pointer grouped MoE expects int64 GPU pointers")
    if pointers.numel() != n_experts:
        raise ValueError("pointer array length must equal n_experts")
    if counts.dtype != torch.int32 or counts.device != x.device:
        raise TypeError("pointer grouped MoE expects int32 GPU counts")
    if counts.numel() != n_experts:
        raise ValueError("counts length must equal n_experts")
    if x.numel() != n_experts * max_count * k_size:
        raise ValueError("grouped activation shape does not match metadata")
    if k_size > 256 * 16:
        raise NotImplementedError("k_size exceeds the grouped kernel register tile")


def fused_gate_up_gemm_ptr_grouped(
    x_all: torch.Tensor,
    gate_pointers: torch.Tensor,
    up_pointers: torch.Tensor,
    counts: torch.Tensor,
    *,
    n_experts: int,
    max_count: int,
    out_features: int,
    k_size: int,
    gate_qtype: str,
    up_qtype: str,
    gate_row_bytes: int,
    up_row_bytes: int,
) -> torch.Tensor:
    if not fused_moe_supported():
        raise RuntimeError("pointer grouped MoE requires CUDA and CuPy")
    _validate(
        x_all,
        gate_pointers,
        counts,
        n_experts=n_experts,
        max_count=max_count,
        out_features=out_features,
        k_size=k_size,
        qtype=gate_qtype,
    )
    _validate(
        x_all,
        up_pointers,
        counts,
        n_experts=n_experts,
        max_count=max_count,
        out_features=out_features,
        k_size=k_size,
        qtype=up_qtype,
    )
    out = torch.empty(
        (n_experts, max_count, out_features),
        dtype=torch.float16,
        device=x_all.device,
    )
    _kernel("dp_ptr_grouped_gate_up_gemm")(
        (out_features, n_experts),
        (256,),
        (
            _to_cupy(x_all),
            _to_cupy(gate_pointers),
            _to_cupy(up_pointers),
            _to_cupy(out),
            _to_cupy(counts),
            max_count,
            out_features,
            k_size,
            _QTYPE_CODES[gate_qtype.upper()],
            _QTYPE_CODES[up_qtype.upper()],
            gate_row_bytes,
            up_row_bytes,
        ),
    )
    return out


def fused_down_gemm_ptr_grouped(
    x_all: torch.Tensor,
    down_pointers: torch.Tensor,
    counts: torch.Tensor,
    *,
    n_experts: int,
    max_count: int,
    out_features: int,
    k_size: int,
    down_qtype: str,
    down_row_bytes: int,
) -> torch.Tensor:
    if not fused_moe_supported():
        raise RuntimeError("pointer grouped MoE requires CUDA and CuPy")
    _validate(
        x_all,
        down_pointers,
        counts,
        n_experts=n_experts,
        max_count=max_count,
        out_features=out_features,
        k_size=k_size,
        qtype=down_qtype,
    )
    out = torch.empty(
        (n_experts, max_count, out_features),
        dtype=torch.float16,
        device=x_all.device,
    )
    _kernel("dp_ptr_grouped_down_gemm")(
        (out_features, n_experts),
        (256,),
        (
            _to_cupy(x_all),
            _to_cupy(down_pointers),
            _to_cupy(out),
            _to_cupy(counts),
            max_count,
            out_features,
            k_size,
            _QTYPE_CODES[down_qtype.upper()],
            down_row_bytes,
        ),
    )
    return out
