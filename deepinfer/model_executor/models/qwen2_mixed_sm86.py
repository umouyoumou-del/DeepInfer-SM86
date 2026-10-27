"""Mixed W4A16/W3A16 SM86 backend for Qwen2.5-7B.

Attention, embedding and lm_head use the existing 4-bit path.  The much larger
MLP matrices use a 3-bit format (8 signed values in 3 bytes) to make the 7B
checkpoint fit in a 4 GB RTX 3050.  Small-batch decode uses native GEMV
kernels; multi-token prefill uses tiled W4/W3 GEMMs.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from deepinfer.hugging_face.qwen2_config import build_qwen2_model_config
from deepinfer.model_executor.models.qwen2 import Qwen2ForCausalLM
from deepinfer.model_executor.models.qwen2_w4a16 import (
    W4A16Linear,
    _native_gqa_decode,
    _torch_to_cupy,
    cp,
)

_W3_KERNEL = r"""
#include <cuda_fp16.h>

__device__ __forceinline__ float s3v(unsigned int value) {
    int v = (int)(value & 0x07);
    if (v >= 4) v -= 8;
    return (float)v;
}

__device__ __forceinline__ float warp_sum3v(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

extern "C" __global__ void qwen2_w3a16_gemv(
    const __half* __restrict__ x,
    const unsigned char* __restrict__ packed,
    const __half* __restrict__ scales,
    __half* __restrict__ out,
    int n_rows,
    int k_size,
    int group_size)
{
    const int batch = blockIdx.y;
    const __half* xb = x + (size_t)batch * k_size;
    extern __shared__ __half shared_x[];
    for (int i = threadIdx.x; i < k_size; i += blockDim.x) shared_x[i] = xb[i];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= n_rows) return;

    const int row_bytes = k_size * 3 / 8;
    const int groups_per_row = k_size / group_size;
    const unsigned char* row_packed = packed + (size_t)row * row_bytes;
    const __half* row_scales = scales + (size_t)row * groups_per_row;
    float acc = 0.0f;

    for (int k0 = lane * 16; k0 < k_size; k0 += 512) {
        const float scale = __half2float(row_scales[k0 / group_size]);
        const unsigned char* group = row_packed + (size_t)(k0 / 8) * 3;
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            const unsigned char* b = group + part * 3;
            const unsigned int b0 = b[0], b1 = b[1], b2 = b[2];
            const int kk = k0 + part * 8;
            acc += __half2float(shared_x[kk + 0]) * scale * s3v(b0 & 0x07);
            acc += __half2float(shared_x[kk + 1]) * scale * s3v((b0 >> 3) & 0x07);
            acc += __half2float(shared_x[kk + 2]) * scale * s3v(((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2));
            acc += __half2float(shared_x[kk + 3]) * scale * s3v((b1 >> 1) & 0x07);
            acc += __half2float(shared_x[kk + 4]) * scale * s3v((b1 >> 4) & 0x07);
            acc += __half2float(shared_x[kk + 5]) * scale * s3v(((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1));
            acc += __half2float(shared_x[kk + 6]) * scale * s3v((b2 >> 2) & 0x07);
            acc += __half2float(shared_x[kk + 7]) * scale * s3v((b2 >> 5) & 0x07);
        }
    }
    acc = warp_sum3v(acc);
    if (lane == 0) out[(size_t)batch * n_rows + row] = __float2half_rn(acc);
}

#define DP_W3_BATCH_TILE 8

// Batch-tiled W3 GEMV used by continuous-batching decode.  One block reuses
// each packed weight row across up to eight activation vectors.
extern "C" __global__ void qwen2_w3a16_batched_gemv(
    const __half* __restrict__ x,
    const unsigned char* __restrict__ packed,
    const __half* __restrict__ scales,
    __half* __restrict__ out,
    int n_rows,
    int k_size,
    int group_size,
    int batch_size)
{
    const int batch_start = blockIdx.y * DP_W3_BATCH_TILE;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= n_rows) return;

    const int row_bytes = k_size * 3 / 8;
    const int groups_per_row = k_size / group_size;
    const unsigned char* row_packed = packed + (size_t)row * row_bytes;
    const __half* row_scales = scales + (size_t)row * groups_per_row;
    float acc[DP_W3_BATCH_TILE];
#pragma unroll
    for (int b = 0; b < DP_W3_BATCH_TILE; ++b) acc[b] = 0.0f;

    for (int k0 = lane * 16; k0 < k_size; k0 += 512) {
        const float scale = __half2float(row_scales[k0 / group_size]);
        const unsigned char* group = row_packed + (size_t)(k0 / 8) * 3;
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            const unsigned char* b = group + part * 3;
            const unsigned int b0 = b[0], b1 = b[1], b2 = b[2];
            const int kk = k0 + part * 8;
            const float v0 = scale * s3v(b0 & 0x07);
            const float v1 = scale * s3v((b0 >> 3) & 0x07);
            const float v2 = scale * s3v(((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2));
            const float v3 = scale * s3v((b1 >> 1) & 0x07);
            const float v4 = scale * s3v((b1 >> 4) & 0x07);
            const float v5 = scale * s3v(((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1));
            const float v6 = scale * s3v((b2 >> 2) & 0x07);
            const float v7 = scale * s3v((b2 >> 5) & 0x07);
#pragma unroll
            for (int bb = 0; bb < DP_W3_BATCH_TILE; ++bb) {
                const int batch = batch_start + bb;
                if (batch < batch_size) {
                    const __half* xb = x + (size_t)batch * k_size;
                    acc[bb] += __half2float(xb[kk + 0]) * v0;
                    acc[bb] += __half2float(xb[kk + 1]) * v1;
                    acc[bb] += __half2float(xb[kk + 2]) * v2;
                    acc[bb] += __half2float(xb[kk + 3]) * v3;
                    acc[bb] += __half2float(xb[kk + 4]) * v4;
                    acc[bb] += __half2float(xb[kk + 5]) * v5;
                    acc[bb] += __half2float(xb[kk + 6]) * v6;
                    acc[bb] += __half2float(xb[kk + 7]) * v7;
                }
            }
        }
    }
#pragma unroll
    for (int bb = 0; bb < DP_W3_BATCH_TILE; ++bb) {
        const int batch = batch_start + bb;
        if (batch < batch_size) {
            const float value = warp_sum3v(acc[bb]);
            if (lane == 0) {
                out[(size_t)batch * n_rows + row] = __float2half_rn(value);
            }
        }
    }
}
"""

_SM86_W3A16_TILED_GEMM_KERNEL = r"""
#include <cuda_fp16.h>

#define QWEN2_W3_GEMM_BM 32
#define QWEN2_W3_GEMM_BN 64
#define QWEN2_W3_GEMM_BK 32
#define QWEN2_W3_GEMM_THREADS 256

__device__ __forceinline__ float qwen2_w3_gemm_s3(unsigned int value) {
    int v = (int)(value & 0x07);
    if (v >= 4) v -= 8;
    return (float)v;
}

// Tiled W3A16 prefill GEMM.  CTA and register tiling mirror the W4 kernel;
// only the packed-weight decode differs.  One warp decodes eight B rows at
// once (four lanes per row, three packed bytes per eight 3-bit values).
extern "C" __global__ void qwen2_w3a16_gemm(
    const __half* __restrict__ a,
    const unsigned char* __restrict__ packed,
    const __half* __restrict__ scales,
    __half* __restrict__ c,
    int m_size,
    int n_rows,
    int k_size,
    int group_size)
{
    __shared__ __half shared_a[QWEN2_W3_GEMM_BM][QWEN2_W3_GEMM_BK];
    __shared__ __half shared_b[QWEN2_W3_GEMM_BK][QWEN2_W3_GEMM_BN + 1];

    const int tid = threadIdx.x;
    const int row_in_tile = tid >> 4;
    const int col_group = tid & 15;
    const int block_m = blockIdx.y * QWEN2_W3_GEMM_BM;
    const int block_n = blockIdx.x * QWEN2_W3_GEMM_BN;
    const int groups_per_row = k_size / group_size;
    const int row_bytes = k_size * 3 / 8;

    float acc[2][4];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[r][j] = 0.0f;
    }

    for (int k0 = 0; k0 < k_size; k0 += QWEN2_W3_GEMM_BK) {
        #pragma unroll
        for (int index = tid; index < QWEN2_W3_GEMM_BM * QWEN2_W3_GEMM_BK;
             index += QWEN2_W3_GEMM_THREADS) {
            const int local_m = index / QWEN2_W3_GEMM_BK;
            const int local_k = index % QWEN2_W3_GEMM_BK;
            const int global_m = block_m + local_m;
            const int global_k = k0 + local_k;
            shared_a[local_m][local_k] =
                (global_m < m_size && global_k < k_size)
                    ? a[(size_t)global_m * k_size + global_k]
                    : __float2half_rn(0.0f);
        }

        const int lane = tid & 31;
        const int warp = tid >> 5;
        #pragma unroll
        for (int n_base = warp * 8; n_base < QWEN2_W3_GEMM_BN; n_base += 64) {
            const int local_n = n_base + (lane >> 2);
            const int k_group = lane & 3;
            const int global_n = block_n + local_n;
            if (global_n < n_rows) {
                const int local_k = k_group * 8;
                const int global_k = k0 + local_k;
                float values[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) values[j] = 0.0f;
                if (global_k < k_size) {
                    const unsigned char* p = packed + (size_t)global_n * row_bytes +
                                             (size_t)(global_k / 8) * 3;
                    const unsigned int b0 = p[0];
                    const unsigned int b1 = p[1];
                    const unsigned int b2 = p[2];
                    const int q0 = b0 & 0x07;
                    const int q1 = (b0 >> 3) & 0x07;
                    const int q2 = ((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2);
                    const int q3 = (b1 >> 1) & 0x07;
                    const int q4 = (b1 >> 4) & 0x07;
                    const int q5 = ((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1);
                    const int q6 = (b2 >> 2) & 0x07;
                    const int q7 = (b2 >> 5) & 0x07;
                    const int decoded[8] = {q0, q1, q2, q3, q4, q5, q6, q7};
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int global_k_j = global_k + j;
                        if (global_k_j < k_size) {
                            const float scale = __half2float(
                                scales[(size_t)global_n * groups_per_row +
                                       global_k_j / group_size]
                            );
                            values[j] = qwen2_w3_gemm_s3(decoded[j]) * scale;
                        }
                    }
                }
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    shared_b[local_k + j][local_n] = __float2half_rn(values[j]);
                }
            }
        }
        __syncthreads();

        const int local_m0 = row_in_tile * 2;
        const int local_n0 = col_group * 4;
        #pragma unroll
        for (int kk = 0; kk < QWEN2_W3_GEMM_BK; ++kk) {
            const float a0 = __half2float(shared_a[local_m0][kk]);
            const float a1 = __half2float(shared_a[local_m0 + 1][kk]);
            const float b0 = __half2float(shared_b[kk][local_n0 + 0]);
            const float b1 = __half2float(shared_b[kk][local_n0 + 1]);
            const float b2 = __half2float(shared_b[kk][local_n0 + 2]);
            const float b3 = __half2float(shared_b[kk][local_n0 + 3]);
            acc[0][0] = fmaf(a0, b0, acc[0][0]);
            acc[0][1] = fmaf(a0, b1, acc[0][1]);
            acc[0][2] = fmaf(a0, b2, acc[0][2]);
            acc[0][3] = fmaf(a0, b3, acc[0][3]);
            acc[1][0] = fmaf(a1, b0, acc[1][0]);
            acc[1][1] = fmaf(a1, b1, acc[1][1]);
            acc[1][2] = fmaf(a1, b2, acc[1][2]);
            acc[1][3] = fmaf(a1, b3, acc[1][3]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int global_m = block_m + row_in_tile * 2 + r;
        if (global_m >= m_size) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int global_n = block_n + col_group * 4 + j;
            if (global_n < n_rows) {
                c[(size_t)global_m * n_rows + global_n] = __float2half_rn(acc[r][j]);
            }
        }
    }
}
"""

_W3_RAW: Any = None
_W3_BATCHED_RAW: Any = None
_W3_GEMM_RAW: Any = None


def _w3_kernel() -> Any:
    global _W3_RAW
    if cp is None:
        raise RuntimeError("CuPy is required for the SM86 W3A16 backend")
    if _W3_RAW is None:
        _W3_RAW = cp.RawKernel(
            _W3_KERNEL, "qwen2_w3a16_gemv", options=("--std=c++14", "--use_fast_math")
        )
    return _W3_RAW


def _w3_batched_kernel() -> Any:
    global _W3_BATCHED_RAW
    if cp is None:
        raise RuntimeError("CuPy is required for the SM86 batched W3A16 kernel")
    if _W3_BATCHED_RAW is None:
        _W3_BATCHED_RAW = cp.RawKernel(
            _W3_KERNEL,
            "qwen2_w3a16_batched_gemv",
            options=("--std=c++14", "--use_fast_math"),
        )
    return _W3_BATCHED_RAW


_W3_BATCHED_BATCH_TILE = 8


def _w3_gemm_kernel() -> Any:
    global _W3_GEMM_RAW
    if cp is None:
        raise RuntimeError("CuPy is required for the tiled W3A16 GEMM")
    if _W3_GEMM_RAW is None:
        _W3_GEMM_RAW = cp.RawKernel(
            _SM86_W3A16_TILED_GEMM_KERNEL,
            "qwen2_w3a16_gemm",
            options=("--std=c++14", "--use_fast_math"),
        )
    return _W3_GEMM_RAW


def _native_w3a16_tiled_gemm(
    inputs: torch.Tensor,
    packed: Any,
    scales: Any,
    n_rows: int,
    k_size: int,
    group_size: int,
) -> torch.Tensor:
    """Native tiled W3A16 prefill GEMM without materializing dense weights."""

    if cp is None:
        raise RuntimeError("CuPy is unavailable")
    if inputs.ndim != 2 or inputs.shape[0] <= 0 or inputs.shape[1] != k_size:
        raise ValueError(
            f"tiled W3A16 GEMM expects [M, {k_size}], got {tuple(inputs.shape)}"
        )
    if k_size % 32 or k_size % 8:
        raise ValueError(f"tiled W3A16 GEMM requires K divisible by 32, got {k_size}")
    flat = inputs.contiguous().to(torch.float16)
    output = torch.empty(
        (int(inputs.shape[0]), int(n_rows)),
        dtype=torch.float16,
        device=inputs.device,
    )
    grid = ((int(n_rows) + 63) // 64, (int(inputs.shape[0]) + 31) // 32)
    stream = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    with stream:
        _w3_gemm_kernel()(
            grid,
            (256,),
            (
                _torch_to_cupy(flat),
                packed,
                scales,
                _torch_to_cupy(output),
                int(inputs.shape[0]),
                int(n_rows),
                int(k_size),
                int(group_size),
            ),
        )
    return output


class W3A16Linear(nn.Module):
    """Symmetric 3-bit weight + FP16 activation linear layer.

    Small-batch decode uses GEMV; multi-token prefill uses a tiled W3A16
    GEMM.  CPU and non-FP16 inputs retain the dequantize + cuBLAS fallback.
    """

    prefill_kernel = "tiled_w3a16_gemm"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        group_size: int = 128,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
        prefill_chunk_rows: int | None = 2048,
        gemv_batch_limit: int = 8,
    ) -> None:
        super().__init__()
        self.prefill_chunk_rows = prefill_chunk_rows
        self.gemv_batch_limit = gemv_batch_limit
        self.last_prefill_backend = "dequant_cublas"
        if in_features % 16 or in_features % group_size:
            raise ValueError("W3A16 in_features must be divisible by 16 and group_size")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.row_bytes = in_features * 3 // 8
        self._packed_cp = None
        self._scales_cp = None
        target_device = device or torch.empty(0).device
        self.register_buffer(
            "packed",
            torch.empty(
                out_features * self.row_bytes + 1,
                dtype=torch.uint8,
                device=target_device,
            ),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                (out_features, in_features // group_size),
                dtype=torch.float16,
                device=target_device,
            ),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype, device=target_device)
            )
        else:
            self.register_parameter("bias", None)

    def _apply(self, fn: Any, recurse: bool = True) -> Any:
        result = super()._apply(fn, recurse)
        self._packed_cp = None
        self._scales_cp = None
        return result

    def _bind_cupy(self) -> None:
        self._packed_cp = _torch_to_cupy(self.packed)
        self._scales_cp = _torch_to_cupy(self.scales)

    def load_from_packed(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        if tuple(packed.shape) != tuple(self.packed.shape):
            raise ValueError(
                f"3-bit packed shape mismatch: {tuple(packed.shape)} vs {tuple(self.packed.shape)}"
            )
        if tuple(scales.shape) != tuple(self.scales.shape):
            raise ValueError(
                f"3-bit scale shape mismatch: {tuple(scales.shape)} vs {tuple(self.scales.shape)}"
            )
        with torch.no_grad():
            self.packed.copy_(packed.to(device=self.packed.device, dtype=torch.uint8))
            self.scales.copy_(scales.to(device=self.scales.device, dtype=torch.float16))
            if self.bias is not None and bias is not None:
                self.bias.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))
            if cp is not None and self.packed.is_cuda:
                self._bind_cupy()

    def _dequantize_rows(self, start: int, end: int) -> torch.Tensor:
        rows = end - start
        flat = (
            self.packed[start * self.row_bytes : end * self.row_bytes]
            .reshape(rows, self.row_bytes)
            .reshape(rows, self.in_features // 8, 3)
            .to(torch.int32)
        )
        b0, b1, b2 = flat[:, :, 0], flat[:, :, 1], flat[:, :, 2]
        q = torch.empty(
            (rows, self.in_features // 8, 8),
            dtype=torch.int8,
            device=self.packed.device,
        )
        q[:, :, 0] = b0 & 7
        q[:, :, 1] = (b0 >> 3) & 7
        q[:, :, 2] = ((b0 >> 6) & 3) | ((b1 & 1) << 2)
        q[:, :, 3] = (b1 >> 1) & 7
        q[:, :, 4] = (b1 >> 4) & 7
        q[:, :, 5] = ((b1 >> 7) & 1) | ((b2 & 3) << 1)
        q[:, :, 6] = (b2 >> 2) & 7
        q[:, :, 7] = (b2 >> 5) & 7
        q = q.reshape(rows, self.in_features)
        q = torch.where(q >= 4, q - 8, q).to(torch.float16)
        grouped = q.view(rows, self.in_features // self.group_size, self.group_size)
        return (grouped * self.scales[start:end, :, None]).view(rows, self.in_features)

    def _dequantize(self) -> torch.Tensor:
        return self._dequantize_rows(0, self.out_features)

    def _forward_batched_gemv(self, inputs: torch.Tensor) -> torch.Tensor:
        """Experimental weight-reuse batch tile.

        Correctness-tested, but measured slower than the original
        ``grid.y=batch`` GEMV on the 3050 because the 8-way accumulator tile
        reduces occupancy.  It is intentionally not the default decode path.
        """

        if self._packed_cp is None or self._scales_cp is None:
            self._bind_cupy()
        flat = inputs.reshape(-1, self.in_features).to(torch.float16).contiguous()
        batch = int(flat.shape[0])
        output = torch.empty(
            (batch, self.out_features), dtype=torch.float16, device=flat.device
        )
        if self.in_features >= 16384:
            block_size, rows_per_block = 768, 24
        else:
            block_size, rows_per_block = 512, 16
        grid = (
            (self.out_features + rows_per_block - 1) // rows_per_block,
            (batch + _W3_BATCHED_BATCH_TILE - 1) // _W3_BATCHED_BATCH_TILE,
        )
        stream = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        with stream:
            _w3_batched_kernel()(
                grid,
                (block_size,),
                (
                    _torch_to_cupy(flat),
                    self._packed_cp,
                    self._scales_cp,
                    _torch_to_cupy(output),
                    self.out_features,
                    self.in_features,
                    self.group_size,
                    batch,
                ),
            )
        if self.bias is not None:
            output.add_(self.bias)
        self.last_prefill_backend = "w3a16_batched_gemv"
        return output

    def _forward_tiled_gemm(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._packed_cp is None or self._scales_cp is None:
            self._bind_cupy()
        flat = inputs.reshape(-1, self.in_features).contiguous().to(torch.float16)
        output = _native_w3a16_tiled_gemm(
            flat,
            self._packed_cp,
            self._scales_cp,
            self.out_features,
            self.in_features,
            self.group_size,
        )
        if self.bias is not None:
            output.add_(self.bias)
        self.last_prefill_backend = "tiled_w3a16_gemm"
        return output

    def forward_tiled_prefill(self, inputs: torch.Tensor) -> torch.Tensor:
        """Force the native tiled GEMM; useful for tests and benchmarks."""
        return self._forward_tiled_gemm(inputs)

    def _forward_dequant_cublas(self, inputs: torch.Tensor) -> torch.Tensor:
        self.last_prefill_backend = "dequant_cublas"
        flat = inputs.reshape(-1, self.in_features)
        chunk_rows = self.prefill_chunk_rows or self.out_features
        parts: list[torch.Tensor] = []
        for start in range(0, self.out_features, chunk_rows):
            end = min(start + chunk_rows, self.out_features)
            weight = self._dequantize_rows(start, end).to(flat.dtype)
            parts.append(F.linear(flat.to(torch.float16), weight, None))
            del weight
        output = torch.cat(parts, dim=-1)
        del parts
        if self.bias is not None:
            output = output + self.bias
        return output

    def _forward_gemv(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._packed_cp is None or self._scales_cp is None:
            self._bind_cupy()
        flat = inputs.reshape(-1, self.in_features).to(torch.float16).contiguous()
        batch = int(flat.shape[0])
        output = torch.empty(
            (batch, self.out_features), dtype=torch.float16, device=flat.device
        )
        kernel = _w3_kernel()
        # Large-K down_proj benefits from 24 rows/block; gate/up are best at
        # 16 rows/block on the 3050.
        if self.in_features >= 16384:
            block_size, rows_per_block = 768, 24
        else:
            block_size, rows_per_block = 512, 16
        grid = (
            (self.out_features + rows_per_block - 1) // rows_per_block,
            batch,
        )
        stream = cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
        with stream:
            kernel(
                grid,
                (block_size,),
                (
                    _torch_to_cupy(flat),
                    self._packed_cp,
                    self._scales_cp,
                    _torch_to_cupy(output),
                    self.out_features,
                    self.in_features,
                    self.group_size,
                ),
                shared_mem=self.in_features * 2,
            )
        if self.bias is not None:
            output.add_(self.bias)
        return output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_shape = inputs.shape
        flat = inputs.reshape(-1, self.in_features)
        can_use_sm86 = (
            flat.is_cuda
            and flat.dtype == torch.float16
            and cp is not None
            and self.in_features % 32 == 0
        )
        if can_use_sm86 and flat.shape[0] == 1:
            self.last_prefill_backend = "w3a16_gemv"
            output = self._forward_gemv(flat)
        elif can_use_sm86 and flat.shape[0] <= self.gemv_batch_limit:
            # The original grid.y=batch GEMV remains faster than the
            # weight-reuse batch-tile prototype on the 4 GB RTX 3050.
            self.last_prefill_backend = "w3a16_gemv"
            output = self._forward_gemv(flat)
        elif can_use_sm86 and flat.shape[0] > 1:
            output = self._forward_tiled_gemm(flat)
        else:
            output = self._forward_dequant_cublas(flat)
        return output.reshape(*original_shape[:-1], self.out_features)


class QuantizedEmbedding(nn.Module):
    """Int4 embedding table with on-the-fly row dequantization."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        group_size: int = 128,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        target_device = device or torch.empty(0).device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.group_size = group_size
        self.register_buffer(
            "packed",
            torch.empty(
                (vocab_size, hidden_size // 2), dtype=torch.uint8, device=target_device
            ),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                (vocab_size, hidden_size // group_size),
                dtype=torch.float16,
                device=target_device,
            ),
        )

    def load_from_packed(self, packed: torch.Tensor, scales: torch.Tensor) -> None:
        with torch.no_grad():
            self.packed.copy_(packed.to(device=self.packed.device, dtype=torch.uint8))
            self.scales.copy_(scales.to(device=self.scales.device, dtype=torch.float16))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        packed = self.packed[input_ids]
        scales = self.scales[input_ids]
        low = (packed & 0x0F).to(torch.int8)
        high = ((packed >> 4) & 0x0F).to(torch.int8)
        low = torch.where(low >= 8, low - 16, low)
        high = torch.where(high >= 8, high - 16, high)
        values = torch.empty(
            (*input_ids.shape, self.hidden_size),
            dtype=torch.float16,
            device=input_ids.device,
        )
        values[..., 0::2] = low
        values[..., 1::2] = high
        grouped = values.view(
            *input_ids.shape, self.hidden_size // self.group_size, self.group_size
        )
        return (grouped * scales[:, :, :, None]).view(
            *input_ids.shape, self.hidden_size
        )


class Qwen2MixedForCausalLM(Qwen2ForCausalLM):
    """Qwen2.5-7B mixed 4-bit attention/head + 3-bit MLP model_executor path."""

    source_backend = "qwen2_mixed_w4w3_sm86_model_executor"
    native_w4a16_kernel = True
    native_w3a16_kernel = True
    native_attention_kernel = True
    enable_qkv_fusion = True
    enable_gate_up_fusion = True
    native_runner_ready = False

    def __init__(
        self,
        cfg: Any,
        *,
        group_size: int = 128,
        attention_bits: int = 4,
        mlp_bits: int = 3,
        embed_bits: int = 4,
        head_bits: int = 3,
    ) -> None:
        self.group_size = group_size
        self.attention_bits = attention_bits
        self.mlp_bits = mlp_bits
        self.embed_bits = embed_bits
        self.head_bits = head_bits
        model_cfg = cfg.gpt if hasattr(cfg, "gpt") else cfg
        self.hidden_size = int(model_cfg.hidden_size)
        self.intermediate_size = int(model_cfg.intermediate_size)
        super().__init__(
            model_cfg,
            linear_factory=self._mixed_linear,
            embedding_factory=self._embedding,
            lm_head_factory=self._lm_head,
            fused_qkv_factory=self._fused_qkv,
            fused_gate_up_factory=self._fused_gate_up,
        )

    def _fused_qkv(self, in_features: int, out_features: int, bias: bool) -> nn.Module:
        cls = W4A16Linear if self.attention_bits == 4 else W3A16Linear
        return cls(
            in_features,
            out_features,
            bias,
            group_size=self.group_size,
            dtype=torch.float16,
            prefill_chunk_rows=2048,
        )

    def _fused_gate_up(
        self, in_features: int, out_features: int, bias: bool
    ) -> nn.Module:
        cls = W4A16Linear if self.mlp_bits == 4 else W3A16Linear
        return cls(
            in_features,
            out_features,
            bias,
            group_size=self.group_size,
            dtype=torch.float16,
            prefill_chunk_rows=2048,
        )

    def _mixed_linear(
        self, in_features: int, out_features: int, bias: bool
    ) -> nn.Module:
        bits = (
            self.mlp_bits
            if (
                out_features == self.intermediate_size
                or in_features == self.intermediate_size
            )
            else self.attention_bits
        )
        cls = W4A16Linear if bits == 4 else W3A16Linear
        return cls(
            in_features,
            out_features,
            bias,
            group_size=self.group_size,
            dtype=torch.float16,
            prefill_chunk_rows=2048,
        )

    def _embedding(self, vocab_size: int, hidden_size: int) -> nn.Module:
        return QuantizedEmbedding(vocab_size, hidden_size, group_size=self.group_size)

    def _lm_head(self, in_features: int, out_features: int, bias: bool) -> nn.Module:
        cls = W4A16Linear if self.head_bits == 4 else W3A16Linear
        return cls(
            in_features,
            out_features,
            bias,
            group_size=self.group_size,
            dtype=torch.float16,
            prefill_chunk_rows=2048,
        )

    def prepare_fused_layout(self) -> None:
        """Replace q/k/v and gate/up modules before checkpoint loading."""
        for layer in self.model.layers:
            attention = layer.self_attn
            q_proj, k_proj, v_proj = (
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
            )
            device = q_proj.packed.device
            fused_qkv = W4A16Linear(
                self.hidden_size,
                q_proj.out_features + k_proj.out_features + v_proj.out_features,
                q_proj.bias is not None,
                group_size=self.group_size,
                dtype=torch.float16,
                device=device,
                prefill_chunk_rows=2048,
            )
            attention.fused_qkv = fused_qkv
            del attention.q_proj, attention.k_proj, attention.v_proj
            del q_proj, k_proj, v_proj

            mlp = layer.mlp
            gate_proj, up_proj = mlp.gate_proj, mlp.up_proj
            device = gate_proj.packed.device
            fused_gate_up = W3A16Linear(
                self.hidden_size,
                2 * self.intermediate_size,
                False,
                group_size=self.group_size,
                dtype=torch.float16,
                device=device,
                prefill_chunk_rows=2048,
            )
            mlp.fused_gate_up = fused_gate_up
            del mlp.gate_proj, mlp.up_proj
            del gate_proj, up_proj, fused_qkv, fused_gate_up
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    def fuse_qkv(self) -> None:
        """Fuse q/k/v 4-bit projections without GPU allocation overlap."""
        for layer in self.model.layers:
            attention = layer.self_attn
            q_proj, k_proj, v_proj = (
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
            )
            device = q_proj.packed.device
            q_rows, k_rows, v_rows = (
                q_proj.out_features,
                k_proj.out_features,
                v_proj.out_features,
            )
            has_bias = q_proj.bias is not None
            q_packed, k_packed, v_packed = (
                q_proj.packed.cpu(),
                k_proj.packed.cpu(),
                v_proj.packed.cpu(),
            )
            q_scales, k_scales, v_scales = (
                q_proj.scales.cpu(),
                k_proj.scales.cpu(),
                v_proj.scales.cpu(),
            )
            q_bias = q_proj.bias.detach().cpu() if has_bias else None
            k_bias = k_proj.bias.detach().cpu() if has_bias else None
            v_bias = v_proj.bias.detach().cpu() if has_bias else None
            del attention.q_proj, attention.k_proj, attention.v_proj
            del q_proj, k_proj, v_proj
            torch.cuda.empty_cache()
            fused = W4A16Linear(
                self.hidden_size,
                q_rows + k_rows + v_rows,
                has_bias,
                group_size=self.group_size,
                dtype=torch.float16,
                device=device,
                prefill_chunk_rows=2048,
            )
            with torch.no_grad():
                fused.packed[:q_rows].copy_(q_packed.to(device))
                fused.packed[q_rows : q_rows + k_rows].copy_(k_packed.to(device))
                fused.packed[q_rows + k_rows :].copy_(v_packed.to(device))
                fused.scales[:q_rows].copy_(q_scales.to(device))
                fused.scales[q_rows : q_rows + k_rows].copy_(k_scales.to(device))
                fused.scales[q_rows + k_rows :].copy_(v_scales.to(device))
                if fused.bias is not None:
                    fused.bias[:q_rows].copy_(q_bias.to(device))
                    fused.bias[q_rows : q_rows + k_rows].copy_(k_bias.to(device))
                    fused.bias[q_rows + k_rows :].copy_(v_bias.to(device))
            fused._bind_cupy()
            attention.fused_qkv = fused
            del q_packed, k_packed, v_packed, q_scales, k_scales, v_scales
            del q_bias, k_bias, v_bias, fused
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    def fuse_gate_up(self) -> None:
        """Fuse gate/up 3-bit projections without GPU allocation overlap."""
        for layer in self.model.layers:
            mlp = layer.mlp
            gate_proj, up_proj = mlp.gate_proj, mlp.up_proj
            device = gate_proj.packed.device
            rows = gate_proj.out_features
            row_bytes = gate_proj.row_bytes
            gate_packed = gate_proj.packed[:-1].cpu()
            up_packed = up_proj.packed[:-1].cpu()
            gate_scales, up_scales = gate_proj.scales.cpu(), up_proj.scales.cpu()
            del mlp.gate_proj, mlp.up_proj
            del gate_proj, up_proj
            torch.cuda.empty_cache()
            fused = W3A16Linear(
                self.hidden_size,
                2 * rows,
                False,
                group_size=self.group_size,
                dtype=torch.float16,
                device=device,
                prefill_chunk_rows=2048,
            )
            with torch.no_grad():
                fused.packed[: rows * row_bytes].copy_(gate_packed.to(device))
                fused.packed[rows * row_bytes : 2 * rows * row_bytes].copy_(
                    up_packed.to(device)
                )
                fused.packed[-1] = 0
                fused.scales[:rows].copy_(gate_scales.to(device))
                fused.scales[rows:].copy_(up_scales.to(device))
            fused._bind_cupy()
            mlp.fused_gate_up = fused
            del gate_packed, up_packed, gate_scales, up_scales, fused
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        device: str | torch.device = "cuda",
        group_size: int = 128,
    ) -> Qwen2MixedForCausalLM:
        root = Path(checkpoint_dir)
        hf = json.loads((root / "config.json").read_text(encoding="utf-8"))
        cfg = build_qwen2_model_config(hf, root / "config.json")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        try:
            with torch.device(device):
                model = cls(
                    cfg,
                    group_size=group_size,
                    attention_bits=manifest["attention_bits"],
                    mlp_bits=manifest["mlp_bits"],
                    embed_bits=manifest["embed_bits"],
                    head_bits=manifest["lm_head_bits"],
                )
        finally:
            torch.set_default_dtype(previous_dtype)
        model.requires_grad_(False)
        for layer_index, layer_file in enumerate(manifest["layer_files"]):
            layer_root = f"model.layers.{layer_index}"
            attention = model.get_submodule(f"{layer_root}.self_attn")
            mlp = model.get_submodule(f"{layer_root}.mlp")
            qkv = attention.fused_qkv
            gate_up = mlp.fused_gate_up
            q_rows = attention.num_heads * attention.head_dim
            kv_rows = attention.num_kv_heads * attention.head_dim
            gate_rows = model.intermediate_size
            gate_row_bytes = gate_up.row_bytes if model.mlp_bits == 3 else 0
            with (
                safe_open(root / layer_file, framework="pt", device="cpu") as tensors,
                torch.no_grad(),
            ):
                qkv.packed[:q_rows].copy_(tensors.get_tensor("self_attn.q_proj.packed"))
                qkv.packed[q_rows : q_rows + kv_rows].copy_(
                    tensors.get_tensor("self_attn.k_proj.packed")
                )
                qkv.packed[q_rows + kv_rows :].copy_(
                    tensors.get_tensor("self_attn.v_proj.packed")
                )
                qkv.scales[:q_rows].copy_(tensors.get_tensor("self_attn.q_proj.scales"))
                qkv.scales[q_rows : q_rows + kv_rows].copy_(
                    tensors.get_tensor("self_attn.k_proj.scales")
                )
                qkv.scales[q_rows + kv_rows :].copy_(
                    tensors.get_tensor("self_attn.v_proj.scales")
                )
                if qkv.bias is not None:
                    qkv.bias[:q_rows].copy_(tensors.get_tensor("self_attn.q_proj.bias"))
                    qkv.bias[q_rows : q_rows + kv_rows].copy_(
                        tensors.get_tensor("self_attn.k_proj.bias")
                    )
                    qkv.bias[q_rows + kv_rows :].copy_(
                        tensors.get_tensor("self_attn.v_proj.bias")
                    )

                if model.mlp_bits == 4:
                    gate_up.packed[:gate_rows].copy_(
                        tensors.get_tensor("mlp.gate_proj.packed")
                    )
                    gate_up.packed[gate_rows:].copy_(
                        tensors.get_tensor("mlp.up_proj.packed")
                    )
                else:
                    gate_up.packed[: gate_rows * gate_row_bytes].copy_(
                        tensors.get_tensor("mlp.gate_proj.packed")[:-1]
                    )
                    gate_up.packed[
                        gate_rows * gate_row_bytes : 2 * gate_rows * gate_row_bytes
                    ].copy_(tensors.get_tensor("mlp.up_proj.packed")[:-1])
                    gate_up.packed[-1] = 0
                gate_up.scales[:gate_rows].copy_(
                    tensors.get_tensor("mlp.gate_proj.scales")
                )
                gate_up.scales[gate_rows:].copy_(
                    tensors.get_tensor("mlp.up_proj.scales")
                )

                attention.o_proj.load_from_packed(
                    tensors.get_tensor("self_attn.o_proj.packed"),
                    tensors.get_tensor("self_attn.o_proj.scales"),
                )
                mlp.down_proj.load_from_packed(
                    tensors.get_tensor("mlp.down_proj.packed"),
                    tensors.get_tensor("mlp.down_proj.scales"),
                )
                model.get_submodule(f"{layer_root}.input_layernorm").weight.copy_(
                    tensors.get_tensor("input_layernorm.weight")
                )
                model.get_submodule(
                    f"{layer_root}.post_attention_layernorm"
                ).weight.copy_(tensors.get_tensor("post_attention_layernorm.weight"))
            qkv._bind_cupy()
            gate_up._bind_cupy()
            torch.cuda.empty_cache()
            gc.collect()
            print(
                f"loaded layer {layer_index + 1}/{len(manifest['layer_files'])}",
                flush=True,
            )

        with safe_open(
            root / "embed.safetensors", framework="pt", device="cpu"
        ) as tensors:
            model.model.embed_tokens.load_from_packed(
                tensors.get_tensor("packed"), tensors.get_tensor("scales")
            )
        lm_head_file = manifest.get("lm_head_file", "lm_head.safetensors")
        with safe_open(root / lm_head_file, framework="pt", device="cpu") as tensors:
            model.lm_head.load_from_packed(
                tensors.get_tensor("packed"), tensors.get_tensor("scales")
            )
        with safe_open(
            root / "norm.safetensors", framework="pt", device="cpu"
        ) as tensors:
            model.model.norm.weight.copy_(tensors.get_tensor("weight"))
        for layer in model.model.layers:
            layer.self_attn.native_gqa_decode = _native_gqa_decode
        model.eval().requires_grad_(False)
        torch.cuda.empty_cache()
        gc.collect()
        return model


__all__ = ["QuantizedEmbedding", "Qwen2MixedForCausalLM", "W3A16Linear"]
