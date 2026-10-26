"""Feasibility benchmark for a 3-bit MLP GEMV kernel on RTX 3050."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cupy as cp

KERNEL = r"""
#include <cuda_fp16.h>

__device__ __forceinline__ float s3(unsigned int value) {
    int v = (int)(value & 0x07);
    if (v >= 4) v -= 8;
    return (float)v;
}

__device__ __forceinline__ float warp_sum3(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

extern "C" __global__ void w3a16_gemv(
    const __half* __restrict__ x,
    const unsigned char* __restrict__ packed,
    const __half* __restrict__ scales,
    __half* __restrict__ out,
    int n_rows,
    int k_size,
    int group_size)
{
    extern __shared__ __half shared_x[];
    for (int i = threadIdx.x; i < k_size; i += blockDim.x) {
        shared_x[i] = x[i];
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + warp;
    if (row >= n_rows) return;

    const int row_bytes = k_size * 3 / 8;
    const int groups_per_row = k_size / group_size;
    const unsigned char* row_packed = packed + (size_t)row * row_bytes;
    const __half* row_scales = scales + (size_t)row * groups_per_row;
    const int vec = 8;
    const int stride = 32 * vec;
    float acc = 0.0f;

    for (int k0 = lane * vec; k0 < k_size; k0 += stride) {
        const unsigned char* b = row_packed + (size_t)(k0 / 8) * 3;
        const unsigned int b0 = b[0];
        const unsigned int b1 = b[1];
        const unsigned int b2 = b[2];
        const float scale = __half2float(row_scales[k0 / group_size]);
        const unsigned int q0 = b0 & 0x07;
        const unsigned int q1 = (b0 >> 3) & 0x07;
        const unsigned int q2 = ((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2);
        const unsigned int q3 = (b1 >> 1) & 0x07;
        const unsigned int q4 = (b1 >> 4) & 0x07;
        const unsigned int q5 = ((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1);
        const unsigned int q6 = (b2 >> 2) & 0x07;
        const unsigned int q7 = (b2 >> 5) & 0x07;
        acc += __half2float(shared_x[k0 + 0]) * scale * s3(q0);
        acc += __half2float(shared_x[k0 + 1]) * scale * s3(q1);
        acc += __half2float(shared_x[k0 + 2]) * scale * s3(q2);
        acc += __half2float(shared_x[k0 + 3]) * scale * s3(q3);
        acc += __half2float(shared_x[k0 + 4]) * scale * s3(q4);
        acc += __half2float(shared_x[k0 + 5]) * scale * s3(q5);
        acc += __half2float(shared_x[k0 + 6]) * scale * s3(q6);
        acc += __half2float(shared_x[k0 + 7]) * scale * s3(q7);
    }
    acc = warp_sum3(acc);
    if (lane == 0) out[row] = __float2half_rn(acc);
}
"""


KERNEL16 = r"""
#include <cuda_fp16.h>

__device__ __forceinline__ float s3_16(unsigned int value) {
    int v = (int)(value & 0x07);
    if (v >= 4) v -= 8;
    return (float)v;
}

__device__ __forceinline__ float warp_sum3_16(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

extern "C" __global__ void w3a16_gemv16(
    const __half* __restrict__ x,
    const unsigned char* __restrict__ packed,
    const __half* __restrict__ scales,
    __half* __restrict__ out,
    int n_rows,
    int k_size,
    int group_size)
{
    extern __shared__ __half shared_x[];
    for (int i = threadIdx.x; i < k_size; i += blockDim.x) shared_x[i] = x[i];
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
            acc += __half2float(shared_x[kk + 0]) * scale * s3_16(b0 & 0x07);
            acc += __half2float(shared_x[kk + 1]) * scale * s3_16((b0 >> 3) & 0x07);
            acc += __half2float(shared_x[kk + 2]) * scale * s3_16(((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2));
            acc += __half2float(shared_x[kk + 3]) * scale * s3_16((b1 >> 1) & 0x07);
            acc += __half2float(shared_x[kk + 4]) * scale * s3_16((b1 >> 4) & 0x07);
            acc += __half2float(shared_x[kk + 5]) * scale * s3_16(((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1));
            acc += __half2float(shared_x[kk + 6]) * scale * s3_16((b2 >> 2) & 0x07);
            acc += __half2float(shared_x[kk + 7]) * scale * s3_16((b2 >> 5) & 0x07);
        }
    }
    acc = warp_sum3_16(acc);
    if (lane == 0) out[row] = __float2half_rn(acc);
}
"""


KERNEL32 = r"""
#include <cuda_fp16.h>

__device__ __forceinline__ float s3_32(unsigned int value) {
    int v = (int)(value & 0x07);
    if (v >= 4) v -= 8;
    return (float)v;
}

__device__ __forceinline__ float warp_sum3_32(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__device__ __forceinline__ float acc8(const __half* x, const unsigned char* b, int kk, float scale) {
    const unsigned int b0 = b[0], b1 = b[1], b2 = b[2];
    float acc = 0.0f;
    acc += __half2float(x[kk + 0]) * scale * s3_32(b0 & 0x07);
    acc += __half2float(x[kk + 1]) * scale * s3_32((b0 >> 3) & 0x07);
    acc += __half2float(x[kk + 2]) * scale * s3_32(((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2));
    acc += __half2float(x[kk + 3]) * scale * s3_32((b1 >> 1) & 0x07);
    acc += __half2float(x[kk + 4]) * scale * s3_32((b1 >> 4) & 0x07);
    acc += __half2float(x[kk + 5]) * scale * s3_32(((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1));
    acc += __half2float(x[kk + 6]) * scale * s3_32((b2 >> 2) & 0x07);
    acc += __half2float(x[kk + 7]) * scale * s3_32((b2 >> 5) & 0x07);
    return acc;
}

extern "C" __global__ void w3a16_gemv32(
    const __half* __restrict__ x,
    const unsigned char* __restrict__ packed,
    const __half* __restrict__ scales,
    __half* __restrict__ out,
    int n_rows,
    int k_size,
    int group_size)
{
    extern __shared__ __half shared_x[];
    for (int i = threadIdx.x; i < k_size; i += blockDim.x) shared_x[i] = x[i];
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
    for (int k0 = lane * 32; k0 < k_size; k0 += 1024) {
        const float scale = __half2float(row_scales[k0 / group_size]);
        const unsigned char* group = row_packed + (size_t)(k0 / 8) * 3;
        acc += acc8(shared_x, group + 0, k0 + 0, scale);
        acc += acc8(shared_x, group + 3, k0 + 8, scale);
        acc += acc8(shared_x, group + 6, k0 + 16, scale);
        acc += acc8(shared_x, group + 9, k0 + 24, scale);
    }
    acc = warp_sum3_32(acc);
    if (lane == 0) out[row] = __float2half_rn(acc);
}
"""


def quantize_w3(weight: cp.ndarray, group_size: int) -> tuple[cp.ndarray, cp.ndarray]:
    n_rows, k_size = weight.shape
    grouped = weight.reshape(n_rows, k_size // group_size, group_size).astype(
        cp.float32
    )
    scales = cp.maximum(cp.max(cp.abs(grouped), axis=2) / 3.0, cp.float32(1e-8)).astype(
        cp.float16
    )
    q = (
        cp.clip(cp.rint(grouped / scales[:, :, None].astype(cp.float32)), -4, 3)
        .astype(cp.int8)
        .reshape(n_rows, k_size)
    )
    q3 = (q & 0x07).astype(cp.uint8).reshape(n_rows, k_size // 8, 8)
    row_bytes = k_size * 3 // 8
    # A single global padding byte lets the kernel perform an unaligned
    # 3-byte read at the end of the last row without changing row stride.
    packed = cp.empty(n_rows * row_bytes + 1, dtype=cp.uint8)
    rows = packed[:-1].reshape(n_rows, row_bytes)
    rows[:, 0::3] = q3[:, :, 0] | (q3[:, :, 1] << 3) | ((q3[:, :, 2] & 0x03) << 6)
    rows[:, 1::3] = (
        (q3[:, :, 2] >> 2)
        | (q3[:, :, 3] << 1)
        | (q3[:, :, 4] << 4)
        | ((q3[:, :, 5] & 0x01) << 7)
    )
    rows[:, 2::3] = (q3[:, :, 5] >> 1) | (q3[:, :, 6] << 2) | (q3[:, :, 7] << 5)
    packed[-1] = 0
    return cp.ascontiguousarray(packed), cp.ascontiguousarray(scales)


def dequantize_w3(
    packed: cp.ndarray, scales: cp.ndarray, n_rows: int, k_size: int
) -> cp.ndarray:
    row_bytes = k_size * 3 // 8
    flat = (
        packed[:-1]
        .reshape(n_rows, row_bytes)
        .reshape(n_rows, k_size // 8, 3)
        .astype(cp.uint32)
    )
    b0, b1, b2 = flat[:, :, 0], flat[:, :, 1], flat[:, :, 2]
    q = cp.empty((n_rows, k_size // 8, 8), dtype=cp.int8)
    q[:, :, 0] = b0 & 7
    q[:, :, 1] = (b0 >> 3) & 7
    q[:, :, 2] = ((b0 >> 6) & 3) | ((b1 & 1) << 2)
    q[:, :, 3] = (b1 >> 1) & 7
    q[:, :, 4] = (b1 >> 4) & 7
    q[:, :, 5] = ((b1 >> 7) & 1) | ((b2 & 3) << 1)
    q[:, :, 6] = (b2 >> 2) & 7
    q[:, :, 7] = (b2 >> 5) & 7
    q = q.reshape(n_rows, k_size)
    q = cp.where(q >= 4, q - 8, q).astype(cp.float32)
    return (
        q.reshape(n_rows, k_size // 128, 128) * scales.astype(cp.float32)[:, :, None]
    ).reshape(n_rows, k_size)


def time_kernel(
    kernel: cp.RawKernel,
    x: cp.ndarray,
    packed: cp.ndarray,
    scales: cp.ndarray,
    out: cp.ndarray,
    *,
    repeats: int,
) -> dict[str, float]:
    n_rows = out.size
    k_size = x.size
    grid = ((n_rows + 7) // 8,)
    args = (x, packed, scales, out, n_rows, k_size, 128)
    for _ in range(5):
        kernel(grid, (256,), args, shared_mem=k_size * 2)
    cp.cuda.runtime.deviceSynchronize()
    starts = [cp.cuda.Event() for _ in range(repeats)]
    ends = [cp.cuda.Event() for _ in range(repeats)]
    for index in range(repeats):
        starts[index].record()
        kernel(grid, (256,), args, shared_mem=k_size * 2)
        ends[index].record()
    ends[-1].synchronize()
    samples = [cp.cuda.get_elapsed_time(starts[i], ends[i]) for i in range(repeats)]
    return {
        "median_ms": round(statistics.median(samples), 6),
        "min_ms": round(min(samples), 6),
        "max_ms": round(max(samples), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SM86 3-bit MLP GEMV feasibility")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--kernel", choices=("v8", "v16", "v32"), default="v32")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "doc"
        / "results"
        / "w3a16_sm86_microbench.json",
    )
    args = parser.parse_args()
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(
        f"GPU={props['name'].decode()} SM={props['major']}.{props['minor']}", flush=True
    )
    if args.kernel == "v32":
        kernel = cp.RawKernel(
            KERNEL32, "w3a16_gemv32", options=("--std=c++14", "--use_fast_math")
        )
    elif args.kernel == "v16":
        kernel = cp.RawKernel(
            KERNEL16, "w3a16_gemv16", options=("--std=c++14", "--use_fast_math")
        )
    else:
        kernel = cp.RawKernel(
            KERNEL, "w3a16_gemv", options=("--std=c++14", "--use_fast_math")
        )

    # Correctness check against an explicit dequantized matvec.
    cp.random.seed(7)
    n_check, k_check = 256, 3584
    w_check = (cp.random.randn(n_check, k_check, dtype=cp.float32) * 0.02).astype(
        cp.float16
    )
    x_check = cp.random.randn(k_check, dtype=cp.float32).astype(cp.float16)
    p_check, s_check = quantize_w3(w_check, 128)
    y_check = cp.empty(n_check, dtype=cp.float16)
    kernel(
        ((n_check + 7) // 8,),
        (256,),
        (x_check, p_check, s_check, y_check, n_check, k_check, 128),
        shared_mem=k_check * 2,
    )
    cp.cuda.runtime.deviceSynchronize()
    ref = dequantize_w3(p_check, s_check, n_check, k_check).astype(cp.float16) @ x_check
    ref = ref.astype(cp.float32)
    got = y_check.astype(cp.float32)
    diff = cp.abs(ref - got)
    cosine = float(cp.dot(ref, got) / (cp.linalg.norm(ref) * cp.linalg.norm(got)))
    correctness = {
        "cosine": round(cosine, 6),
        "max_abs": round(float(diff.max()), 6),
        "mean_abs": round(float(diff.mean()), 6),
    }
    print("correctness", correctness, flush=True)

    shapes = {
        "gate_proj": (18944, 3584),
        "up_proj": (18944, 3584),
        "down_proj": (3584, 18944),
    }
    results: dict[str, Any] = {
        "device": props["name"].decode(),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "quantization": "symmetric int3, 3 bytes per 8 values, group 128",
        "correctness": correctness,
        "shapes": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for name, (n_rows, k_size) in shapes.items():
        weight = (cp.random.randn(n_rows, k_size, dtype=cp.float32) * 0.02).astype(
            cp.float16
        )
        x = cp.random.randn(k_size, dtype=cp.float32).astype(cp.float16)
        packed, scales = quantize_w3(weight, 128)
        out = cp.empty(n_rows, dtype=cp.float16)
        timing = time_kernel(kernel, x, packed, scales, out, repeats=args.repeats)
        timing.update({"n_rows": n_rows, "k_size": k_size})
        results["shapes"][name] = timing
        print(name, timing, flush=True)
        del weight, x, packed, scales, out
        cp.get_default_memory_pool().free_all_blocks()

    mlp_ms = sum(results["shapes"][name]["median_ms"] for name in shapes)
    results["projection"] = {
        "mlp_per_layer_ms": round(mlp_ms, 6),
        "28_layers_mlp_only_ms": round(28 * mlp_ms, 6),
        "28_layers_mlp_only_tokens_per_second": round(1000.0 / (28 * mlp_ms), 3),
        "note": "Excludes 4-bit attention, embedding/lm_head, RMSNorm, RoPE, KV cache and launch overhead.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results["projection"], indent=2), flush=True)
    print(f"output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
