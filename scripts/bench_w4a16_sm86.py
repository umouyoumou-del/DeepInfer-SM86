"""Microbenchmark a SM86 W4A16 GEMV kernel with Qwen2.5 layer shapes.

This is a feasibility probe for a future DeepInfer native Qwen2 SM86 backend.
It uses CuPy only to compile and launch a standalone CUDA kernel; it does not
depend on Ollama, llama.cpp, ExLlama, Triton or bitsandbytes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cupy as cp

KERNEL_BLOCK = r"""
#include <cuda_fp16.h>

__device__ __forceinline__ float int4_to_float(unsigned int value) {
    int v = (int)(value & 0x0f);
    if (v >= 8) v -= 16;
    return (float)v;
}

__device__ __forceinline__ float block_reduce_sum(float value) {
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    value = (threadIdx.x < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
    }
    return value;
}

extern "C" __global__ void w4a16_gemv(
    const __half* __restrict__ x,
    const unsigned char* __restrict__ packed,
    const __half* __restrict__ scales,
    __half* __restrict__ out,
    int n_rows,
    int k_size,
    int group_size,
    int rows_per_warp)
{
    (void)rows_per_warp;
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    float acc = 0.0f;
    const int vec = 8;
    const int stride = blockDim.x * vec;
    const int groups_per_row = k_size / group_size;
    const unsigned char* row_packed = packed + (size_t)row * (k_size / 2);
    const __half* row_scales = scales + (size_t)row * groups_per_row;

    for (int k0 = threadIdx.x * vec; k0 < k_size; k0 += stride) {
        const size_t packed_offset = (size_t)k0 / 2;
        const unsigned int word = *reinterpret_cast<const unsigned int*>(row_packed + packed_offset);
        const float scale = __half2float(row_scales[k0 / group_size]);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int kk = k0 + j * 2;
            acc += __half2float(x[kk]) * scale * int4_to_float((word >> (j * 8)) & 0x0f);
            acc += __half2float(x[kk + 1]) * scale * int4_to_float((word >> (j * 8 + 4)) & 0x0f);
        }
    }
    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0) out[row] = __float2half_rn(acc);
}
"""

KERNEL_SHARED = r"""
#include <cuda_fp16.h>

__device__ __forceinline__ float int4_to_float_shared(unsigned int value) {
    int v = (int)(value & 0x0f);
    if (v >= 8) v -= 16;
    return (float)v;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

extern "C" __global__ void w4a16_gemv_shared(
    const __half* __restrict__ x,
    const unsigned char* __restrict__ packed,
    const __half* __restrict__ scales,
    __half* __restrict__ out,
    int n_rows,
    int k_size,
    int group_size,
    int rows_per_warp)
{
    extern __shared__ __half shared_x[];
    for (int i = threadIdx.x; i < k_size; i += blockDim.x) {
        shared_x[i] = x[i];
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int first_row = (blockIdx.x * warps_per_block + warp) * rows_per_warp;
    const int groups_per_row = k_size / group_size;
    const int vec = 8;
    const int stride = 32 * vec;

    for (int local_row = 0; local_row < rows_per_warp; ++local_row) {
        const int row = first_row + local_row;
        if (row >= n_rows) break;
        const unsigned char* row_packed = packed + (size_t)row * (k_size / 2);
        const __half* row_scales = scales + (size_t)row * groups_per_row;
        float acc = 0.0f;
        for (int k0 = lane * vec; k0 < k_size; k0 += stride) {
            const unsigned int word = *reinterpret_cast<const unsigned int*>(
                row_packed + (size_t)k0 / 2
            );
            const float scale = __half2float(row_scales[k0 / group_size]);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int kk = k0 + j * 2;
                acc += __half2float(shared_x[kk]) * scale * int4_to_float_shared((word >> (j * 8)) & 0x0f);
                acc += __half2float(shared_x[kk + 1]) * scale * int4_to_float_shared((word >> (j * 8 + 4)) & 0x0f);
            }
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) out[row] = __float2half_rn(acc);
    }
}
"""


def quantize_w4a16(
    weight: cp.ndarray, group_size: int
) -> tuple[cp.ndarray, cp.ndarray]:
    n_rows, k_size = weight.shape
    grouped = weight.reshape(n_rows, k_size // group_size, group_size).astype(
        cp.float32
    )
    absmax = cp.max(cp.abs(grouped), axis=2)
    scales = cp.maximum(absmax / 7.0, cp.float32(1e-8)).astype(cp.float16)
    normalized = grouped / scales[:, :, None].astype(cp.float32)
    quantized = (
        cp.clip(cp.rint(normalized), -8, 7).astype(cp.int8).reshape(n_rows, k_size)
    )
    low = (quantized[:, 0::2] & 0x0F).astype(cp.uint8)
    high = ((quantized[:, 1::2] & 0x0F) << 4).astype(cp.uint8)
    packed = (low | high).reshape(n_rows, k_size // 2)
    return cp.ascontiguousarray(packed), cp.ascontiguousarray(scales)


def time_kernel(
    kernel: cp.RawKernel,
    x: cp.ndarray,
    packed: cp.ndarray,
    scales: cp.ndarray,
    out: cp.ndarray,
    block: int,
    rows_per_block: int,
    shared_bytes: int,
    *,
    warmup: int = 5,
    repeats: int = 30,
) -> dict[str, float]:
    n_rows, k_size = packed.shape[0], x.size
    n_blocks = (n_rows + rows_per_block - 1) // rows_per_block
    launch_args = (x, packed, scales, out, n_rows, k_size, 128, rows_per_block)
    for _ in range(warmup):
        kernel((n_blocks,), (block,), launch_args, shared_mem=shared_bytes)
    cp.cuda.runtime.deviceSynchronize()
    starts = [cp.cuda.Event() for _ in range(repeats)]
    ends = [cp.cuda.Event() for _ in range(repeats)]
    for index in range(repeats):
        starts[index].record()
        kernel((n_blocks,), (block,), launch_args, shared_mem=shared_bytes)
        ends[index].record()
    ends[-1].synchronize()
    samples = [cp.cuda.get_elapsed_time(starts[i], ends[i]) for i in range(repeats)]
    return {
        "median_ms": round(statistics.median(samples), 6),
        "min_ms": round(min(samples), 6),
        "max_ms": round(max(samples), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark SM86 W4A16 GEMV for Qwen2.5-3B"
    )
    parser.add_argument("--kernel", choices=("block", "shared"), default="shared")
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--rows-per-warp", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "doc"
        / "results"
        / "w4a16_sm86_microbench.json",
    )
    args = parser.parse_args()

    if not cp.cuda.is_available():
        raise SystemExit("CuPy CUDA is unavailable")
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    print(
        f"GPU={props['name'].decode()} SM={props['major']}.{props['minor']}", flush=True
    )
    if args.kernel == "shared":
        source, entry = KERNEL_SHARED, "w4a16_gemv_shared"
    else:
        source, entry = KERNEL_BLOCK, "w4a16_gemv"
    kernel = cp.RawKernel(source, entry, options=("--std=c++14", "--use_fast_math"))

    shapes = {
        "q_proj": (2048, 2048),
        "k_proj": (256, 2048),
        "v_proj": (256, 2048),
        "o_proj": (2048, 2048),
        "gate_proj": (11008, 2048),
        "up_proj": (11008, 2048),
        "down_proj": (2048, 11008),
        "lm_head": (151936, 2048),
    }
    results: dict[str, Any] = {
        "device": props["name"].decode(),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "kernel": args.kernel,
        "block": args.block,
        "rows_per_warp": args.rows_per_warp if args.kernel == "shared" else 1,
        "group_size": 128,
        "repeats": args.repeats,
        "shapes": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for name, (n_rows, k_size) in shapes.items():
        weight = (cp.random.randn(n_rows, k_size, dtype=cp.float32) * 0.02).astype(
            cp.float16
        )
        x = cp.random.randn(k_size, dtype=cp.float32).astype(cp.float16)
        packed, scales = quantize_w4a16(weight, 128)
        out = cp.empty(n_rows, dtype=cp.float16)
        warps_per_block = max(1, args.block // 32)
        rows_per_block = (
            warps_per_block * max(1, args.rows_per_warp)
            if args.kernel == "shared"
            else 1
        )
        shared_bytes = k_size * 2 if args.kernel == "shared" else 0
        timing = time_kernel(
            kernel,
            x,
            packed,
            scales,
            out,
            args.block,
            rows_per_block,
            shared_bytes,
            repeats=args.repeats,
        )
        timing.update({"n_rows": n_rows, "k_size": k_size})
        results["shapes"][name] = timing
        del weight, x, packed, scales, out
        cp.get_default_memory_pool().free_all_blocks()
        print(f"{name}: {timing}", flush=True)

    # One Qwen2.5-3B layer = q/k/v/o + gate/up/down. lm_head is once per token.
    per_layer_ms = sum(
        results["shapes"][name]["median_ms"]
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
    )
    total_ms = 36 * per_layer_ms + results["shapes"]["lm_head"]["median_ms"]
    results["projection"] = {
        "per_layer_gemv_ms": round(per_layer_ms, 6),
        "36_layers_plus_lm_head_ms": round(total_ms, 6),
        "gemv_only_tokens_per_second": round(1000.0 / total_ms, 3),
        "note": "Upper-bound feasibility estimate only; attention, RMSNorm, RoPE, sampling, KV-cache and launch overhead are excluded.",
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
