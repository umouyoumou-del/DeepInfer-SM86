// Extracted from deepinfer/model_executor/models/qwen2_w4a16.py:_SM86_W4A16_KERNEL

#include <cuda_fp16.h>

__device__ __forceinline__ float int4_value(unsigned int value) {
    int v = (int)(value & 0x0f);
    if (v >= 8) v -= 16;
    return (float)v;
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

extern "C" __global__ void qwen2_w4a16_gemv(
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

    const int groups_per_row = k_size / group_size;
    const unsigned char* row_packed = packed + (size_t)row * (k_size / 2);
    const __half* row_scales = scales + (size_t)row * groups_per_row;
    const int vec = 8;
    const int stride = 32 * vec;
    float acc = 0.0f;

    for (int k0 = lane * vec; k0 < k_size; k0 += stride) {
        const unsigned int word = *reinterpret_cast<const unsigned int*>(
            row_packed + (size_t)k0 / 2
        );
        const float scale = __half2float(row_scales[k0 / group_size]);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int kk = k0 + j * 2;
            acc += __half2float(shared_x[kk]) * scale * int4_value((word >> (j * 8)) & 0x0f);
            acc += __half2float(shared_x[kk + 1]) * scale * int4_value((word >> (j * 8 + 4)) & 0x0f);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) out[row] = __float2half_rn(acc);
}
