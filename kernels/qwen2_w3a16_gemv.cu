// Extracted from deepinfer/model_executor/models/qwen2_mixed_sm86.py:_W3_KERNEL

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
    if (lane == 0) out[row] = __float2half_rn(acc);
}
