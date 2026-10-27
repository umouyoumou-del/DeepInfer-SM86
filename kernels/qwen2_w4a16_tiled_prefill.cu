// Extracted from deepinfer/model_executor/models/qwen2_w4a16.py:_SM86_W4A16_TILED_GEMM_KERNEL

#include <cuda_fp16.h>

#define QWEN2_GEMM_BM 32
#define QWEN2_GEMM_BN 64
#define QWEN2_GEMM_BK 32
#define QWEN2_GEMM_THREADS 256

__device__ __forceinline__ float qwen2_gemm_int4(unsigned int value) {
    int v = (int)(value & 0x0f);
    if (v >= 8) v -= 16;
    return (float)v;
}

// C[M, N] = A[M, K] @ dequantize(B[N, K]) with FP32 accumulation.
//
// Every CTA owns a 32x64 output tile.  For each K tile it dequantizes the
// matching packed-weight tile directly into shared memory.  Peak temporary
// weight storage is O(BK * BN) instead of O(N * K), which is the property the
// previous per-output-row dequantize + cuBLAS prefill path lacked.
extern "C" __global__ void qwen2_w4a16_gemm(
    const __half* __restrict__ a,
    const unsigned char* __restrict__ packed,
    const __half* __restrict__ scales,
    __half* __restrict__ c,
    int m_size,
    int n_rows,
    int k_size,
    int group_size)
{
    __shared__ __half shared_a[QWEN2_GEMM_BM][QWEN2_GEMM_BK];
    __shared__ __half shared_b[QWEN2_GEMM_BK][QWEN2_GEMM_BN + 1];

    const int tid = threadIdx.x;
    const int row_in_tile = tid >> 4;
    const int col_group = tid & 15;
    const int block_m = blockIdx.y * QWEN2_GEMM_BM;
    const int block_n = blockIdx.x * QWEN2_GEMM_BN;
    const int groups_per_row = k_size / group_size;

    float acc[2][4];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[r][j] = 0.0f;
    }

    for (int k0 = 0; k0 < k_size; k0 += QWEN2_GEMM_BK) {
        #pragma unroll
        for (int index = tid; index < QWEN2_GEMM_BM * QWEN2_GEMM_BK;
             index += QWEN2_GEMM_THREADS) {
            const int local_m = index / QWEN2_GEMM_BK;
            const int local_k = index % QWEN2_GEMM_BK;
            const int global_m = block_m + local_m;
            const int global_k = k0 + local_k;
            shared_a[local_m][local_k] =
                (global_m < m_size && global_k < k_size)
                    ? a[(size_t)global_m * k_size + global_k]
                    : __float2half_rn(0.0f);
        }

        // A warp handles two B rows at a time: lane 0..15 cover the 16 packed
        // bytes (32 K values) of one row.
        const int lane = tid & 31;
        const int warp = tid >> 5;
        #pragma unroll
        for (int n_base = warp * 2; n_base < QWEN2_GEMM_BN; n_base += 16) {
            const int local_n = n_base + (lane >> 4);
            const int byte_in_tile = lane & 15;
            const int global_n = block_n + local_n;
            if (global_n < n_rows) {
                const int local_k = byte_in_tile * 2;
                const int global_k = k0 + local_k;
                float v0 = 0.0f;
                float v1 = 0.0f;
                if (global_k < k_size) {
                    const unsigned int byte = packed[
                        (size_t)global_n * (k_size / 2) + (k0 / 2) + byte_in_tile
                    ];
                    const float scale0 = __half2float(
                        scales[(size_t)global_n * groups_per_row + global_k / group_size]
                    );
                    v0 = qwen2_gemm_int4(byte & 0x0f) * scale0;
                    if (global_k + 1 < k_size) {
                        const float scale1 = __half2float(
                            scales[(size_t)global_n * groups_per_row +
                                   (global_k + 1) / group_size]
                        );
                        v1 = qwen2_gemm_int4(byte >> 4) * scale1;
                    }
                }
                shared_b[local_k][local_n] = __float2half_rn(v0);
                shared_b[local_k + 1][local_n] = __float2half_rn(v1);
            }
        }
        __syncthreads();

        const int local_m0 = row_in_tile * 2;
        const int local_n0 = col_group * 4;
        #pragma unroll
        for (int kk = 0; kk < QWEN2_GEMM_BK; ++kk) {
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
