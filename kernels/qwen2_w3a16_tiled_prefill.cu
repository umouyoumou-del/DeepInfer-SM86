// Extracted from deepinfer/model_executor/models/qwen2_mixed_sm86.py:_SM86_W3A16_TILED_GEMM_KERNEL

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
