// ============================================================================
// STATUS: NOT VALIDATED - DO NOT ENABLE
//
// This file compiles only where the CUDA *toolkit* headers are installed.
// <mma.h> pulls in "crt/mma.h", which ships with nvcc, not with the pip
// nvidia-cuda-runtime-cu12 package that CuPy's JIT uses on this machine.
// Attempting to build it here fails with:
//
//   mma.h(55): catastrophic error: cannot open source file "crt/mma.h"
//
// To unblock, either install a full CUDA toolkit (so CUDA_PATH/include/crt/mma.h
// exists) or the nvidia-cuda-nvcc-cu12 wheel, then run the numeric comparison
// against kernels/qwen2_w4a16_tiled_prefill.cu before wiring it into
// _native_w4a16_tiled_gemm.  It has never been executed, so its fragment
// mapping and the store epilogue are unchecked.
//
// Alternative if headers stay unavailable: rewrite with inline PTX
// mma.sync.aligned.m16n8k16 and no <mma.h> dependency.
// ============================================================================

// Tensor Core variant of qwen2_w4a16_gemm (tiled prefill).
//
// Same 32x64 CTA tile and the same int4 dequantization into shared memory as
// kernels/qwen2_w4a16_tiled_prefill.cu, but the inner product runs through
// wmma 16x16x16 FP16->FP32 MMA instead of per-thread FMA on CUDA cores.
// Eight warps cover the 2x4 grid of 16x16 accumulator fragments.

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define TC_BM 32
#define TC_BN 64
#define TC_BK 32
#define TC_THREADS 256

__device__ __forceinline__ float tc_gemm_int4(unsigned int value) {
    int v = (int)(value & 0x0f);
    if (v >= 8) v -= 16;
    return (float)v;
}

extern "C" __global__ void qwen2_w4a16_gemm_tc(
    const __half* __restrict__ a,
    const unsigned char* __restrict__ packed,
    const __half* __restrict__ scales,
    __half* __restrict__ c,
    int m_size,
    int n_rows,
    int k_size,
    int group_size)
{
    __shared__ __half sa[TC_BM][TC_BK];
    __shared__ __half sb[TC_BK][TC_BN];
    __shared__ float sc[TC_BM][TC_BN];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int frag_m = (warp >> 2) * 16;
    const int frag_n = (warp & 3) * 16;
    const int block_m = blockIdx.y * TC_BM;
    const int block_n = blockIdx.x * TC_BN;
    const int groups_per_row = k_size / group_size;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < k_size; k0 += TC_BK) {
        for (int index = tid; index < TC_BM * TC_BK; index += TC_THREADS) {
            const int local_m = index / TC_BK;
            const int local_k = index % TC_BK;
            const int global_m = block_m + local_m;
            const int global_k = k0 + local_k;
            sa[local_m][local_k] =
                (global_m < m_size && global_k < k_size)
                    ? a[(size_t)global_m * k_size + global_k]
                    : __float2half_rn(0.0f);
        }

        for (int n_base = warp * 2; n_base < TC_BN; n_base += 16) {
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
                        scales[(size_t)global_n * groups_per_row +
                               global_k / group_size]
                    );
                    v0 = tc_gemm_int4(byte & 0x0f) * scale0;
                    if (global_k + 1 < k_size) {
                        const float scale1 = __half2float(
                            scales[(size_t)global_n * groups_per_row +
                                   (global_k + 1) / group_size]
                        );
                        v1 = tc_gemm_int4(byte >> 4) * scale1;
                    }
                }
                sb[local_k][local_n] = __float2half_rn(v0);
                sb[local_k + 1][local_n] = __float2half_rn(v1);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TC_BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
            wmma::load_matrix_sync(fa, &sa[frag_m][kk], TC_BK);
            wmma::load_matrix_sync(fb, &sb[kk][frag_n], TC_BN);
            wmma::mma_sync(acc, fa, fb, acc);
        }
        __syncthreads();
    }

    wmma::store_matrix_sync(&sc[frag_m][frag_n], acc, TC_BN, wmma::mem_row_major);
    __syncthreads();

    for (int index = tid; index < TC_BM * TC_BN; index += TC_THREADS) {
        const int local_m = index / TC_BN;
        const int local_n = index % TC_BN;
        const int global_m = block_m + local_m;
        const int global_n = block_n + local_n;
        if (global_m < m_size && global_n < n_rows) {
            c[(size_t)global_m * n_rows + global_n] =
                __float2half_rn(sc[local_m][local_n]);
        }
    }
}
