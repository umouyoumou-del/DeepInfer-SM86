"""Batched prefill for the fused OLMoE decoder: the prompt in one pass on tensor cores.

The fused decoder (olmoe_fused) reads a prompt one token at a time through its
decode step, about 640 tok/s on the 4080 SUPER, where llama.cpp's pp512 runs
about 20,000.  Decode is bound by weight bandwidth; prefill is bound by
arithmetic.  For 512 tokens a layer is a [512 x 2048] x [2048 x 2048] product
per attention matrix and, after routing, 64 expert products of about 64 tokens
each: roughly 1.1 TFLOP for the model.

So every matrix goes through one tensor-core GEMM that dequantises its weight
tile.  Each K step decodes 64 elements of every weight row in the tile from
Q4_K/Q6_K into fp16 in shared memory and multiplies with WMMA (fp16 inputs,
fp32 accumulation).  A dense projection is one group.  The experts are 64
groups whose row tiles come from a table the routing builds on the device
(count tokens per expert, scan, scatter), so nothing waits on the host.

Per layer:

  1. RMSNorm, folding in the previous layer's expert outputs
  2. q/k/v GEMM
  3. q/k norm, RoPE and the cache write
  4. causal attention (torch SDPA)
  5. output GEMM, added into the residual stream
  6. RMSNorm
  7. router (torch)
  8. dispatch
  9. gate/up GEMM
 10. down GEMM

The last token's state is handed to the decoder's own LM head and argmax, so
decoding continues from the filled cache.  Activations enter the GEMMs in fp16
(the decode path keeps them in fp32) and weights are rounded to fp16 in shared
memory; the residual stream stays fp32.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch

from deepinfer.moe.gpu_quant import _to_cupy
from deepinfer.moe.olmoe_fused import _HELPERS, _QTYPES, cp, fused_supported
from deepinfer.moe.olmoe_fused import _launch as _decode_launch

_BM = 64  # GEMM rows (tokens) per tile; the dispatch cuts expert groups to this

_SOURCE = _HELPERS + r"""
#define PF_BM 64
#define PF_BN 64
#define PF_BK 64
#define PF_LDS (PF_BK + 8)
#define PF_LDC (PF_BN + 4)

// mma.sync / ldmatrix by inline PTX (sm_80+): the WMMA header is not shipped
// with the pip CUDA runtime that CuPy compiles against, and the fragment
// layouts below are the documented ones, which WMMA's are not.
__device__ __forceinline__ unsigned int pf_smem(const void* p)
{
    unsigned int addr;
    asm("{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }" : "=r"(addr) : "l"(p));
    return addr;
}

__device__ __forceinline__ void pf_ldmatrix_x4(unsigned int (&r)[4], const __half* p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(pf_smem(p)));
}

// c (16 x 8, fp32) += a (16 x 16, row) . b (16 x 8, col)
__device__ __forceinline__ void pf_mma(float (&c)[4], const unsigned int (&a)[4],
                                       unsigned int b0, unsigned int b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 32 elements of one weight row, k0 .. k0 + 63 split in two parts, as fp16.
// Weights are fetched one K step ahead into registers, so their loads are in
// flight during the previous step's MMA; then dequantised into fp16 in shared
// memory.  A thread owns half a tile row and 32 of the K step's 64 elements.
// Q4_K: chunk c = (k0 % 256) / 64 is quant pair c, part 0 its low-nibble
// sub-block 2c, part 1 its high-nibble sub-block 2c + 1 (ggml order), from
// three 16-byte loads.
__device__ __forceinline__ void pf_fetch_q4k(
    const unsigned char* row, int k0, uint4& head, uint4& q0, uint4& q1)
{
    const unsigned char* block = row + (size_t)(k0 >> 8) * 144;
    if ((k0 & 255) == 0) head = *reinterpret_cast<const uint4*>(block);  // d, dmin, scales: once a super-block
    const uint4* qs = reinterpret_cast<const uint4*>(block + 16 + 32 * ((k0 >> 6) & 3));
    q0 = qs[0];
    q1 = qs[1];
}

__device__ __forceinline__ void pf_store_q4k(
    const uint4& head, const uint4& q0, const uint4& q1, int k0, int part, __half* dst)
{
    const int c = (k0 >> 6) & 3;
    __half2* out2 = reinterpret_cast<__half2*>(dst);
    const float d = __half2float(__ushort_as_half((unsigned short)(head.x & 0xFFFFu)));
    const float dmin = __half2float(__ushort_as_half((unsigned short)(head.x >> 16)));
    float scale, minimum;
    fz_q4k_scale_min(head.y, head.z, head.w, 2 * c + part, &scale, &minimum);
    const float a = d * scale, b = dmin * minimum;
    const unsigned int words[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    const int shift = part * 4;
    #pragma unroll
    for (int w = 0; w < 8; w += 2) {
        __half2 h[4];
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            const unsigned int v = words[w + u] >> shift;
            h[2 * u] = __floats2half2_rn(a * (float)(v & 0xF) - b, a * (float)((v >> 8) & 0xF) - b);
            h[2 * u + 1] = __floats2half2_rn(a * (float)((v >> 16) & 0xF) - b, a * (float)((v >> 24) & 0xF) - b);
        }
        reinterpret_cast<uint4*>(out2)[w >> 1] = *reinterpret_cast<const uint4*>(h);
    }
}

// Q6_K in the prefill's planar copy (OLMoEFusedPrefill builds it): per matrix,
// or per expert slab, of P = rows x K/256 blocks, ql [P][128] then qh [P][64],
// scales [P][16], d [P][2] - the same 210 bytes a block, but every field
// 16-byte aligned (d 2-byte), where GGUF's 210-byte blocks allow only 2-byte
// loads and ran 5.6x slower than Q4_K.  Chunk c is quarters 2(c % 2) and
// 2(c % 2) + 1 of half c / 2; part p takes the first or the second (ggml
// order), from six loads.
__device__ __forceinline__ void pf_fetch_q6k(
    const unsigned char* slab, size_t blocks, int row, int nb, int k0, int part,
    uint4 (&ql)[2], uint4 (&qh)[2], uint2& sc, unsigned short& d)
{
    const int sb = k0 >> 8, c = (k0 >> 6) & 3, half = c >> 1, q = 2 * (c & 1) + part;
    const size_t at = (size_t)row * nb + sb;
    const uint4* qlp = reinterpret_cast<const uint4*>(slab + at * 128 + half * 64 + 32 * (q & 1));
    ql[0] = qlp[0];
    ql[1] = qlp[1];
    const uint4* qhp = reinterpret_cast<const uint4*>(slab + blocks * 128 + at * 64 + half * 32);
    qh[0] = qhp[0];
    qh[1] = qhp[1];
    sc = *reinterpret_cast<const uint2*>(slab + blocks * 192 + at * 16 + half * 8);
    d = *reinterpret_cast<const unsigned short*>(slab + blocks * 208 + at * 2);
}

__device__ __forceinline__ void pf_store_q6k(
    const uint4 (&ql)[2], const uint4 (&qh)[2], const uint2& sc, unsigned short d_bits,
    int k0, int part, __half* dst)
{
    const int c = (k0 >> 6) & 3, q = 2 * (c & 1) + part;
    const int nib = (q >> 1) * 4, hs = 2 * q;
    const float d = __half2float(__ushort_as_half(d_bits));
    // scale index (l / 16) + 2q of the half's eight: bytes 2q and 2q + 1
    const unsigned int sw = q < 2 ? sc.x : sc.y;
    const int sh = (q & 1) * 16;
    const float s_lo = d * (float)(signed char)((sw >> sh) & 0xFFu);
    const float s_hi = d * (float)(signed char)((sw >> (sh + 8)) & 0xFFu);
    const unsigned int lw[8] = {ql[0].x, ql[0].y, ql[0].z, ql[0].w, ql[1].x, ql[1].y, ql[1].z, ql[1].w};
    const unsigned int hw[8] = {qh[0].x, qh[0].y, qh[0].z, qh[0].w, qh[1].x, qh[1].y, qh[1].z, qh[1].w};
    __half2* out2 = reinterpret_cast<__half2*>(dst);
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        const unsigned int lo = lw[w] >> nib, hi = hw[w] >> hs;
        const float s = w < 4 ? s_lo : s_hi;  // elements 4w .. 4w + 3
        const int a0 = (int)((lo & 0xF) | ((hi & 3) << 4)) - 32;
        const int a1 = (int)(((lo >> 8) & 0xF) | (((hi >> 8) & 3) << 4)) - 32;
        const int a2 = (int)(((lo >> 16) & 0xF) | (((hi >> 16) & 3) << 4)) - 32;
        const int a3 = (int)(((lo >> 24) & 0xF) | (((hi >> 24) & 3) << 4)) - 32;
        out2[2 * w] = __floats2half2_rn(s * (float)a0, s * (float)a1);
        out2[2 * w + 1] = __floats2half2_rn(s * (float)a2, s * (float)a3);
    }
}

// C[M, N] = A[M, K] . W[N, K]^T, W packed (QT 0: GGUF Q4_K, 1: planar Q6_K)
// and dequantised per tile; mma.sync m16n8k16 with fp32 accumulation.  A
// 64 x 64 output tile, K in steps of 64, four warps of 32 x 32 (two m16 by four
// n8 tiles each).  MODE 0: dense, stored as fp32; blockIdx.z picks one of up
// to three matrices of one format and slot_bits (2 bits per z) its output
// slot.  1: dense, added into out (the residual).  2: expert gate/up ->
// silu(g) * u as fp16 (pair p is token p / top_k), a tile's B rows 0-31 from
// gate and 32-63 from up.  3: expert down, scaled by the routing weight and
// stored per (token, slot).  MODE 2 and 3 take their row tiles from the
// dispatch table.
template <int MODE, int QT>
__global__ void __launch_bounds__(128) pf_gemm(
    const __half* __restrict__ A, const int* __restrict__ pair_index,
    const unsigned long long* __restrict__ table_a, const unsigned long long* __restrict__ table_b,
    int slot_bits, const int* __restrict__ tiles, const int* __restrict__ tile_count,
    int M, int N, int K, float* __restrict__ out_f, __half* __restrict__ out_h,
    const float* __restrict__ route_w, int top_k)
{
    __shared__ __align__(128) unsigned char smem[2 * PF_BM * PF_LDS * 2];
    __half* As = reinterpret_cast<__half*>(smem);
    __half* Bs = As + PF_BM * PF_LDS;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int group = 0, row0, rows;
    if (MODE <= 1) {
        row0 = blockIdx.y * PF_BM;
        rows = min(PF_BM, M - row0);
    } else {
        const int t = blockIdx.y;
        if (t >= *tile_count) return;
        group = tiles[3 * t];
        row0 = tiles[3 * t + 1];
        rows = tiles[3 * t + 2];
    }
    const int z = MODE <= 1 ? (int)blockIdx.z : 0;
    const int n0 = blockIdx.x * (MODE == 2 ? PF_BN / 2 : PF_BN);
    const int nb = K >> 8;

    // A: each thread copies 4 x 8 halves per K step, rows tid/8 + 16j
    const int a_col = (tid & 7) * 8;
    const __half* a_rows[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int r = (tid >> 3) + 16 * j;
        a_rows[j] = nullptr;
        if (r < rows) {
            const int src = MODE == 2 ? pair_index[row0 + r] / top_k : row0 + r;
            a_rows[j] = A + (size_t)src * K + a_col;
        }
    }
    // B: thread tid owns half of tile row tid / 2; warp w's 16 rows are
    // consecutive rows of one matrix (MODE 2: warps 0-1 gate, 2-3 up)
    unsigned long long base;
    int n;
    const int part = tid & 1, br = tid >> 1;
    if (MODE <= 1) {
        base = table_a[z];
        n = n0 + br;
    } else if (MODE == 2) {
        base = warp >= 2 ? table_b[group] : table_a[group];
        n = n0 + br - (warp >= 2 ? PF_BN / 2 : 0);
    } else {
        base = table_a[group];
        n = n0 + br;
    }
    const bool live = n < N;
    const unsigned char* slab = reinterpret_cast<const unsigned char*>(base);
    const unsigned char* w_row = slab + (size_t)n * nb * 144;  // Q4_K only
    const size_t blocks = (size_t)N * nb;                     // Q6_K planes
    __half* b_dst = Bs + br * PF_LDS + part * 32;

    float acc[2][4][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    const int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    const int m16_live = rows > wm ? min(2, (rows - wm + 15) >> 4) : 0;  // warp-uniform

    uint4 ra[4];
    uint4 h4, q4a, q4b;          // Q4_K
    uint4 ql[2], qh[2];          // Q6_K
    uint2 sc;
    unsigned short db = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        ra[j] = a_rows[j] != nullptr ? *reinterpret_cast<const uint4*>(a_rows[j]) : make_uint4(0u, 0u, 0u, 0u);
    if (live) {
        if (QT == 0) pf_fetch_q4k(w_row, 0, h4, q4a, q4b);
        else pf_fetch_q6k(slab, blocks, n, nb, 0, part, ql, qh, sc, db);
    }

    for (int k0 = 0; k0 < K; k0 += PF_BK) {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            *reinterpret_cast<uint4*>(As + ((tid >> 3) + 16 * j) * PF_LDS + a_col) = ra[j];
        if (!live) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) reinterpret_cast<uint4*>(b_dst)[i] = make_uint4(0u, 0u, 0u, 0u);
        } else if (QT == 0) {
            pf_store_q4k(h4, q4a, q4b, k0, part, b_dst);
        } else {
            pf_store_q6k(ql, qh, sc, db, k0, part, b_dst);
        }
        __syncthreads();
        const int next = k0 + PF_BK;
        if (next < K) {  // in flight during this step's MMA
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                ra[j] = a_rows[j] != nullptr ? *reinterpret_cast<const uint4*>(a_rows[j] + next)
                                             : make_uint4(0u, 0u, 0u, 0u);
            if (live) {
                if (QT == 0) pf_fetch_q4k(w_row, next, h4, q4a, q4b);
                else pf_fetch_q6k(slab, blocks, n, nb, next, part, ql, qh, sc, db);
            }
        }
        // expert tiles are often short (8 tokens per expert in a 64-token
        // prompt): a warp skips the m16 tiles that lie wholly past the last row
        if (m16_live > 0) {
            #pragma unroll
            for (int kk = 0; kk < PF_BK; kk += 16) {
                // a[i]: rows wm + 16i .. +15 as four 8 x 8 matrices (rows 0-7 / 8-15
                // by k 0-7 / 8-15); b[q]: weight rows wn + 16q .. +15 by k, so that
                // registers 0-1 feed n8 tile 2q and 2-3 feed tile 2q + 1
                unsigned int a[2][4], b[2][4];
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    if (i < m16_live) {
                        pf_ldmatrix_x4(a[i], As + (wm + 16 * i + (lane & 15)) * PF_LDS + kk + (lane >> 4) * 8);
                    }
                    pf_ldmatrix_x4(b[i], Bs + (wn + 16 * i + (lane & 7) + ((lane >> 4) << 3)) * PF_LDS
                                             + kk + ((lane >> 3) & 1) * 8);
                }
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    if (i >= m16_live) continue;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        pf_mma(acc[i][j], a[i], b[j >> 1][(j & 1) * 2], b[j >> 1][(j & 1) * 2 + 1]);
                }
            }
        }
        __syncthreads();
    }

    // accumulator layout: element e of tile (i, j) sits at row g (+8 for e >= 2),
    // column 2t (+1 for odd e), g = lane / 4, t = lane % 4
    float* Cs = reinterpret_cast<float*>(smem);
    const int g = lane >> 2, t4 = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = wm + 16 * i + g, col = wn + 8 * j + 2 * t4;
            Cs[r * PF_LDC + col] = acc[i][j][0];
            Cs[r * PF_LDC + col + 1] = acc[i][j][1];
            Cs[(r + 8) * PF_LDC + col] = acc[i][j][2];
            Cs[(r + 8) * PF_LDC + col + 1] = acc[i][j][3];
        }
    __syncthreads();
    if (MODE == 2) {
        for (int idx = tid; idx < PF_BM * (PF_BN / 2); idx += blockDim.x) {
            const int r = idx / (PF_BN / 2), c = idx % (PF_BN / 2);
            if (r >= rows || n0 + c >= N) continue;
            const float gv = Cs[r * PF_LDC + c], uv = Cs[r * PF_LDC + c + PF_BN / 2];
            out_h[(size_t)(row0 + r) * N + n0 + c] = __float2half_rn(gv / (1.0f + expf(-gv)) * uv);
        }
        return;
    }
    const int slot = (slot_bits >> (2 * z)) & 3;
    for (int idx = tid; idx < PF_BM * PF_BN; idx += blockDim.x) {
        const int r = idx / PF_BN, c = idx % PF_BN;
        if (r >= rows || n0 + c >= N) continue;
        const float v = Cs[r * PF_LDC + c];
        if (MODE == 0) {
            out_f[((size_t)slot * M + row0 + r) * N + n0 + c] = v;
        } else if (MODE == 1) {
            out_f[(size_t)(row0 + r) * N + n0 + c] += v;
        } else {
            const int p = pair_index[row0 + r];
            out_f[(size_t)p * N + n0 + c] = route_w[p] * v;
        }
    }
}

extern "C" __global__ void pf_embed(
    const long long* __restrict__ tokens, const unsigned char* __restrict__ table,
    int row_bytes, int qtype, float* __restrict__ hidden, int n)
{
    const unsigned char* row = table + (size_t)tokens[blockIdx.x] * row_bytes;
    float* h = hidden + (size_t)blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) h[i] = fz_dequant(row, i, qtype);
}

// One block per token: hidden += sum_s moe_out[token, s] when has_moe, then
// normed = rmsnorm(hidden) * weight as fp16 (the GEMM input) and, when
// write_f is set, as fp32 too (the router input).
extern "C" __global__ void pf_rmsnorm(
    float* __restrict__ hidden, const float* __restrict__ moe_out, int has_moe, int top_k,
    const float* __restrict__ weight, __half* __restrict__ out_h, float* __restrict__ out_f,
    int write_f, int n, float eps)
{
    __shared__ float scratch[32];
    const size_t base = (size_t)blockIdx.x * n;
    float* h = hidden + base;
    float squares = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = h[i];
        if (has_moe) {
            const float* m = moe_out + (size_t)blockIdx.x * top_k * n + i;
            for (int s = 0; s < top_k; ++s) v += m[(size_t)s * n];
            h[i] = v;
        }
        squares += v * v;
    }
    const float inv = rsqrtf(fz_block_sum(squares, scratch) / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = h[i] * inv * weight[i];
        out_h[base + i] = __float2half_rn(v);
        if (write_f) out_f[base + i] = v;
    }
}

// One block per token m (position m), one thread per rotation pair: q and k
// normalised over the whole vector and rotated; q into q_out [heads, M, dim]
// as fp16 for SDPA, k and v into the cache at row m.
extern "C" __global__ void pf_qk_rope_kv(
    const float* __restrict__ qkv, const float* __restrict__ q_norm, const float* __restrict__ k_norm,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    __half* __restrict__ q_out, __half* __restrict__ k_cache, __half* __restrict__ v_cache,
    int M, int n, int head_dim, int max_len, float eps)
{
    __shared__ float scratch[32];
    const int m = blockIdx.x, t = threadIdx.x;
    const int half_dim = head_dim >> 1;
    const int head = t / half_dim, i = t - head * half_dim;
    const int a = head * head_dim + i, b = a + half_dim;
    const float* q = qkv + (size_t)m * n;
    const float* k = qkv + ((size_t)M + m) * n;
    const float* v = qkv + ((size_t)2 * M + m) * n;
    const float qa = q[a], qb = q[b], ka = k[a], kb = k[b];
    const float inv_q = rsqrtf(fz_block_sum(qa * qa + qb * qb, scratch) / (float)n + eps);
    const float inv_k = rsqrtf(fz_block_sum(ka * ka + kb * kb, scratch) / (float)n + eps);
    const float c = rope_cos[(size_t)m * half_dim + i], s = rope_sin[(size_t)m * half_dim + i];
    const float nqa = qa * inv_q * q_norm[a], nqb = qb * inv_q * q_norm[b];
    const float nka = ka * inv_k * k_norm[a], nkb = kb * inv_k * k_norm[b];
    const size_t qo = ((size_t)head * M + m) * head_dim;
    q_out[qo + i] = __float2half_rn(nqa * c - nqb * s);
    q_out[qo + i + half_dim] = __float2half_rn(nqb * c + nqa * s);
    const size_t at = ((size_t)head * max_len + m) * head_dim;
    k_cache[at + i] = __float2half_rn(nka * c - nkb * s);
    k_cache[at + i + half_dim] = __float2half_rn(nkb * c + nka * s);
    v_cache[at + i] = __float2half_rn(v[a]);
    v_cache[at + i + half_dim] = __float2half_rn(v[b]);
}

// Dispatch, three launches: count the (token, slot) pairs per expert, plan
// offsets and row tiles, scatter each pair to its expert's run.  The order
// within a run follows the atomics, but every GEMM row is computed alone and
// written back by (token, slot), so results do not depend on it.
extern "C" __global__ void pf_count(const long long* __restrict__ ids, int pairs, int* __restrict__ counts)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < pairs) atomicAdd(counts + ids[p], 1);
}

extern "C" __global__ void pf_plan(
    const int* __restrict__ counts, int experts, int* __restrict__ offsets, int* __restrict__ fill,
    int* __restrict__ tiles, int* __restrict__ tile_count)
{
    __shared__ int cached[1024];
    for (int e = threadIdx.x; e < experts; e += blockDim.x) cached[e] = counts[e];
    __syncthreads();
    if (threadIdx.x != 0) return;
    int offset = 0, t = 0;
    for (int e = 0; e < experts; ++e) {
        const int c = cached[e];
        offsets[e] = offset;
        fill[e] = 0;
        for (int r = 0; r < c; r += PF_BM) {
            tiles[3 * t] = e;
            tiles[3 * t + 1] = offset + r;
            tiles[3 * t + 2] = min(PF_BM, c - r);
            ++t;
        }
        offset += c;
    }
    *tile_count = t;
}

extern "C" __global__ void pf_scatter(
    const long long* __restrict__ ids, int pairs, const int* __restrict__ offsets,
    int* __restrict__ fill, int* __restrict__ pair_index)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) return;
    const int e = (int)ids[p];
    pair_index[offsets[e] + atomicAdd(fill + e, 1)] = p;
}
"""

_MODULE: Any = None
_STREAMS: dict[int, Any] = {}


def _function(name: str) -> Any:
    global _MODULE
    if _MODULE is None:
        _MODULE = cp.RawModule(code=_SOURCE, options=("--std=c++14",),
                               name_expressions=[f"pf_gemm<{mode}, {qt}>" for mode in range(4) for qt in (0, 1)])
    return _MODULE.get_function(name)


def _launch(name: str, grid: tuple, block: tuple, args: tuple, shared: int = 0) -> None:
    kernel = _function(name)
    handle = torch.cuda.current_stream().cuda_stream
    if handle == 0:
        kernel(grid, block, args, shared_mem=shared)
        return
    stream = _STREAMS.get(handle)
    if stream is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            stream = _STREAMS[handle] = cp.cuda.ExternalStream(handle)
    with stream:
        kernel(grid, block, args, shared_mem=shared)


def _i32(value: int) -> np.int32:
    return np.int32(value)


def _planar_q6k(storage: torch.Tensor, slabs: int) -> torch.Tensor:
    """GGUF Q6_K blocks ``[slabs][P][210]`` as planes per slab.

    ql ``[P][128]``, qh ``[P][64]``, scales ``[P][16]``, d ``[P][2]``: the same
    bytes, every field aligned for vector loads (see pf_fetch_q6k).
    """

    blocks = storage.view(slabs, -1, 210)
    planes = [blocks[:, :, a:b].reshape(slabs, -1) for a, b in ((0, 128), (128, 192), (192, 208), (208, 210))]
    return torch.cat(planes, dim=1).reshape(-1).contiguous()


def _format_groups(matrices: list, pointer, device) -> list[tuple]:
    """Matrices sharing one A, grouped by format for pf_gemm<0, QT>.

    Each group is (pointer table, QT, slot bits, count): the kernel is
    specialised per format, so q/k/v with a Q6_K v are two launches.
    ``pointer(matrix)`` gives the weights the kernel reads (Q6_K: planar).
    """

    groups = []
    for qt in sorted({_QTYPES[m.qtype] for m in matrices}):
        slots = [i for i, m in enumerate(matrices) if _QTYPES[m.qtype] == qt]
        table = torch.tensor([pointer(matrices[i]) for i in slots], dtype=torch.int64, device=device)
        groups.append((table, qt, sum(slot << (2 * z) for z, slot in enumerate(slots)), len(slots)))
    return groups


def gemm(a: torch.Tensor, matrices: list) -> torch.Tensor:
    """``[len(matrices), M, N]`` float32 = ``a`` (fp16 ``[M, K]``) times each packed matrix."""

    if not 1 <= len(matrices) <= 3:
        raise ValueError("one to three matrices per launch")
    rows, cols = matrices[0].rows, matrices[0].cols
    if any(m.rows != rows or m.cols != cols for m in matrices):
        raise ValueError("matrices sharing a launch need one shape")
    if a.dtype != torch.float16 or a.dim() != 2 or a.shape[1] != cols:
        raise TypeError("a must be fp16 [M, K]")
    a = a.contiguous()
    m_rows = a.shape[0]
    out = torch.empty((len(matrices), m_rows, rows), dtype=torch.float32, device=a.device)
    dummy = torch.zeros(1, dtype=torch.int32, device=a.device)
    c = _to_cupy
    planar = {id(m): _planar_q6k(m.storage, 1) for m in matrices if m.qtype == "Q6_K"}

    def pointer(m) -> int:
        return planar[id(m)].data_ptr() if id(m) in planar else m.storage.data_ptr()

    groups = _format_groups(matrices, pointer, a.device)
    for table, qt, slot_bits, count in groups:
        _launch(f"pf_gemm<0, {qt}>", ((rows + 63) // 64, (m_rows + _BM - 1) // _BM, count), (128,),
                (c(a), c(dummy), c(table), c(table), _i32(slot_bits), c(dummy), c(dummy),
                 _i32(m_rows), _i32(rows), _i32(cols), c(out), c(dummy), c(dummy), _i32(1)))
    torch.cuda.current_stream().synchronize()  # the tables are temporaries
    return out


class OLMoEFusedPrefill:
    """Batched prefill sharing an ``OLMoEFusedDecoder``'s weights, cache and state.

    ``run(prompt)`` fills the decoder's key/value cache for positions
    0 .. len(prompt) - 1 and leaves it exactly as stepping the prompt through
    the decoder would: the position after the prompt, the first prediction as
    the next input and in ``history``.  Decoding then continues with
    ``decoder.step()``.  The prompt starts a new sequence.
    """

    def __init__(self, decoder, max_tokens: int, *, use_graphs: bool = True) -> None:
        if not fused_supported():
            raise RuntimeError("the fused prefill needs CUDA and CuPy")
        if not 1 <= max_tokens <= decoder.max_len:
            raise ValueError(f"max_tokens must be 1 .. {decoder.max_len}")
        if decoder.experts > 1024:
            raise ValueError("at most 1024 experts")
        self.decoder, self.max_tokens, self.use_graphs = decoder, max_tokens, use_graphs
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        hidden, experts, top_k = decoder.hidden, decoder.experts, decoder.top_k
        inter = max(layer.inter for layer in decoder.model.layers)
        if hidden % 64 or any(layer.inter % 64 for layer in decoder.model.layers):
            raise ValueError("hidden and expert widths must be multiples of 64")
        device = decoder.device

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=device)

        self.tokens = zeros(max_tokens, dtype=torch.int64)
        self.hidden = zeros(max_tokens, hidden)
        self.normed_h = zeros(max_tokens, hidden, dtype=torch.float16)
        self.normed_f = zeros(max_tokens, hidden)
        self.qkv = zeros(3 * max_tokens * hidden)
        self.query = zeros(max_tokens * hidden, dtype=torch.float16)
        self.counts = zeros(experts, dtype=torch.int32)
        self.offsets = zeros(experts, dtype=torch.int32)
        self.fill = zeros(experts, dtype=torch.int32)
        self.max_tiles = (max_tokens * top_k + _BM - 1) // _BM + experts
        self.tiles = zeros(3 * self.max_tiles, dtype=torch.int32)
        self.tile_count = zeros(1, dtype=torch.int32)
        self.pair_index = zeros(max_tokens * top_k, dtype=torch.int32)
        self.act = zeros(max_tokens * top_k * inter, dtype=torch.float16)
        self.moe_out = zeros(max_tokens, top_k, hidden)
        # Q6_K weights are read from a planar copy (_planar_q6k): with GGUF's
        # 210-byte blocks the GEMM ran Q6_K layers 5.6x slower than Q4_K ones.
        # The copy costs VRAM - 0.9 GB for OLMoE - and the decoder keeps GGUF's.
        self._planar: list[torch.Tensor] = []

        def pointer(matrix) -> int:
            if matrix.qtype != "Q6_K":
                return matrix.storage.data_ptr()
            self._planar.append(_planar_q6k(matrix.storage, 1))
            return self._planar[-1].data_ptr()

        def expert_table(role_entry) -> torch.Tensor:
            storage, table, qtype = role_entry
            if qtype != "Q6_K":
                return table
            copy = _planar_q6k(storage, experts)
            self._planar.append(copy)
            stride = copy.numel() // experts
            return torch.tensor([copy.data_ptr() + e * stride for e in range(experts)],
                                dtype=torch.int64, device=device)

        self._tables = []
        for index, layer in enumerate(decoder.model.layers):
            qkv = _format_groups([layer.q, layer.k, layer.v], pointer, device)
            o = torch.tensor([pointer(layer.o)], dtype=torch.int64, device=device)
            gate_qt, up_qt, down_qt = (_QTYPES[layer.experts[role][2]] for role in ("gate", "up", "down"))
            if gate_qt != up_qt:  # one gate/up tile holds rows of both
                raise NotImplementedError(f"layer {index}: gate and up in different formats")
            tables = [expert_table(layer.experts[role]) for role in ("gate", "up", "down")]
            self._tables.append((qkv, o, _QTYPES[layer.o.qtype], gate_qt, down_qt, tables))

    def run(self, prompt_ids: list[int]) -> int:
        """Prefill the prompt; returns the first predicted token.

        The first prompt of a given length is captured as a CUDA graph, after a
        warm-up on a side stream; later prompts of that length replay it.
        Without graphs a 512-token prompt spent more time launching about 240
        kernels from Python than the GPU spent running them.
        """

        d = self.decoder
        count = len(prompt_ids)
        if not 1 <= count <= self.max_tokens:
            raise ValueError(f"prompt length must be 1 .. {self.max_tokens}")
        if d.graph is None:
            raise RuntimeError("capture the decoder before prefilling: capture() resets it")
        self.tokens[:count].copy_(torch.tensor(prompt_ids, dtype=torch.int64))
        if not self.use_graphs:
            self._enqueue(count)
        else:
            graph = self._graphs.get(count)
            if graph is None:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    self._enqueue(count)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._enqueue(count)
                self._graphs[count] = graph
            graph.replay()
        d.used = count
        return int(d.token)

    def _enqueue(self, count: int) -> None:
        """Every launch of one prefill, from the embedding to the decoder's tail."""

        d = self.decoder
        c, n, k = _to_cupy, d.hidden, d.top_k
        heads, hd = d.num_heads, d.head_dim
        eps = np.float32(d.eps)
        pairs = count * k
        m_tiles = (count + _BM - 1) // _BM
        expert_tiles = (pairs + _BM - 1) // _BM + d.experts
        embed = d.model.embed
        _launch("pf_embed", (count,), (256,),
                (c(self.tokens), c(embed.storage), _i32(embed.row_bytes), _i32(_QTYPES[embed.qtype]),
                 c(self.hidden), _i32(n)))
        query = self.query[:count * n].view(1, heads, count, hd)
        qkv = self.qkv[:3 * count * n]
        act = self.act[:pairs * max(layer.inter for layer in d.model.layers)]
        for index, layer in enumerate(d.model.layers):
            qkv_groups, o_table, o_qt, gate_up_qt, down_qt, expert_tables = self._tables[index]
            keys, values = d.keys[index], d.values[index]
            _launch("pf_rmsnorm", (count,), (256,),
                    (c(self.hidden), c(self.moe_out), _i32(int(index > 0)), _i32(k),
                     c(layer.attn_norm), c(self.normed_h), c(self.normed_f), _i32(0), _i32(n), eps))
            for table, qt, slot_bits, matrices in qkv_groups:
                _launch(f"pf_gemm<0, {qt}>", (n // 64, m_tiles, matrices), (128,),
                        (c(self.normed_h), c(self.pair_index), c(table), c(table),
                         _i32(slot_bits), c(self.tiles), c(self.tile_count), _i32(count), _i32(n),
                         _i32(n), c(qkv), c(act), c(self.normed_f), _i32(k)))
            _launch("pf_qk_rope_kv", (count,), (n // 2,),
                    (c(qkv), c(layer.q_norm), c(layer.k_norm), c(d.rope_cos), c(d.rope_sin),
                     c(query), c(keys), c(values), _i32(count), _i32(n), _i32(hd), _i32(d.max_len),
                     eps))
            context = torch.nn.functional.scaled_dot_product_attention(
                query, keys[:, :count].unsqueeze(0), values[:, :count].unsqueeze(0), is_causal=True)
            context = context.transpose(1, 2).reshape(count, n).contiguous()
            _launch(f"pf_gemm<1, {o_qt}>", (n // 64, m_tiles, 1), (128,),
                    (c(context), c(self.pair_index), c(o_table), c(o_table), _i32(0),
                     c(self.tiles), c(self.tile_count), _i32(count), _i32(n), _i32(n),
                     c(self.hidden), c(act), c(self.normed_f), _i32(k)))
            _launch("pf_rmsnorm", (count,), (256,),
                    (c(self.hidden), c(self.moe_out), _i32(0), _i32(k), c(layer.ffn_norm),
                     c(self.normed_h), c(self.normed_f), _i32(1), _i32(n), eps))
            # OlmoeTopKRouter: softmax over all experts, top_k, no renormalisation
            probabilities = torch.softmax(self.normed_f[:count] @ layer.router.t(), dim=-1)
            weights, ids = torch.topk(probabilities, k, dim=-1)
            weights, ids = weights.contiguous(), ids.contiguous()
            self.counts.zero_()
            _launch("pf_count", ((pairs + 255) // 256,), (256,),
                    (c(ids), _i32(pairs), c(self.counts)))
            _launch("pf_plan", (1,), (256,),
                    (c(self.counts), _i32(d.experts), c(self.offsets), c(self.fill), c(self.tiles),
                     c(self.tile_count)))
            _launch("pf_scatter", ((pairs + 255) // 256,), (256,),
                    (c(ids), _i32(pairs), c(self.offsets), c(self.fill), c(self.pair_index)))
            gate_table, up_table, down_table = expert_tables
            inter = layer.inter
            _launch(f"pf_gemm<2, {gate_up_qt}>", (inter // 32, expert_tiles), (128,),
                    (c(self.normed_h), c(self.pair_index), c(gate_table), c(up_table),
                     _i32(0), c(self.tiles),
                     c(self.tile_count), _i32(count), _i32(inter), _i32(n), c(self.normed_f),
                     c(act), c(weights), _i32(k)))
            _launch(f"pf_gemm<3, {down_qt}>", (n // 64, expert_tiles), (128,),
                    (c(act), c(self.pair_index), c(down_table), c(down_table),
                     _i32(0), c(self.tiles), c(self.tile_count), _i32(count),
                     _i32(n), _i32(inter), c(self.moe_out), c(act), c(weights), _i32(k)))
        # the last token's state goes through the decoder's own tail: final
        # RMSNorm (folding in the last experts), LM head with argmax, finish
        last = count - 1
        d.hidden_state.copy_(self.hidden[last])
        d.moe_out.copy_(self.moe_out[last])
        d.position.fill_(last)
        d.argmax_key.zero_()
        for name, grid, block, args, shared in d._plan[-3:]:
            _decode_launch(name, grid, block, args, shared)
