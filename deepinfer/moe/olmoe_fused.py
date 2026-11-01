"""OLMoE decode in seven fused kernels per layer, every weight kept quantised.

OLMOE_PORT_PLAN section 37 put the CUDA-graph decoder next to llama.cpp on the
4080 SUPER: 4.04 vs 1.73 ms/token, all of it GPU time.  Per token the graph
decoder ran 1,153 kernels and read 1.24 GB.  PyTorch splits each RMSNorm into
nine kernels (three of them dtype casts) and RoPE into seven, and the dense
weights were dequantised to fp16, 0.75 GB where their GGUF encoding is 0.25 GB.

Here a layer is

  1. RMSNorm, folding in the previous layer's expert outputs and residual
  2. q, k and v projections in one launch, reading Q4_K/Q6_K rows directly
  3. q/k norm, RoPE, the key/value cache write and attention over the
     positions written so far (not a masked full buffer), one block per head
  4. output projection, added into the residual stream in place
  5. RMSNorm, router logits, softmax and top-k: every block normalises the
     hidden state itself and takes one expert's logit; the last block to
     finish picks the experts
  6. gate/up for every routed expert, one warp per (row, expert)
  7. down for every routed expert, one warp per (row, expert)

and the step ends with the LM head, whose GEMV also takes the argmax through a
64-bit ``atomicMax``, and a one-thread kernel that feeds the token back.  The
residual stream and all activations stay in fp32; only the cache is fp16.

The GEMVs keep packed_gemv's lane mapping.  The super-block count is a template
parameter, so the loop unrolls and every load of a row is issued before the
first one is consumed, which packed_gemv's runtime-length loop could not do.
Each block stages the activation in shared memory first.  Two variants that
looked better on paper measured slower on the 4080 SUPER (OLMOE_PORT_PLAN
section 38): issuing the row's loads before staging, and reading the
activation through L1 with no staging at all.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from deepinfer.moe.gpu_quant import _to_cupy

try:
    import cupy as cp
except Exception:  # noqa: BLE001  # pragma: no cover - CuPy can fail to load with a CUDA error, not only ImportError
    cp = None

_QTYPES = {"Q4_K": 0, "Q6_K": 1}
_BLOCK_BYTES = {"Q4_K": 144, "Q6_K": 210}
_SUPER_BLOCKS = (1, 2, 4, 8)  # rows of 256 .. 2048 elements
_WARPS = 8

# reductions and GGUF block decoding, shared with olmoe_prefill
_HELPERS = r"""
#include <cuda_fp16.h>

#define FZ_NEG_INF __int_as_float(0xff800000)

__device__ __forceinline__ float fz_warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float fz_warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Every thread gets the total; the partials are summed in one fixed order.
__device__ float fz_block_sum(float v, float* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    v = fz_warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = 0.0f;
    for (int i = 0; i < warps; ++i) total += scratch[i];
    __syncthreads();
    return total;
}

__device__ float fz_block_max(float v, float* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    v = fz_warp_max(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float best = scratch[0];
    for (int i = 1; i < warps; ++i) best = fmaxf(best, scratch[i]);
    __syncthreads();
    return best;
}

__device__ __forceinline__ unsigned int fz_byte(
    unsigned int w0, unsigned int w1, unsigned int w2, int i)
{
    const unsigned int w = i < 4 ? w0 : (i < 8 ? w1 : w2);
    return (w >> (8 * (i & 3))) & 0xFFu;
}

// get_scale_min_k4 on the twelve scale bytes held in three words
__device__ __forceinline__ void fz_q4k_scale_min(
    unsigned int w0, unsigned int w1, unsigned int w2, int sub, float* scale, float* minimum)
{
    if (sub < 4) {
        *scale = (float)(fz_byte(w0, w1, w2, sub) & 63u);
        *minimum = (float)(fz_byte(w0, w1, w2, sub + 4) & 63u);
    } else {
        *scale = (float)((fz_byte(w0, w1, w2, sub + 4) & 0xFu) | ((fz_byte(w0, w1, w2, sub - 4) >> 6) << 4));
        *minimum = (float)((fz_byte(w0, w1, w2, sub + 4) >> 4) | ((fz_byte(w0, w1, w2, sub) >> 6) << 4));
    }
}

// One element of a packed Q4_K / Q6_K row (ggml dequantize_row_q4_K / _q6_K).
__device__ float fz_dequant(const unsigned char* row, int i, int qtype)
{
    const int j = i & 255;
    if (qtype == 0) {
        const unsigned char* block = row + (size_t)(i >> 8) * 144;
        const float d = __half2float(*reinterpret_cast<const __half*>(block));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(block + 2));
        const unsigned int* w = reinterpret_cast<const unsigned int*>(block + 4);
        const int pair = j >> 6, high = (j >> 5) & 1, l = j & 31;
        float scale, minimum;
        fz_q4k_scale_min(w[0], w[1], w[2], 2 * pair + high, &scale, &minimum);
        const unsigned int q = block[16 + 32 * pair + l];
        return d * scale * (float)(high ? (q >> 4) : (q & 0xF)) - dmin * minimum;
    }
    const unsigned char* block = row + (size_t)(i >> 8) * 210;
    const int half = j >> 7, quarter = (j >> 5) & 3, l = j & 31;
    const unsigned char* ql = block + half * 64;
    const unsigned char* qh = block + 128 + half * 32;
    const signed char* sc = reinterpret_cast<const signed char*>(block + 192) + half * 8;
    const unsigned int low = (quarter & 1) ? ql[l + 32] : ql[l];
    const int q = (int)(((quarter >> 1) ? (low >> 4) : (low & 0xF)) | (((qh[l] >> (2 * quarter)) & 3) << 4)) - 32;
    const float d = __half2float(*reinterpret_cast<const __half*>(block + 208));
    return d * (float)sc[(l >> 4) + 2 * quarter] * (float)q;
}

"""

_SOURCE = _HELPERS + r"""
// packed_gemv's mapping: lane L reads quant bytes [4L, 4L+4) of each super-block,
// elements 4(L%8).. of sub-block 2(L/8) (low nibbles) and 2(L/8)+1 (high nibbles).
// All NB super-blocks are loaded before any is used.
template <int NB>
__device__ __forceinline__ float fz_q4k_dot(const unsigned char* row, const float* xs, int lane)
{
    const int pair = lane >> 3, pos = (lane & 7) * 4, s0 = 2 * pair, s1 = s0 + 1;
    unsigned int q[NB], w0[NB], w1[NB], w2[NB];
    float2 dm[NB];
    #pragma unroll
    for (int b = 0; b < NB; ++b) {
        const unsigned char* block = row + b * 144;
        const uint4 head = *reinterpret_cast<const uint4*>(block);  // d, dmin, scales
        dm[b] = make_float2(__half2float(__ushort_as_half((unsigned short)(head.x & 0xFFFFu))),
                            __half2float(__ushort_as_half((unsigned short)(head.x >> 16))));
        w0[b] = head.y;
        w1[b] = head.z;
        w2[b] = head.w;
        q[b] = *reinterpret_cast<const unsigned int*>(block + 16 + lane * 4);
    }
    float acc = 0.0f;
    #pragma unroll
    for (int b = 0; b < NB; ++b) {
        float sc0, m0, sc1, m1;
        fz_q4k_scale_min(w0[b], w1[b], w2[b], s0, &sc0, &m0);
        fz_q4k_scale_min(w0[b], w1[b], w2[b], s1, &sc1, &m1);
        const float4 xa = *reinterpret_cast<const float4*>(xs + b * 256 + s0 * 32 + pos);
        const float4 xb = *reinterpret_cast<const float4*>(xs + b * 256 + s1 * 32 + pos);
        const unsigned int v = q[b];
        const float dot0 = (float)(v & 0xF) * xa.x + (float)((v >> 8) & 0xF) * xa.y
                         + (float)((v >> 16) & 0xF) * xa.z + (float)((v >> 24) & 0xF) * xa.w;
        const float dot1 = (float)((v >> 4) & 0xF) * xb.x + (float)((v >> 12) & 0xF) * xb.y
                         + (float)((v >> 20) & 0xF) * xb.z + (float)((v >> 28) & 0xF) * xb.w;
        const float sum0 = xa.x + xa.y + xa.z + xa.w;
        const float sum1 = xb.x + xb.y + xb.z + xb.w;
        acc += dm[b].x * (sc0 * dot0 + sc1 * dot1) - dm[b].y * (m0 * sum0 + m1 * sum1);
    }
    return acc;
}

// packed_gemv's mapping: lane L covers half L/16 of each super-block at
// positions l = 2(L%16), l+1 in all four quarters.  Q6_K blocks are 210 bytes,
// so only 2-byte loads are aligned.
template <int NB>
__device__ __forceinline__ float fz_q6k_dot(const unsigned char* row, const float* xs, int lane)
{
    const int half = lane >> 4, l = (lane & 15) * 2, is = l >> 4;
    unsigned int lo_a[NB], lo_b[NB], hi[NB], s01[NB], s23[NB];
    float d[NB];
    #pragma unroll
    for (int b = 0; b < NB; ++b) {
        const unsigned char* block = row + b * 210;
        const unsigned char* ql = block + half * 64;
        const unsigned short* sc = reinterpret_cast<const unsigned short*>(block + 192 + half * 8);
        lo_a[b] = *reinterpret_cast<const unsigned short*>(ql + l);
        lo_b[b] = *reinterpret_cast<const unsigned short*>(ql + l + 32);
        hi[b] = *reinterpret_cast<const unsigned short*>(block + 128 + half * 32 + l);
        // scales is + 0, 2, 4, 6 sit in the low (is = 0) or high byte of ushorts 0..3
        s01[b] = (unsigned int)sc[0] | ((unsigned int)sc[1] << 16);
        s23[b] = (unsigned int)sc[2] | ((unsigned int)sc[3] << 16);
        d[b] = __half2float(*reinterpret_cast<const __half*>(block + 208));
    }
    float acc = 0.0f;
    #pragma unroll
    for (int b = 0; b < NB; ++b) {
        const float* xh = xs + b * 256 + half * 128 + l;
        float sum = 0.0f;
        #pragma unroll
        for (int quarter = 0; quarter < 4; ++quarter) {
            const unsigned int src = (quarter & 1) ? lo_b[b] : lo_a[b];
            const int shift = (quarter >> 1) * 4;
            const int q0 = (int)(((src >> shift) & 0xF) | (((hi[b] >> (2 * quarter)) & 3) << 4)) - 32;
            const int q1 = (int)(((src >> (8 + shift)) & 0xF)
                               | (((hi[b] >> (8 + 2 * quarter)) & 3) << 4)) - 32;
            const unsigned int word = quarter < 2 ? s01[b] : s23[b];
            const int scale = (int)(signed char)((word >> (16 * (quarter & 1) + 8 * is)) & 0xFF);
            const float2 xv = *reinterpret_cast<const float2*>(xh + 32 * quarter);
            sum += (float)scale * ((float)q0 * xv.x + (float)q1 * xv.y);
        }
        acc += d[b] * sum;
    }
    return acc;
}

template <int NB>
__device__ __forceinline__ float fz_row_dot(const unsigned char* row, const float* xs, int qtype, int lane)
{
    return qtype ? fz_q6k_dot<NB>(row, xs, lane) : fz_q4k_dot<NB>(row, xs, lane);
}

// x (NB*256 floats) into shared memory with float4 copies
template <int NB>
__device__ __forceinline__ void fz_stage(const float* x, float4* shared4)
{
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (int i = threadIdx.x; i < NB * 64; i += blockDim.x) shared4[i] = x4[i];
    __syncthreads();
}

// argmax as a max over keys: orderable float bits, then the inverted row so a
// tie goes to the lower row, as torch.argmax does
__device__ __forceinline__ unsigned long long fz_argmax_key(float value, int row)
{
    unsigned int bits = __float_as_uint(value);
    bits = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    return ((unsigned long long)bits << 32) | (unsigned long long)(0xFFFFFFFFu - (unsigned int)row);
}

// One warp per row; blockIdx.y picks one of up to three matrices sharing x.
// mode 0 stores, 1 adds into out (residual), 2 stores and folds in the argmax.
template <int NB>
__global__ void fz_gemv(
    const float* __restrict__ x, const unsigned long long* __restrict__ table, int qtype_bits,
    float* __restrict__ out, int rows, int mode, unsigned long long* __restrict__ argmax_key)
{
    extern __shared__ float4 shared4[];
    fz_stage<NB>(x, shared4);
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int m = blockIdx.y;
    const int qtype = (qtype_bits >> m) & 1;
    const unsigned char* w = reinterpret_cast<const unsigned char*>(table[m])
                           + (size_t)row * (NB * (qtype ? 210 : 144));
    const float dot = fz_warp_sum(fz_row_dot<NB>(w, reinterpret_cast<const float*>(shared4), qtype, lane));
    if (lane != 0) return;
    float* o = out + (size_t)m * rows;
    if (mode == 1) {
        o[row] += dot;
    } else {
        o[row] = dot;
        if (mode == 2) atomicMax(argmax_key, fz_argmax_key(dot, row));
    }
}

// act[s, r] = silu(gate_e[r] . x) * (up_e[r] . x), e = ids[s]; grid (rows / warps, top_k)
template <int NB>
__global__ void fz_gate_up(
    const float* __restrict__ x, const long long* __restrict__ ids,
    const unsigned long long* __restrict__ gate_table, const unsigned long long* __restrict__ up_table,
    float* __restrict__ act, int inter, int gate_qtype, int up_qtype)
{
    extern __shared__ float4 shared4[];
    fz_stage<NB>(x, shared4);
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= inter) return;
    const int slot = blockIdx.y;
    const long long expert = ids[slot];
    const float* xs = reinterpret_cast<const float*>(shared4);
    const unsigned char* g = reinterpret_cast<const unsigned char*>(gate_table[expert])
                           + (size_t)row * (NB * (gate_qtype ? 210 : 144));
    const unsigned char* u = reinterpret_cast<const unsigned char*>(up_table[expert])
                           + (size_t)row * (NB * (up_qtype ? 210 : 144));
    const float gate = fz_warp_sum(fz_row_dot<NB>(g, xs, gate_qtype, lane));
    const float up = fz_warp_sum(fz_row_dot<NB>(u, xs, up_qtype, lane));
    if (lane == 0) act[(size_t)slot * inter + row] = gate / (1.0f + expf(-gate)) * up;
}

// out[s, r] = w[s] * (down_e[r] . act[s]); one warp per (row, expert), grid (rows / warps, top_k).
// The sum over experts is left to the next RMSNorm, in a fixed order.
template <int NB>
__global__ void fz_down(
    const float* __restrict__ act, const long long* __restrict__ ids,
    const float* __restrict__ route_weights, const unsigned long long* __restrict__ down_table,
    float* __restrict__ out, int rows, int qtype)
{
    extern __shared__ float4 shared4[];
    const int slot = blockIdx.y;
    fz_stage<NB>(act + (size_t)slot * NB * 256, shared4);
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;
    const unsigned char* d = reinterpret_cast<const unsigned char*>(down_table[ids[slot]])
                           + (size_t)row * (NB * (qtype ? 210 : 144));
    const float dot = fz_warp_sum(fz_row_dot<NB>(d, reinterpret_cast<const float*>(shared4), qtype, lane));
    if (lane == 0) out[(size_t)slot * rows + row] = route_weights[slot] * dot;
}

extern "C" __global__ void fz_embed(
    const long long* __restrict__ token, const unsigned char* __restrict__ table,
    int row_bytes, int qtype, float* __restrict__ hidden, int n)
{
    const unsigned char* row = table + (size_t)(*token) * row_bytes;
    for (int i = threadIdx.x; i < n; i += blockDim.x) hidden[i] = fz_dequant(row, i, qtype);
}

// hidden += sum_s moe_out[s] (when has_moe), then normed = rmsnorm(hidden) * weight
extern "C" __global__ void fz_rmsnorm(
    float* __restrict__ hidden, const float* __restrict__ moe_out, int has_moe, int top_k,
    const float* __restrict__ weight, float* __restrict__ normed, int n, float eps)
{
    __shared__ float scratch[32];
    float squares = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float h = hidden[i];
        if (has_moe) {
            for (int s = 0; s < top_k; ++s) h += moe_out[(size_t)s * n + i];
            hidden[i] = h;
        }
        squares += h * h;
    }
    const float inv = rsqrtf(fz_block_sum(squares, scratch) / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) normed[i] = hidden[i] * inv * weight[i];
}

// One block per head, a multiple of head_dim threads.  q_norm and k_norm span
// the whole vector, so every block takes both sums of squares itself (the same
// order in every block, so the same result); then it rotates its own head,
// writes its key and value at the position and attends over 0..position.
// The weighted sum over positions is split between blockDim / head_dim groups.
extern "C" __global__ void fz_attend(
    const float* __restrict__ qkv, const float* __restrict__ q_norm, const float* __restrict__ k_norm,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    const long long* __restrict__ position, __half* __restrict__ k_cache, __half* __restrict__ v_cache,
    float* __restrict__ out, int n, int head_dim, int max_len, float eps, float scale)
{
    extern __shared__ float smem[];
    __shared__ float scratch[32];
    float* qs = smem;
    float* scores = smem + head_dim;
    const int head = blockIdx.x, t = threadIdx.x;
    const float* q = qkv;
    const float* k = qkv + n;
    const float* v = qkv + 2 * (size_t)n;
    float sq = 0.0f, sk = 0.0f;
    for (int i = t; i < n; i += blockDim.x) {
        const float a = q[i], b = k[i];
        sq += a * a;
        sk += b * b;
    }
    const float inv_q = rsqrtf(fz_block_sum(sq, scratch) / (float)n + eps);
    const float inv_k = rsqrtf(fz_block_sum(sk, scratch) / (float)n + eps);
    const long long pos = *position;
    const int half_dim = head_dim >> 1;
    const size_t cache_head = (size_t)head * max_len * head_dim;
    if (t < half_dim) {
        const int a = head * head_dim + t, b = a + half_dim;
        const float c = rope_cos[pos * half_dim + t], s = rope_sin[pos * half_dim + t];
        const float qa = q[a] * inv_q * q_norm[a], qb = q[b] * inv_q * q_norm[b];
        qs[t] = qa * c - qb * s;
        qs[t + half_dim] = qb * c + qa * s;
        const float ka = k[a] * inv_k * k_norm[a], kb = k[b] * inv_k * k_norm[b];
        const size_t at = cache_head + (size_t)pos * head_dim;
        k_cache[at + t] = __float2half_rn(ka * c - kb * s);
        k_cache[at + t + half_dim] = __float2half_rn(kb * c + ka * s);
        v_cache[at + t] = __float2half_rn(v[a]);
        v_cache[at + t + half_dim] = __float2half_rn(v[b]);
    }
    __syncthreads();  // this block's cache row is visible to it from here on
    const int len = (int)pos + 1;
    const int lane = t & 31, warp = t >> 5, warps = blockDim.x >> 5;
    const __half* keys = k_cache + cache_head;
    for (int j = warp; j < len; j += warps) {
        const __half* key = keys + (size_t)j * head_dim;
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += 32) partial += qs[d] * __half2float(key[d]);
        partial = fz_warp_sum(partial);
        if (lane == 0) scores[j] = partial * scale;
    }
    __syncthreads();
    float best = FZ_NEG_INF;
    for (int j = t; j < len; j += blockDim.x) best = fmaxf(best, scores[j]);
    best = fz_block_max(best, scratch);
    float total = 0.0f;
    for (int j = t; j < len; j += blockDim.x) {
        const float e = expf(scores[j] - best);
        scores[j] = e;
        total += e;
    }
    total = fz_block_sum(total, scratch);
    const int groups = blockDim.x / head_dim, d = t % head_dim, group = t / head_dim;
    float* partial = scores + max_len;
    const __half* values = v_cache + cache_head + d;
    float acc = 0.0f;
    for (int j = group; j < len; j += groups) acc += scores[j] * __half2float(values[(size_t)j * head_dim]);
    partial[group * head_dim + d] = acc;
    __syncthreads();
    if (t < head_dim) {
        float sum = 0.0f;
        for (int g = 0; g < groups; ++g) sum += partial[g * head_dim + t];
        out[head * head_dim + t] = sum / total;
    }
}

// RMSNorm, router and top-k in one launch, one block per expert.  Every block
// normalises the hidden state itself (8 KB from L2), writes its slice of the
// normed vector for the expert kernels and computes its expert's logit.  The
// last block to finish takes the softmax over all experts and the top_k
// largest probabilities, highest first, un-renormalised (OlmoeTopKRouter).
extern "C" __global__ void fz_route(
    const float* __restrict__ hidden, const float* __restrict__ norm_weight,
    const float* __restrict__ router, float* __restrict__ normed, float* __restrict__ logits,
    unsigned int* __restrict__ counter, int n, float eps, int top_k,
    long long* __restrict__ ids, float* __restrict__ weights)
{
    __shared__ float scratch[32];
    __shared__ int last;
    const int e = blockIdx.x, experts = gridDim.x;
    const float4* h4 = reinterpret_cast<const float4*>(hidden);
    const float4* w4 = reinterpret_cast<const float4*>(norm_weight);
    const float4* r4 = reinterpret_cast<const float4*>(router + (size_t)e * n);
    float squares = 0.0f, dot = 0.0f;
    for (int i = threadIdx.x; i < (n >> 2); i += blockDim.x) {
        const float4 h = h4[i], w = w4[i], r = r4[i];
        squares += h.x * h.x + h.y * h.y + h.z * h.z + h.w * h.w;
        dot += r.x * (h.x * w.x) + r.y * (h.y * w.y) + r.z * (h.z * w.z) + r.w * (h.w * w.w);
    }
    const float inv = rsqrtf(fz_block_sum(squares, scratch) / (float)n + eps);
    dot = fz_block_sum(dot, scratch) * inv;
    const int chunk = (n + experts - 1) / experts;
    const int end = min(n, (e + 1) * chunk);
    for (int i = e * chunk + threadIdx.x; i < end; i += blockDim.x) {
        normed[i] = hidden[i] * inv * norm_weight[i];
    }
    if (threadIdx.x == 0) {
        logits[e] = dot;
        __threadfence();
        last = atomicAdd(counter, 1u) == (unsigned int)(experts - 1);
    }
    __syncthreads();
    if (!last || threadIdx.x >= 32) return;
    const int lane = threadIdx.x;
    const float v0 = lane < experts ? __ldcg(logits + lane) : FZ_NEG_INF;
    const float v1 = lane + 32 < experts ? __ldcg(logits + lane + 32) : FZ_NEG_INF;
    const float top = fz_warp_max(fmaxf(v0, v1));
    const float e0 = lane < experts ? expf(v0 - top) : 0.0f;
    const float e1 = lane + 32 < experts ? expf(v1 - top) : 0.0f;
    const float total = fz_warp_sum(e0 + e1);
    float p0 = lane < experts ? e0 / total : -1.0f;
    float p1 = lane + 32 < experts ? e1 / total : -1.0f;
    for (int k = 0; k < top_k; ++k) {
        float value = p0;
        int index = lane;
        if (p1 > value) {
            value = p1;
            index = lane + 32;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, value, o);
            const int other_index = __shfl_xor_sync(0xffffffffu, index, o);
            if (other > value || (other == value && other_index < index)) {
                value = other;
                index = other_index;
            }
        }
        if (lane == 0) {
            ids[k] = index;
            weights[k] = value;
        }
        if (index == lane) p0 = -1.0f;
        if (index == lane + 32) p1 = -1.0f;
    }
    if (lane == 0) *counter = 0u;  // ready for the next layer
}

// The argmax becomes the next input; history[p] is the token predicted after position p.
extern "C" __global__ void fz_finish(
    unsigned long long* __restrict__ argmax_key, long long* __restrict__ token,
    long long* __restrict__ history, long long* __restrict__ position)
{
    const long long pos = *position;
    const long long predicted = (long long)(0xFFFFFFFFu - (unsigned int)(*argmax_key & 0xFFFFFFFFull));
    token[0] = predicted;
    history[pos] = predicted;
    position[0] = pos + 1;
    *argmax_key = 0ull;
}
"""

_TEMPLATES = ("fz_gemv", "fz_gate_up", "fz_down")
_MODULE: Any = None
_STREAMS: dict[int, Any] = {}


def _function(name: str) -> Any:
    global _MODULE
    if _MODULE is None:
        expressions = [f"{t}<{nb}>" for t in _TEMPLATES for nb in _SUPER_BLOCKS]
        _MODULE = cp.RawModule(code=_SOURCE, options=("--std=c++14",),
                               name_expressions=expressions)
    return _MODULE.get_function(name)


def _launch(name: str, grid: tuple, block: tuple, args: tuple, shared: int = 0) -> None:
    """Launch on the caller's torch stream, so eager use and graph capture both work."""

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


def fused_supported() -> bool:
    return cp is not None and torch.cuda.is_available()


@dataclass(frozen=True)
class PackedMatrix:
    """A weight matrix as GGUF stores it: ``rows`` rows of packed super-blocks."""

    storage: torch.Tensor  # uint8, rows * row_bytes, on the device
    qtype: str
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if self.qtype not in _QTYPES:
            raise ValueError(f"unsupported quantisation {self.qtype}")
        if self.cols % 256 or self.cols // 256 not in _SUPER_BLOCKS:
            raise ValueError(f"row length {self.cols} must be 256 x one of {_SUPER_BLOCKS}")
        if self.storage.dtype != torch.uint8 or self.storage.numel() != self.rows * self.row_bytes:
            raise ValueError(f"expected {self.rows * self.row_bytes} packed bytes")

    @property
    def row_bytes(self) -> int:
        return self.cols // 256 * _BLOCK_BYTES[self.qtype]

    @property
    def nbytes(self) -> int:
        return self.rows * self.row_bytes


@dataclass
class FusedLayer:
    q: PackedMatrix
    k: PackedMatrix
    v: PackedMatrix
    o: PackedMatrix
    attn_norm: torch.Tensor  # float32 [hidden]
    ffn_norm: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    router: torch.Tensor  # float32 [experts, hidden]
    experts: dict  # role -> (storage, pointer table, qtype), as load_packed_experts returns
    inter: int


@dataclass
class FusedModel:
    embed: PackedMatrix
    lm_head: PackedMatrix
    final_norm: torch.Tensor
    layers: list


def packed_matrix(index, name: str, device: str = "cuda") -> PackedMatrix:
    info = index.tensor(name)
    raw = np.array(index.read_raw(name), dtype=np.uint8, copy=True)
    return PackedMatrix(torch.from_numpy(raw).to(device), info.qtype.name,
                        rows=int(info.dims[1]), cols=int(info.dims[0]))


def load_fused_model(index, spec, layers: int | None = None, device: str = "cuda") -> FusedModel:
    """Every weight as stored in the GGUF: matrices packed, norms and router F32."""

    from deepinfer.moe.olmoe_packed import load_packed_experts
    from deepinfer.moe.olmoe_weights import layer_tensors

    def f32(name: str) -> torch.Tensor:
        array = np.asarray(index.read_tensor(name), dtype=np.float32)
        return torch.from_numpy(array.copy()).to(device)

    count = spec.block_count if layers is None else layers
    stack = []
    for layer_index in range(count):
        names = layer_tensors(spec, layer_index)
        inter = int(index.tensor(names.expert_gate).numpy_shape[1])
        experts = {
            role: load_packed_experts(index, name, experts=spec.expert_count, out_features=rows)
            for role, name, rows in (("gate", names.expert_gate, inter),
                                     ("up", names.expert_up, inter),
                                     ("down", names.expert_down, spec.hidden_size))
        }
        attention = {role: packed_matrix(index, name, device) for role, name in names.attention.items()}
        stack.append(FusedLayer(
            q=attention["q_proj"], k=attention["k_proj"], v=attention["v_proj"],
            o=attention["o_proj"], attn_norm=f32(names.norms["attn_norm"]),
            ffn_norm=f32(names.norms["ffn_norm"]), q_norm=f32(names.norms["attn_q_norm"]),
            k_norm=f32(names.norms["attn_k_norm"]), router=f32(names.router),
            experts=experts, inter=inter,
        ))
    return FusedModel(embed=packed_matrix(index, "token_embd.weight", device),
                      lm_head=packed_matrix(index, "output.weight", device),
                      final_norm=f32("output_norm.weight"), layers=stack)


def _f32(value: float) -> np.float32:
    return np.float32(value)


def _i32(value: int) -> np.int32:
    return np.int32(value)


def gemv(x: torch.Tensor, matrices: list[PackedMatrix]) -> torch.Tensor:
    """``[len(matrices), rows]`` = each matrix times ``x``; up to three sharing one launch."""

    if not 1 <= len(matrices) <= 3:
        raise ValueError("one to three matrices per launch")
    rows, cols = matrices[0].rows, matrices[0].cols
    if any(m.rows != rows or m.cols != cols for m in matrices):
        raise ValueError("matrices sharing a launch need one shape")
    if x.dtype != torch.float32 or x.numel() != cols:
        raise TypeError("x must be float32 with one element per column")
    table = torch.tensor([m.storage.data_ptr() for m in matrices], dtype=torch.int64,
                         device=x.device)
    bits = sum(_QTYPES[m.qtype] << i for i, m in enumerate(matrices))
    out = torch.empty((len(matrices), rows), dtype=torch.float32, device=x.device)
    key = torch.zeros(1, dtype=torch.int64, device=x.device)
    _launch(f"fz_gemv<{cols // 256}>", ((rows + _WARPS - 1) // _WARPS, len(matrices)),
            (32 * _WARPS,), (_to_cupy(x.contiguous()), _to_cupy(table), _i32(bits),
                             _to_cupy(out), _i32(rows), _i32(0), _to_cupy(key)),
            shared=cols * 4)
    torch.cuda.current_stream().synchronize()  # table and key are temporaries
    return out


class OLMoEFusedDecoder:
    """Greedy single-token decode over packed weights, replayed from one CUDA graph.

    Same interface as ``OLMoEGraphDecoder``: ``capture()``, then ``step()`` or
    ``generate()``; ``logits`` is the last step's ``[1, 1, vocab]`` float32 logits.
    """

    def __init__(self, model: FusedModel, *, num_heads: int, eps: float, rope_base: float,
                 top_k: int, max_len: int) -> None:
        if not fused_supported():
            raise RuntimeError("the fused decoder needs CUDA and CuPy")
        if not model.layers:
            raise ValueError("need at least one layer")
        hidden = model.embed.cols
        head_dim = hidden // num_heads
        experts = model.layers[0].router.shape[0]
        if hidden % num_heads or head_dim % 64 or head_dim > 1024 or hidden // 2 > 1024:
            raise ValueError(f"unsupported hidden {hidden} / heads {num_heads}")
        if not 1 <= top_k <= experts <= 64 or hidden % 4:
            raise ValueError("top_k <= experts <= 64 is required")
        for index, layer in enumerate(model.layers):
            if any((m.rows, m.cols) != (hidden, hidden) for m in (layer.q, layer.k, layer.v, layer.o)):
                raise ValueError(f"layer {index}: attention matrices must be {hidden} x {hidden}")
            if tuple(layer.router.shape) != (experts, hidden):
                raise ValueError(f"layer {index}: router must be {experts} x {hidden}")
        if model.lm_head.cols != hidden:
            raise ValueError("the LM head must read the hidden state")
        self.model, self.num_heads, self.eps, self.top_k = model, num_heads, eps, top_k
        self.hidden, self.head_dim, self.experts, self.max_len = hidden, head_dim, experts, max_len
        device = model.embed.storage.device
        self.device = device

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=device)

        inter = max(layer.inter for layer in model.layers)
        vocab = model.lm_head.rows
        self.hidden_state, self.normed = zeros(hidden), zeros(hidden)
        self.qkv, self.context = zeros(3, hidden), zeros(hidden)
        self.router_logits, self.route_weights = zeros(experts), zeros(top_k)
        self.expert_ids = zeros(top_k, dtype=torch.int64)
        self.route_counter = zeros(1, dtype=torch.int32)
        self.act, self.moe_out = zeros(top_k, inter), zeros(top_k, hidden)
        self._logits = zeros(vocab)
        self.logits = self._logits.view(1, 1, vocab)
        self.argmax_key = zeros(1, dtype=torch.int64)
        self.token = zeros(1, 1, dtype=torch.int64)
        self.position = zeros(1, dtype=torch.int64)
        self.history = zeros(max_len, dtype=torch.int64)
        cache_shape = (len(model.layers), num_heads, max_len, head_dim)
        self.keys = zeros(*cache_shape, dtype=torch.float16)
        self.values = zeros(*cache_shape, dtype=torch.float16)

        # RoPE angles exactly as build_rope computes them, kept in float32
        half = head_dim // 2
        inverse = torch.pow(float(rope_base), -(torch.arange(half, dtype=torch.float32) * (2.0 / head_dim)))
        angles = torch.arange(max_len, dtype=torch.float32).unsqueeze(1) * inverse.unsqueeze(0)
        self.rope_cos = torch.cos(angles).to(device)
        self.rope_sin = torch.sin(angles).to(device)

        self._tables: list[torch.Tensor] = []  # keep pointer tables alive
        self._plan = self._build_plan()
        self.used = 0
        self.graph: torch.cuda.CUDAGraph | None = None

    def _table(self, matrices: list[PackedMatrix]) -> tuple[Any, int]:
        table = torch.tensor([m.storage.data_ptr() for m in matrices], dtype=torch.int64,
                             device=self.device)
        self._tables.append(table)
        return _to_cupy(table), sum(_QTYPES[m.qtype] << i for i, m in enumerate(matrices))

    def _build_plan(self) -> list[tuple]:
        """The step as a fixed list of launches over fixed buffers."""

        c = _to_cupy
        n, hd, k = self.hidden, self.head_dim, self.top_k
        nb = n // 256
        warp_grid = (n + _WARPS - 1) // _WARPS
        # four groups share an attention head's weighted sum when they fit in a block
        attend_threads = hd * max(1, min(4, 1024 // hd))
        threads = (32 * _WARPS,)
        eps = _f32(self.eps)
        plan: list[tuple] = []
        embed = self.model.embed
        plan.append(("fz_embed", (1,), (256,),
                     (c(self.token), c(embed.storage), _i32(embed.row_bytes),
                      _i32(_QTYPES[embed.qtype]), c(self.hidden_state), _i32(n)), 0))
        for index, layer in enumerate(self.model.layers):
            qkv_table, qkv_bits = self._table([layer.q, layer.k, layer.v])
            o_table, o_bits = self._table([layer.o])
            keys, values = self.keys[index], self.values[index]
            _, gate_table, gate_qtype = layer.experts["gate"]
            _, up_table, up_qtype = layer.experts["up"]
            _, down_table, down_qtype = layer.experts["down"]
            inter = layer.inter
            for qtype in (gate_qtype, up_qtype, down_qtype):
                if qtype not in _QTYPES:
                    raise ValueError(f"layer {index}: unsupported expert quantisation {qtype}")
            if inter % 256 or inter // 256 not in _SUPER_BLOCKS:
                raise ValueError(f"layer {index}: expert width {inter} is not supported")
            plan += [
                ("fz_rmsnorm", (1,), (1024,),
                 (c(self.hidden_state), c(self.moe_out), _i32(int(index > 0)), _i32(k),
                  c(layer.attn_norm), c(self.normed), _i32(n), eps), 0),
                (f"fz_gemv<{nb}>", (warp_grid, 3), threads,
                 (c(self.normed), qkv_table, _i32(qkv_bits), c(self.qkv), _i32(n), _i32(0),
                  c(self.argmax_key)), n * 4),
                ("fz_attend", (self.num_heads,), (attend_threads,),
                 (c(self.qkv), c(layer.q_norm), c(layer.k_norm), c(self.rope_cos),
                  c(self.rope_sin), c(self.position), c(keys), c(values), c(self.context),
                  _i32(n), _i32(hd), _i32(self.max_len), eps, _f32(1.0 / math.sqrt(hd))),
                 (hd + self.max_len + attend_threads) * 4),
                (f"fz_gemv<{nb}>", (warp_grid, 1), threads,
                 (c(self.context), o_table, _i32(o_bits), c(self.hidden_state), _i32(n),
                  _i32(1), c(self.argmax_key)), n * 4),
                ("fz_route", (self.experts,), (128,),
                 (c(self.hidden_state), c(layer.ffn_norm), c(layer.router), c(self.normed),
                  c(self.router_logits), c(self.route_counter), _i32(n), eps, _i32(k),
                  c(self.expert_ids), c(self.route_weights)), 0),
                (f"fz_gate_up<{nb}>", ((inter + _WARPS - 1) // _WARPS, k), threads,
                 (c(self.normed), c(self.expert_ids), c(gate_table), c(up_table), c(self.act),
                  _i32(inter), _i32(_QTYPES[gate_qtype]), _i32(_QTYPES[up_qtype])), n * 4),
                (f"fz_down<{inter // 256}>", (warp_grid, k), threads,
                 (c(self.act), c(self.expert_ids), c(self.route_weights), c(down_table),
                  c(self.moe_out), _i32(n), _i32(_QTYPES[down_qtype])), inter * 4),
            ]
        head = self.model.lm_head
        head_table, head_bits = self._table([head])
        plan += [
            ("fz_rmsnorm", (1,), (1024,),
             (c(self.hidden_state), c(self.moe_out), _i32(1), _i32(k), c(self.model.final_norm),
              c(self.normed), _i32(n), eps), 0),
            (f"fz_gemv<{nb}>", ((head.rows + _WARPS - 1) // _WARPS, 1), threads,
             (c(self.normed), head_table, _i32(head_bits), c(self._logits), _i32(head.rows),
              _i32(2), c(self.argmax_key)), n * 4),
            ("fz_finish", (1,), (1,),
             (c(self.argmax_key), c(self.token), c(self.history), c(self.position)), 0),
        ]
        return plan

    @property
    def kernels_per_step(self) -> int:
        return len(self._plan)

    def bytes_per_token(self) -> dict[str, float]:
        """Weight bytes one step reads: dense matrices whole, top_k experts per layer."""

        dense = self.model.lm_head.nbytes + self.model.embed.row_bytes
        expert = 0
        for layer in self.model.layers:
            dense += sum(m.nbytes for m in (layer.q, layer.k, layer.v, layer.o))
            dense += 4 * (layer.router.numel() + 4 * self.hidden)
            for storage, _pointers, _qtype in layer.experts.values():
                expert += self.top_k * storage.numel() // self.experts
        return {"dense_gb": dense / 1e9, "expert_gb": expert / 1e9,
                "total_gb": (dense + expert) / 1e9}

    def _step(self) -> None:
        for name, grid, block, args, shared in self._plan:
            _launch(name, grid, block, args, shared)

    def capture(self, warmup: int = 3) -> None:
        """Compile and warm up on a side stream, then capture one step."""

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self._step()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        self.reset()

    def reset(self) -> None:
        """Start a new sequence; cache rows past the position are never read."""

        self.position.zero_()
        self.argmax_key.zero_()
        self.route_counter.zero_()
        self.used = 0

    def step(self, token: torch.Tensor | int | None = None) -> torch.Tensor:
        """Read one token (default: the previous prediction) and predict the next.

        Returns the device tensor holding the prediction; the next step
        overwrites it, so clone it to keep it.
        """

        if self.graph is None:
            raise RuntimeError("call capture() first")
        if self.used >= self.max_len:
            raise RuntimeError(f"sequence exceeds max_len={self.max_len}")
        if isinstance(token, int):
            self.token.fill_(token)
        elif token is not None and token.data_ptr() != self.token.data_ptr():
            self.token.copy_(token.view(1, 1))
        self.graph.replay()
        self.used += 1
        return self.token

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        """Greedy: the prompt one token at a time, then the graph feeds itself."""

        if not prompt_ids or max_new_tokens < 1:
            raise ValueError("need a prompt and at least one new token")
        self.reset()
        for token in prompt_ids:
            self.step(int(token))
        for _ in range(max_new_tokens - 1):
            self.step()
        start = len(prompt_ids) - 1
        return self.history[start:start + max_new_tokens].tolist()
