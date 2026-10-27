// Extracted from deepinfer/engine/paged_attention.py:_SM86_PAGED_GQA_I8_KERNEL

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define DP_PAGED_MAX_THREADS 1024

__device__ __forceinline__ float dp_warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

extern "C" __global__ void dp_paged_gqa_decode_i8(
    const __half* __restrict__ query,
    const signed char* __restrict__ key_cache,
    const signed char* __restrict__ value_cache,
    const __half* __restrict__ key_scales,
    const __half* __restrict__ value_scales,
    const long long* __restrict__ block_table,
    const long long* __restrict__ valid_lengths,
    __half* __restrict__ output,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks,
    int max_length,
    int window_size,
    float scale,
    float softcap,
    int use_softcap)
{
    const int head = blockIdx.x;
    const int batch = blockIdx.y;
    const int tid = threadIdx.x;
    const int kv_group = num_heads / num_kv_heads;
    const int kv_head = head / kv_group;
    const int valid_len = (int)valid_lengths[batch];
    const int start = window_size > 0 ? max(0, valid_len - window_size) : 0;

    __shared__ float scores[DP_PAGED_MAX_THREADS];
    __shared__ float probs[DP_PAGED_MAX_THREADS];
    __shared__ float reduce_buf[DP_PAGED_MAX_THREADS];

    float score = -1.0e30f;
    if (tid < valid_len && tid >= start && tid < max_length) {
        const int block_index = tid / block_size;
        const int offset = tid % block_size;
        const long long block_id = block_table[batch * max_blocks + block_index];
        const size_t kv_scale_offset = (
            (size_t)block_id * num_kv_heads + kv_head
        ) * block_size + offset;
        const size_t kv_offset = (
            ((size_t)block_id * num_kv_heads + kv_head) * block_size + offset
        ) * head_dim;
        const signed char* kptr = key_cache + kv_offset;
        const float k_scale = __half2float(key_scales[kv_scale_offset]);
        float dot = 0.0f;
#pragma unroll 4
        for (int d = 0; d < head_dim; ++d) {
            dot += __half2float(query[
                ((size_t)batch * num_heads + head) * head_dim + d
            ]) * ((float)kptr[d] * k_scale);
        }
        score = dot * scale;
        if (use_softcap) {
            score = tanhf(score / softcap) * softcap;
        }
    }
    scores[tid] = score;
    reduce_buf[tid] = score;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce_buf[0];
    const bool valid = (tid < valid_len && tid >= start && tid < max_length);
    const float probability = valid ? __expf(scores[tid] - max_score) : 0.0f;
    probs[tid] = probability;
    reduce_buf[tid] = probability;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / reduce_buf[0];

    if (tid < head_dim) {
        float acc = 0.0f;
        for (int token = start; token < valid_len; ++token) {
            if (token >= max_length) {
                break;
            }
            const int block_index = token / block_size;
            const int offset = token % block_size;
            const long long block_id = block_table[
                batch * max_blocks + block_index
            ];
            const size_t v_scale_offset = (
                (size_t)block_id * num_kv_heads + kv_head
            ) * block_size + offset;
            const size_t v_offset = (
                ((size_t)block_id * num_kv_heads + kv_head) * block_size + offset
            ) * head_dim + tid;
            acc += probs[token] * inv_sum
                * ((float)value_cache[v_offset]
                   * __half2float(value_scales[v_scale_offset]));
        }
        output[
            ((size_t)batch * num_heads + head) * head_dim + tid
        ] = __float2half_rn(acc);
    }
}
