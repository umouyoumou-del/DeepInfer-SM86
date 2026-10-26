// Extracted from deepinfer/model_executor/models/qwen2_w4a16.py:_SM86_GQA_DECODE_KERNEL

#include <cuda_fp16.h>

#define QWEN2_GQA_MAX_SEQ 256

extern "C" __global__ void qwen2_gqa_decode_256(
    const __half* __restrict__ query,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const unsigned char* __restrict__ mask,
    __half* __restrict__ output,
    int num_heads,
    int num_kv_heads,
    int max_seq,
    int head_dim,
    float scale)
{
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    if (head >= num_heads || max_seq > QWEN2_GQA_MAX_SEQ) return;

    const int kv_group = num_heads / num_kv_heads;
    const int kv_head = head / kv_group;
    const __half* q = query + (size_t)head * head_dim;
    const __half* k = key_cache + (size_t)kv_head * max_seq * head_dim;
    const __half* v = value_cache + (size_t)kv_head * max_seq * head_dim;

    __shared__ float scores[QWEN2_GQA_MAX_SEQ];
    __shared__ float probs[QWEN2_GQA_MAX_SEQ];
    __shared__ float reduce_buf[QWEN2_GQA_MAX_SEQ];

    const bool valid = (tid < max_seq) && (mask[tid] != 0);
    float dot = 0.0f;
    if (valid) {
        const __half* kt = k + (size_t)tid * head_dim;
        #pragma unroll 4
        for (int d = 0; d < head_dim; ++d) {
            dot += __half2float(q[d]) * __half2float(kt[d]);
        }
    }
    scores[tid] = valid ? dot * scale : -1.0e30f;
    reduce_buf[tid] = scores[tid];
    __syncthreads();

    for (int stride = QWEN2_GQA_MAX_SEQ / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce_buf[0];
    const float probability = valid ? __expf(scores[tid] - max_score) : 0.0f;
    probs[tid] = probability;
    reduce_buf[tid] = probability;
    __syncthreads();

    for (int stride = QWEN2_GQA_MAX_SEQ / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / reduce_buf[0];

    if (tid < head_dim) {
        float acc = 0.0f;
        for (int token = 0; token < max_seq; ++token) {
            if (mask[token] != 0) {
                acc += probs[token] * inv_sum * __half2float(v[(size_t)token * head_dim + tid]);
            }
        }
        output[(size_t)head * head_dim + tid] = __float2half_rn(acc);
    }
}
