#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>

#define DP_MAX_TOKENS 64

#ifdef _WIN32
#define DP_EXPORT __declspec(dllexport)
#else
#define DP_EXPORT __attribute__((visibility("default")))
#endif

static inline float dp_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static inline uint16_t dp_load_u16(const uint8_t* p) {
    uint16_t value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

static inline uint32_t dp_load_u32(const uint8_t* p) {
    uint32_t value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

static inline float dp_dot_avx2(const float* a, const float* b, int n) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i + 8 <= n; i += 8) {
        sum = _mm256_fmadd_ps(
            _mm256_loadu_ps(a + i),
            _mm256_loadu_ps(b + i),
            sum
        );
    }
    float lanes[8];
    _mm256_storeu_ps(lanes, sum);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                   lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (int i = n & ~7; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

static inline float dp_silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static inline void dp_decode_q4_k(const uint8_t* block, float* out) {
    const float d = dp_half_to_float(dp_load_u16(block));
    const float dmin = dp_half_to_float(dp_load_u16(block + 2));
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;
    for (int sub = 0; sub < 8; ++sub) {
        uint32_t scale;
        uint32_t minimum;
        if (sub < 4) {
            scale = scales[sub] & 0x3fu;
            minimum = scales[sub + 4] & 0x3fu;
        } else {
            scale = (scales[sub + 4] & 0x0fu) |
                    ((scales[sub - 4] >> 2) & 0x30u);
            minimum = (scales[sub + 4] >> 4) |
                      ((scales[sub] >> 2) & 0x30u);
        }
        const float scale_f = d * (float)scale;
        const float min_f = dmin * (float)minimum;
        const uint8_t* qrow = qs + (sub >> 1) * 32;
        float* dst = out + sub * 32;
        if ((sub & 1) == 0) {
            for (int j = 0; j < 32; ++j) {
                dst[j] = scale_f * (float)(qrow[j] & 0x0fu) - min_f;
            }
        } else {
            for (int j = 0; j < 32; ++j) {
                dst[j] = scale_f * (float)(qrow[j] >> 4) - min_f;
            }
        }
    }
}

static inline void dp_decode_q5_0(const uint8_t* block, float* out) {
    const float d = dp_half_to_float(dp_load_u16(block));
    const uint32_t high_mask = dp_load_u32(block + 2);
    const uint8_t* qs = block + 6;
    for (int j = 0; j < 32; ++j) {
        const int low = (j < 16) ? (qs[j] & 0x0f) : (qs[j - 16] >> 4);
        const int high = (int)((high_mask >> j) & 1u);
        out[j] = d * (float)((low | (high << 4)) - 16);
    }
}

static inline void dp_decode_q8_0(const uint8_t* block, float* out) {
    const float d = dp_half_to_float(dp_load_u16(block));
    const __m256 scale = _mm256_set1_ps(d);
    for (int j = 0; j < 32; j += 8) {
        const __m128i bytes =
            _mm_loadl_epi64((const __m128i*)(block + 2 + j));
        const __m256i quantized = _mm256_cvtepi8_epi32(bytes);
        const __m256 values = _mm256_cvtepi32_ps(quantized);
        _mm256_storeu_ps(out + j, _mm256_mul_ps(values, scale));
    }
}

extern "C" DP_EXPORT void dp_gate_up_q4k(
    const uint8_t* gate,
    const uint8_t* up,
    const float* x,
    float* out,
    int rows,
    int k_size,
    int tokens)
{
    if (tokens > DP_MAX_TOKENS) return;
    const int blocks = k_size / 256;
#pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; ++row) {
        const uint8_t* gate_row = gate + (size_t)row * blocks * 144;
        const uint8_t* up_row = up + (size_t)row * blocks * 144;
        float gate_acc[DP_MAX_TOKENS];
        float up_acc[DP_MAX_TOKENS];
        for (int token = 0; token < tokens; ++token) {
            gate_acc[token] = 0.0f;
            up_acc[token] = 0.0f;
        }
        alignas(32) float gate_w[256];
        alignas(32) float up_w[256];
        for (int block = 0; block < blocks; ++block) {
            dp_decode_q4_k(gate_row + block * 144, gate_w);
            dp_decode_q4_k(up_row + block * 144, up_w);
            const int offset = block * 256;
            for (int token = 0; token < tokens; ++token) {
                const float* xt = x + (size_t)token * k_size + offset;
                gate_acc[token] += dp_dot_avx2(gate_w, xt, 256);
                up_acc[token] += dp_dot_avx2(up_w, xt, 256);
            }
        }
        for (int token = 0; token < tokens; ++token) {
            out[(size_t)token * rows + row] =
                dp_silu(gate_acc[token]) * up_acc[token];
        }
    }
}

extern "C" DP_EXPORT void dp_down_q4k(
    const uint8_t* down,
    const float* x,
    float* out,
    int rows,
    int k_size,
    int tokens)
{
    if (tokens > DP_MAX_TOKENS) return;
    const int blocks = k_size / 256;
#pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; ++row) {
        const uint8_t* down_row = down + (size_t)row * blocks * 144;
        float acc[DP_MAX_TOKENS];
        for (int token = 0; token < tokens; ++token) acc[token] = 0.0f;
        alignas(32) float weight[256];
        for (int block = 0; block < blocks; ++block) {
            dp_decode_q4_k(down_row + block * 144, weight);
            const int offset = block * 256;
            for (int token = 0; token < tokens; ++token) {
                const float* xt = x + (size_t)token * k_size + offset;
                acc[token] += dp_dot_avx2(weight, xt, 256);
            }
        }
        for (int token = 0; token < tokens; ++token) {
            out[(size_t)token * rows + row] = acc[token];
        }
    }
}

extern "C" DP_EXPORT void dp_down_q5_0(
    const uint8_t* down,
    const float* x,
    float* out,
    int rows,
    int k_size,
    int tokens)
{
    if (tokens > DP_MAX_TOKENS) return;
    const int blocks = k_size / 32;
#pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; ++row) {
        const uint8_t* down_row = down + (size_t)row * blocks * 22;
        float acc[DP_MAX_TOKENS];
        for (int token = 0; token < tokens; ++token) acc[token] = 0.0f;
        alignas(32) float weight[32];
        for (int block = 0; block < blocks; ++block) {
            dp_decode_q5_0(down_row + block * 22, weight);
            const int offset = block * 32;
            for (int token = 0; token < tokens; ++token) {
                const float* xt = x + (size_t)token * k_size + offset;
                acc[token] += dp_dot_avx2(weight, xt, 32);
            }
        }
        for (int token = 0; token < tokens; ++token) {
            out[(size_t)token * rows + row] = acc[token];
        }
    }
}

extern "C" DP_EXPORT void dp_down_q8_0(
    const uint8_t* down,
    const float* x,
    float* out,
    int rows,
    int k_size,
    int tokens)
{
    if (tokens > DP_MAX_TOKENS) return;
    const int blocks = k_size / 32;
#pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; ++row) {
        const uint8_t* down_row = down + (size_t)row * blocks * 34;
        float acc[DP_MAX_TOKENS];
        for (int token = 0; token < tokens; ++token) acc[token] = 0.0f;
        alignas(32) float weight[32];
        for (int block = 0; block < blocks; ++block) {
            dp_decode_q8_0(down_row + block * 34, weight);
            const int offset = block * 32;
            for (int token = 0; token < tokens; ++token) {
                const float* xt = x + (size_t)token * k_size + offset;
                acc[token] += dp_dot_avx2(weight, xt, 32);
            }
        }
        for (int token = 0; token < tokens; ++token) {
            out[(size_t)token * rows + row] = acc[token];
        }
    }
}
