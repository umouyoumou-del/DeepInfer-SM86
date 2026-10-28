#include <immintrin.h>
#include <cstdint>
#include <cstring>

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

extern "C" DP_EXPORT void dp_dequant_q4_k(
    const uint8_t* packed,
    float* out,
    int rows,
    int blocks)
{
    for (int row = 0; row < rows; ++row) {
        for (int block_index = 0; block_index < blocks; ++block_index) {
            const uint8_t* block =
                packed + ((size_t)row * blocks + block_index) * 144;
            const float d = dp_half_to_float(dp_load_u16(block));
            const float dmin = dp_half_to_float(dp_load_u16(block + 2));
            const uint8_t* scales = block + 4;
            const uint8_t* qs = block + 16;
            float* dst =
                out + ((size_t)row * blocks + block_index) * 256;
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
                float* sub_dst = dst + sub * 32;
                if ((sub & 1) == 0) {
                    for (int j = 0; j < 32; ++j) {
                        sub_dst[j] = scale_f * (float)(qrow[j] & 0x0fu) - min_f;
                    }
                } else {
                    for (int j = 0; j < 32; ++j) {
                        sub_dst[j] = scale_f * (float)(qrow[j] >> 4) - min_f;
                    }
                }
            }
        }
    }
}

extern "C" DP_EXPORT void dp_dequant_q5_0(
    const uint8_t* packed,
    float* out,
    int rows,
    int blocks)
{
    for (int row = 0; row < rows; ++row) {
        for (int block_index = 0; block_index < blocks; ++block_index) {
            const uint8_t* block =
                packed + ((size_t)row * blocks + block_index) * 22;
            const float d = dp_half_to_float(dp_load_u16(block));
            const uint32_t high_mask = dp_load_u32(block + 2);
            const uint8_t* qs = block + 6;
            float* dst =
                out + ((size_t)row * blocks + block_index) * 32;
            for (int j = 0; j < 32; ++j) {
                const int low = (j < 16) ? (qs[j] & 0x0f) : (qs[j - 16] >> 4);
                const int high = (int)((high_mask >> j) & 1u);
                dst[j] = d * (float)((low | (high << 4)) - 16);
            }
        }
    }
}

extern "C" DP_EXPORT void dp_dequant_q8_0(
    const uint8_t* packed,
    float* out,
    int rows,
    int blocks)
{
    for (int row = 0; row < rows; ++row) {
        for (int block_index = 0; block_index < blocks; ++block_index) {
            const uint8_t* block =
                packed + ((size_t)row * blocks + block_index) * 34;
            const float d = dp_half_to_float(dp_load_u16(block));
            const __m256 scale = _mm256_set1_ps(d);
            float* dst =
                out + ((size_t)row * blocks + block_index) * 32;
            for (int j = 0; j < 32; j += 8) {
                const __m128i bytes =
                    _mm_loadl_epi64((const __m128i*)(block + 2 + j));
                const __m256i quantized = _mm256_cvtepi8_epi32(bytes);
                const __m256 values = _mm256_cvtepi32_ps(quantized);
                _mm256_storeu_ps(dst + j, _mm256_mul_ps(values, scale));
            }
        }
    }
}
