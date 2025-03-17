#include <float.h>
#include <stdbool.h>

#define IS_X86_64 (defined(__x86_64__) || defined(_M_X64))

#if IS_X86_64
    #include <immintrin.h>

    #ifdef _MSC_VER
        #include <intrin.h>  // MSVC
    #else
        #include <cpuid.h>  // GCC and Clang
    #endif

    #ifndef bit_AVX512F
    #define bit_AVX512F     (1 << 16)
    #endif
#endif


typedef struct {
    float min_val;
    float max_val;
} minmax_result_float32;

typedef struct {
    int16_t min_val;
    int16_t max_val;
} minmax_result_int16;


typedef unsigned char Byte;

#if IS_X86_64
bool system_supports_avx512() {
    unsigned int eax, ebx, ecx, edx;

    // EAX=7, ECX=0: Extended Features
    #ifdef _MSC_VER
        // MSVC
        int cpuInfo[4];
        __cpuid(cpuInfo, 7);
        ebx = cpuInfo[1];
    #else
        // GCC, Clang
        __cpuid(7, eax, ebx, ecx, edx);
    #endif

    // Check the AVX512F bit in EBX
    return (ebx & bit_AVX512F) != 0;
}
#endif

static inline minmax_result_int16 minmax_pairwise_int16(const int16_t *a, size_t length) {
    minmax_result_int16 result = {.min_val = a[length -1], .max_val = a[length-1]};

    for (size_t i = 0; i < length - 1; i += 2) {
        int16_t smaller = a[i] < a[i + 1] ? a[i] : a[i + 1];
        int16_t larger = a[i] < a[i + 1] ? a[i + 1] : a[i];

        if (smaller < result.min_val) {
            result.min_val = smaller;
        }
        if (larger > result.max_val) {
            result.max_val = larger;
        }
    }
    return result;
}

static inline minmax_result_float32 minmax_pairwise_float32(const float *a, size_t length) {
    // Initialize min and max with the last element of the array.
    // This ensures that it works correctly for odd length arrays as well as even.
    minmax_result_float32 result = {.min_val = a[length -1], .max_val = a[length-1]};

    // Process elements in pairs
    for (size_t i = 0; i < length - 1; i += 2) {
        float smaller = a[i] < a[i + 1] ? a[i] : a[i + 1];
        float larger = a[i] < a[i + 1] ? a[i + 1] : a[i];

        if (smaller < result.min_val) {
            result.min_val = smaller;
        }
        if (larger > result.max_val) {
            result.max_val = larger;
        }
    }
    return result;
}

#if IS_X86_64
static inline minmax_result_int16 reduce_result_from_mm256i_int16(__m256i min_vals, __m256i max_vals, minmax_result_int16 result) {
    int16_t temp_min[16], temp_max[16];
    _mm256_storeu_si256((__m256i*)temp_min, min_vals);
    _mm256_storeu_si256((__m256i*)temp_max, max_vals);
    for (size_t i = 0; i < 16; ++i) {
        if (temp_min[i] < result.min_val) result.min_val = temp_min[i];
        if (temp_max[i] > result.max_val) result.max_val = temp_max[i];
    }
    return result;
}

static inline minmax_result_float32 reduce_result_from_mm256_float32(__m256 min_vals, __m256 max_vals, minmax_result_float32 result) {
    float temp_min[8], temp_max[8];
    _mm256_storeu_ps(temp_min, min_vals);
    _mm256_storeu_ps(temp_max, max_vals);
    for (size_t i = 0; i < 8; ++i) {
        if (temp_min[i] < result.min_val) result.min_val = temp_min[i];
        if (temp_max[i] > result.max_val) result.max_val = temp_max[i];
    }
    return result;
}

minmax_result_int16 minmax_avx_int16(const int16_t *a, size_t length) {
    minmax_result_int16 result = { .min_val = INT16_MAX, .max_val = INT16_MIN };

    __m256i min_vals = _mm256_loadu_si256((__m256i*)a);
    __m256i max_vals = min_vals;

    // Process elements in chunks of 16 (256 bits / 16 bits per int16_t)
    size_t i = 16;
    for (; i <= length - 16; i += 16) {
        __m256i vals = _mm256_loadu_si256((__m256i*)(a + i));
        min_vals = _mm256_min_epi16(min_vals, vals);
        max_vals = _mm256_max_epi16(max_vals, vals);
    }
    // Process remainder elements
    if (i < length) {
        result = minmax_pairwise_int16(a + i, length - i);
    }

    return reduce_result_from_mm256i_int16(min_vals, max_vals, result);
}

minmax_result_float32 minmax_avx_float32(const float *a, size_t length) {
    minmax_result_float32 result = { .min_val = FLT_MAX, .max_val = -FLT_MAX };

    __m256 min_vals = _mm256_loadu_ps(a);
    __m256 max_vals = min_vals;

    // Process elements in chunks of eight
    size_t i = 8;
    for (; i <= length - 8; i += 8) {
        __m256 vals = _mm256_loadu_ps(a + i);
        min_vals = _mm256_min_ps(min_vals, vals);
        max_vals = _mm256_max_ps(max_vals, vals);
    }
    // Process remainder elements
    if (i < length) {
        result = minmax_pairwise_float32(a + i, length - i);
    }

    return reduce_result_from_mm256_float32(min_vals, max_vals, result);
}

static inline minmax_result_float32 reduce_result_from_mm512_float32(__m512 min_vals, __m512 max_vals, minmax_result_float32 result) {
    float temp_min[16], temp_max[16];
    _mm512_storeu_ps(temp_min, min_vals);
    _mm512_storeu_ps(temp_max, max_vals);
    for (size_t i = 0; i < 16; ++i) {
        if (temp_min[i] < result.min_val) result.min_val = temp_min[i];
        if (temp_max[i] > result.max_val) result.max_val = temp_max[i];
    }
    return result;
}

minmax_result_float32 minmax_avx512_float32(const float *a, size_t length) {
    minmax_result_float32 result = { .min_val = FLT_MAX, .max_val = -FLT_MAX };

    __m512 min_vals = _mm512_loadu_ps(a);
    __m512 max_vals = min_vals;

    // Process elements in chunks of sixteen
    size_t i = 16;
    for (; i <= length - 16; i += 16) {
        __m512 vals = _mm512_loadu_ps(a + i);
        min_vals = _mm512_min_ps(min_vals, vals);
        max_vals = _mm512_max_ps(max_vals, vals);
    }

    // Process remainder elements
    if (i < length) {
        result = minmax_pairwise_float32(a + i, length - i);
    }

    return reduce_result_from_mm512_float32(min_vals, max_vals, result);
}
#endif

minmax_result_int16 minmax_contiguous_int16(const int16_t *a, size_t length) {
    // Return early for empty arrays
    if (length == 0) {
        return (minmax_result_int16){0, 0};
    }

#if IS_X86_64
    if (length >= 16) {
        // TODO: Consider adding AVX512 support
        return minmax_avx_int16(a, length);
    } else {
        return minmax_pairwise_int16(a, length);
    }
#else
    return minmax_pairwise_int16(a, length);
#endif
}

minmax_result_float32 minmax_contiguous_float32(const float *a, size_t length) {
    // Return early for empty arrays
    if (length == 0) {
        return (minmax_result_float32){0.0, 0.0};
    }

#if IS_X86_64
    if (length >= 16) {
        if (system_supports_avx512()) {
            return minmax_avx512_float32(a, length);
        } else {
            return minmax_avx_float32(a, length);
        }
    } else {
        return minmax_pairwise_float32(a, length);
    }
#else
    return minmax_pairwise_float32(a, length);
#endif
}

// Takes the pairwise min/max on strided input. Strides are in number of bytes,
// which is why the data pointer is Byte (i.e. unsigned char)
minmax_result_float32 minmax_pairwise_strided_float32(const Byte *a, size_t length, long stride) {
    minmax_result_float32 result;

    // Initialize min and max with the last element of the array.
    // This ensures that it works correctly for odd length arrays as well as even.
    result.min_val = *(float*)(a + (length -1)*stride);
    result.max_val = result.min_val;

    // Process elements in pairs
    float smaller;
    float larger;
    for (size_t i = 0; i < (length - 1)*stride; i += 2*stride) {
        if (*(float*)(a + i) < *(float*)(a + i + stride)) {
            smaller = *(float*)(a + i);
            larger = *(float*)(a + i + stride);
        } else {
            smaller = *(float*)(a + i + stride);
            larger = *(float*)(a + i);
        }

        if (smaller < result.min_val) {
            result.min_val = smaller;
        }
        if (larger > result.max_val) {
            result.max_val = larger;
        }
    }
    return result;
}

#if IS_X86_64
// Takes the avx min/max on strided input. Strides are in number of bytes,
// which is why the data pointer is Byte (i.e. unsigned char)
minmax_result_float32 minmax_avx_strided_float32(const Byte *a, size_t length, long stride) {
    minmax_result_float32 result = { .min_val = FLT_MAX, .max_val = -FLT_MAX };

    // This is faster than intrinsic gather on tested platforms
    __m256 min_vals = _mm256_set_ps(
        *(float*)(a),
        *(float*)(a + stride),
        *(float*)(a + 2*stride),
        *(float*)(a + 3*stride),
        *(float*)(a + 4*stride),
        *(float*)(a + 5*stride),
        *(float*)(a + 6*stride),
        *(float*)(a + 7*stride)
        );
    __m256 max_vals = min_vals;

    // Process elements in chunks of eight
    size_t i = 8*stride;
    for (; i <= (length - 8)*stride; i += 8*stride) {
        __m256 vals = _mm256_set_ps(
            *(float*)(a + i),
            *(float*)(a + i + stride),
            *(float*)(a + i + 2*stride),
            *(float*)(a + i + 3*stride),
            *(float*)(a + i + 4*stride),
            *(float*)(a + i + 5*stride),
            *(float*)(a + i + 6*stride),
            *(float*)(a + i + 7*stride)
            );
        min_vals = _mm256_min_ps(min_vals, vals);
        max_vals = _mm256_max_ps(max_vals, vals);
    }

    // Process remainder elements
    if (i < length*stride){
        result = minmax_pairwise_strided_float32(a + i, length - i / stride, stride);
    }

    return reduce_result_from_mm256_float32(min_vals, max_vals, result);
}
#endif


minmax_result_float32 minmax_1d_strided_float32(const float *a, size_t length, long stride) {
    // Return early for empty arrays
    if (length == 0) {
        return (minmax_result_float32){0.0, 0.0};
    }

    if (stride < 0){
        if (-stride == sizeof(float)){
            return minmax_contiguous_float32(a - length + 1, length);
        }
        return minmax_pairwise_strided_float32((Byte*)(a) + (length - 1)*stride, length, -stride);
    }

#if IS_X86_64
    if (length < 16){
        return minmax_pairwise_strided_float32((Byte*)a, length, stride);
    }
    return minmax_avx_strided_float32((Byte*)a, length, stride);
#else
    return minmax_pairwise_strided_float32((Byte*)a, length, stride);
#endif
}
