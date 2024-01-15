#include <immintrin.h>
#include <float.h>

#define REDUCE_RESULT_FROM_MM256(min_vals, max_vals, result) ({\
    float temp_min[8], temp_max[8]; \
    _mm256_storeu_ps(temp_min, min_vals); \
    _mm256_storeu_ps(temp_max, max_vals); \
    for (i = 0; i < 8; ++i) { \
        if (temp_min[i] < result.min_val) result.min_val = temp_min[i]; \
        if (temp_max[i] > result.max_val) result.max_val = temp_max[i]; \
}})

typedef struct {
    float min_val;
    float max_val;
} MinMaxResult;

MinMaxResult minmax_pairwise(float *a, size_t length) {
    MinMaxResult result;

    // Initialize min and max with the last element of the array.
    // This ensures that it works correctly for odd length arrays as well as even.
    result.min_val = a[length-1];
    result.max_val = result.min_val;

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


MinMaxResult minmax_avx(float *a, size_t length) {
    MinMaxResult result = { .min_val = FLT_MAX, .max_val = -FLT_MAX };

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
    for (; i < length; ++i) {
        if (a[i] < result.min_val) result.min_val = a[i];
        if (a[i] > result.max_val) result.max_val = a[i];
    }

    REDUCE_RESULT_FROM_MM256(min_vals, max_vals, result);

    return result;
}

MinMaxResult minmax_contiguous(float *a, size_t length) {
    // Return early for empty arrays
    if (length == 0) {
        return (MinMaxResult){0.0, 0.0};
    }
    if (length >= 16) {
        return minmax_avx(a, length);
    } else {
        return minmax_pairwise(a, length);
    }
}

// Takes the pairwise min/max on strided input. Strides are in number of bytes,
// which is why the data pointer is char
MinMaxResult minmax_pairwise_strided(char *a, size_t length, long stride) {
    MinMaxResult result;

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

// Takes the avx min/max on strided input. Strides are in number of bytes,
// which is why the data pointer is char
MinMaxResult minmax_avx_strided(char *a, size_t length, long stride) {
    MinMaxResult result = { .min_val = FLT_MAX, .max_val = -FLT_MAX };

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
    for (; i < length*stride; i+=stride) {
        if (*(float*)(a + i) < result.min_val) result.min_val = *(float*)(a + i);
        if (*(float*)(a + i) > result.max_val) result.max_val = *(float*)(a + i);
    }

    REDUCE_RESULT_FROM_MM256(min_vals, max_vals, result);

    return result;
}


MinMaxResult minmax_1d_strided(float *a, size_t length, long stride) {
    // Return early for empty arrays
    if (length == 0) {
        return (MinMaxResult){0.0, 0.0};
    }
    if (stride < 0){
        if (-stride == sizeof(float)){
            return minmax_contiguous(a - length + 1, length);
        }
        if (length < 16){
            return minmax_pairwise_strided((char*)(a) + (length - 1)*stride, length, -stride);
        } else {
            return minmax_avx_strided((char*)(a) + (length - 1)*stride, length, -stride);
        }
    }
    if (length < 16){
        return minmax_pairwise_strided((char*)a, length, stride);
    }
    return minmax_avx_strided((char*)a, length, stride);
}
