#include <immintrin.h>
#include <float.h>

typedef struct {
    float min_val;
    float max_val;
} MinMaxResult;

MinMaxResult minmax(float *a, size_t length) {
    MinMaxResult result = { .min_val = FLT_MAX, .max_val = -FLT_MAX };

    // Return early for empty arrays
    if (length == 0) {
        return (MinMaxResult){0.0, 0.0};
    }

    __m256 min_vals = _mm256_set1_ps(result.min_val);
    __m256 max_vals = _mm256_set1_ps(result.max_val);

    // Process elements in chunks of eight
    size_t i = 0;
    for (; i <= length - 8; i += 8) {
        __m256 vals = _mm256_loadu_ps(&a[i]);
        min_vals = _mm256_min_ps(min_vals, vals);
        max_vals = _mm256_max_ps(max_vals, vals);
    }

    // Process remainder elements
    for (; i < length; ++i) {
        if (a[i] < result.min_val) result.min_val = a[i];
        if (a[i] > result.max_val) result.max_val = a[i];
    }

    // Reduce min and max values from AVX registers
    float temp_min[8], temp_max[8];
    _mm256_storeu_ps(temp_min, min_vals);
    _mm256_storeu_ps(temp_max, max_vals);
    for (i = 0; i < 8; ++i) {
        if (temp_min[i] < result.min_val) result.min_val = temp_min[i];
        if (temp_max[i] > result.max_val) result.max_val = temp_max[i];
    }

    return result;
}
