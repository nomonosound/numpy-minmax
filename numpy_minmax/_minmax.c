#include <immintrin.h>

typedef struct {
    float min_val;
    float max_val;
} MinMaxResult;


MinMaxResult minmax(float *a, size_t length) {
    MinMaxResult result;

    if (length == 0) {
        result.min_val = 0.0;
        result.max_val = 0.0;
        return result;
    }

    // Initialize min and max with the first element
    result.min_val = a[0];
    result.max_val = a[0];

    __m256 min_vals = _mm256_set1_ps(result.min_val);
    __m256 max_vals = _mm256_set1_ps(result.max_val);

    // Process eight elements at a time
    size_t i;
    for (i = 0; i <= length - 8; i += 8) {
        __m256 vals = _mm256_loadu_ps(&a[i]); // using loadu for unaligned data
        min_vals = _mm256_min_ps(min_vals, vals);
        max_vals = _mm256_max_ps(max_vals, vals);
    }

    // Handling the remainder of the array
    for (; i < length; ++i) {
        if (a[i] < result.min_val) result.min_val = a[i];
        if (a[i] > result.max_val) result.max_val = a[i];
    }

    // Extracting results from AVX registers
    float temp_min[8], temp_max[8];
    _mm256_storeu_ps(temp_min, min_vals);
    _mm256_storeu_ps(temp_max, max_vals);
    for (i = 0; i < 8; ++i) {
        if (temp_min[i] < result.min_val) result.min_val = temp_min[i];
        if (temp_max[i] > result.max_val) result.max_val = temp_max[i];
    }

    return result;
}
