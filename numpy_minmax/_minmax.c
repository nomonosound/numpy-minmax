#include <immintrin.h>
#include <math.h>
#include <string.h>

typedef struct {
    float min_val;
    float max_val;
} MinMaxResult;

MinMaxResult minmax(float *a, size_t length) {
    MinMaxResult result;

    if (length <= 0) {
        // Handle the case where the array is empty or length is invalid
        result.min_val = FLT_MAX;
        result.max_val = FLT_MIN;
        return result;
    }

    // Initialize min and max with the first element of the array
    result.min_val = a[0];
    result.max_val = a[0];

    // Iterate over the array, starting from the second element
    for (int i = 1; i < length; i++) {
        if (a[i] < result.min_val) {
            result.min_val = a[i]; // Update min_val if a smaller value is found
        }
        if (a[i] > result.max_val) {
            result.max_val = a[i]; // Update max_val if a larger value is found
        }
    }

    return result;
}
