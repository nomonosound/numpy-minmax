typedef struct {
    float min_val;
    float max_val;
} MinMaxResult;

MinMaxResult minmax(float *a, size_t length) {
    MinMaxResult result;

    if (length <= 0) {
        // Handle the case where the array is empty or length is invalid
        result.min_val = 0.0;
        result.max_val = 0.0;
        return result;
    }

    // Initialize min and max with the last element of the array
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
