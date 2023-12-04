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

    // Initialize min and max with the first element of the array
    result.min_val = a[length-1];
    result.max_val = a[length-1];

    // Process elements in pairs
    for (int i = 0; i < length - 1; i += 2) {
        float smaller = a[i] < a[i + 1] ? a[i] : a[i + 1];
        float larger = a[i] < a[i + 1] ? a[i + 1] : a[i];

        result.min_val = result.min_val < smaller ? result.min_val : smaller;
        result.max_val = result.max_val > larger ? result.max_val : larger;
    }

    /*
    // old approach: requires 2n comparisons
    // Iterate over the array, starting from the second element
    for (int i = 1; i < length; i++) {
        if (a[i] < result.min_val) {
            result.min_val = a[i]; // Update min_val if a smaller value is found
        } else if (a[i] > result.max_val) {
            result.max_val = a[i]; // Update max_val if a larger value is found
        }
    }
    */

    return result;
}
