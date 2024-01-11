import time

import diplib as dip
import numpy as np

import numpy_minmax


class timer(object):
    """
    timer: A class used to measure the execution time of a block of code that is
    inside a "with" statement.

    Example:

    ```
    with timer("Count to 500000"):
        x = 0
        for i in range(500000):
            x += 1
        print(x)
    ```

    Will output:
    500000
    Count to 500000: 0.04 s
    """

    def __init__(self, description="Execution time"):
        self.description = description
        self.execution_time = None

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.execution_time = time.time() - self.t
        print("{}: {:.3f} s".format(self.description, self.execution_time))


def perf_benchmark_many_small_1d_c_contiguous():
    print("===\nperf_benchmark_many_small_1d_c_contiguous:")
    arrays = []
    for i in range(100_000):
        a = np.random.uniform(low=-4.0, high=3.9, size=(9,)).astype(np.float32)
        arrays.append(a)

    with timer("numpy.amax and numpy.amin sequentially"):
        for a in arrays:
            min_val = np.amin(a)
            max_val = np.amax(a)

    with timer("diplib"):
        for a in arrays:
            min_val, max_val = dip.MaximumAndMinimum(a)

    with timer("minmax") as t:
        for a in arrays:
            min_val, max_val = numpy_minmax.minmax(a)


def perf_benchmark_large_1d_c_contiguous():
    print("===\nperf_benchmark_large_1d_c_contiguous:")
    a = np.random.uniform(low=-4.0, high=3.9, size=(999_999_999,)).astype(np.float32)

    with timer("numpy.amax and numpy.amin sequentially"):
        min_val = np.amin(a)
        max_val = np.amax(a)
        print(min_val, max_val)

    with timer("diplib"):
        min_val, max_val = dip.MaximumAndMinimum(a)
        print(min_val, max_val, "diplib")

    times = []
    for i in range(5):
        with timer("minmax") as t:
            min_val, max_val = numpy_minmax.minmax(a)
            print(min_val, max_val)
        times.append(t.execution_time)

    print(f"===\nnumpy-minmax median: {np.median(times):.3f}")


def perf_benchmark_large_2d_c_contiguous():
    print("===\nperf_benchmark_large_2d_c_contiguous:")
    a = np.random.uniform(low=-4.0, high=3.9, size=(2, 999_999_999)).astype(np.float32)

    with timer("numpy.amax and numpy.amin sequentially"):
        min_val = np.amin(a)
        max_val = np.amax(a)
        print(min_val, max_val)

    with timer("diplib"):
        min_val, max_val = dip.MaximumAndMinimum(a)
        print(min_val, max_val, "diplib")

    times = []
    for i in range(5):
        with timer("minmax") as t:
            min_val, max_val = numpy_minmax.minmax(a)
            print(min_val, max_val)
        times.append(t.execution_time)

    print(f"===\nnumpy-minmax median: {np.median(times):.3f}")


def perf_benchmark_large_1d_not_c_contiguous():
    print("===\nperf_benchmark_large_1d_not_c_contiguous:")
    a = np.flip(
        np.random.uniform(low=-4.0, high=3.9, size=(999_999_999,)).astype(np.float32)
    )

    with timer("numpy.amax and numpy.amin sequentially"):
        min_val = np.amin(a)
        max_val = np.amax(a)
        print(min_val, max_val)

    with timer("diplib"):
        min_val, max_val = dip.MaximumAndMinimum(a)
        print(min_val, max_val, "diplib")

    times = []
    for i in range(5):
        with timer("minmax") as t:
            min_val, max_val = numpy_minmax.minmax(np.ascontiguousarray(a))
            print(min_val, max_val)
        times.append(t.execution_time)

    print(f"===\nnumpy-minmax median: {np.median(times):.3f}")


def perf_benchmark_large_2d_not_c_contiguous():
    print("===\nperf_benchmark_large_2d_not_c_contiguous:")
    a = np.flip(
        np.random.uniform(low=-4.0, high=3.9, size=(2, 999_999_999)).astype(np.float32)
    )

    times = []
    for i in range(5):
        with timer("minmax") as t:
            min_val, max_val = numpy_minmax.minmax(np.ascontiguousarray(a))
            print(min_val, max_val)
        times.append(t.execution_time)

    print(f"===\nnumpy-minmax median: {np.median(times):.3f}")

    with timer("numpy.amax and numpy.amin sequentially"):
        min_val = np.amin(a)
        max_val = np.amax(a)
        print(min_val, max_val)

    with timer("diplib"):
        min_val, max_val = dip.MaximumAndMinimum(a)
        print(min_val, max_val, "diplib")


if __name__ == "__main__":
    perf_benchmark_many_small_1d_c_contiguous()
    perf_benchmark_large_1d_c_contiguous()
    perf_benchmark_large_1d_not_c_contiguous()
    perf_benchmark_large_2d_c_contiguous()
    perf_benchmark_large_1d_not_c_contiguous()
