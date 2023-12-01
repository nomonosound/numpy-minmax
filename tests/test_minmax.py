import time

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


class TestMinMax:
    def test_minmax(self):
        arr = np.array([0.0, 1.0, -2.0, 0.0], dtype=np.float32)
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == -2.0
        assert max_val == 1.0

    def test_perf(self):
        a = np.random.uniform(low=-4.0, high=3.9, size=(999_999_999,)).astype(np.float32)

        with timer("numpy.amax and numpy.amin sequentially"):
            min_val = np.amin(a)
            max_val = np.amax(a)
            print(min_val, max_val)

        with timer("minmax"):
            min_val, max_val = numpy_minmax.minmax(a)
            print(min_val, max_val)
