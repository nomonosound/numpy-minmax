import numpy as np

import numpy_minmax


class TestMinMax:
    def test_minmax_even(self):
        arr = np.array([0.0, 1.0, -2.0, 0.0], dtype=np.float32)
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == -2.0
        assert max_val == 1.0

    def test_minmax_odd(self):
        arr = np.array([0.0, 1.0, -2.0, -5.0], dtype=np.float32)
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == -5.0
        assert max_val == 1.0
