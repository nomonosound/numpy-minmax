import numpy as np
import pytest

import numpy_minmax


class TestMinMax:
    def test_minmax_even(self):
        arr = np.array([0.0, 1.0, -2.0, 0.0], dtype=np.float32)
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == -2.0
        assert max_val == 1.0
        assert isinstance(min_val, np.float32)
        assert isinstance(max_val, np.float32)

    def test_minmax_odd(self):
        arr = np.array([1.0, -2.0, -5.0], dtype=np.float32)
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == -5.0
        assert max_val == 1.0

    def test_minmax_single_item(self):
        arr = np.array([1337.0], dtype=np.float32)
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == 1337.0
        assert max_val == 1337.0

    def test_minmax_thirteen_min_value_first(self):
        arr = np.arange(13, dtype=np.float32)
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == 0.0
        assert max_val == 12.0

    def test_minmax_thirteen_min_value_last_and_not_contiguous(self):
        arr = np.flip(np.arange(13, dtype=np.float32))
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == 0.0
        assert max_val == 12.0

    def test_minmax_999_values(self):
        arr = np.arange(999, dtype=np.float32)
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == 0.0
        assert max_val == 998.0
        assert isinstance(min_val, np.float32)
        assert isinstance(max_val, np.float32)

    def test_minmax_float64_numpy_fallback(self):
        arr = np.arange(17, dtype=np.float64)
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == 0.0
        assert max_val == 16.0
        assert isinstance(min_val, np.float64)
        assert isinstance(max_val, np.float64)

    def test_minmax_2d_shape_numpy_fallback(self):
        arr = np.arange(16, dtype=np.float32).reshape((2, 8))
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == 0.0
        assert max_val == 15.0

    def test_minmax_empty_array(self):
        arr = np.empty(shape=(0, 0), dtype=np.float32)
        with pytest.raises(ValueError):
            numpy_minmax.minmax(arr)
