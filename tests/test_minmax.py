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

    def test_minmax_2d_small1(self):
        arr = np.random.uniform(low=-6.0, high=3.0, size=(15, 2)).astype(np.float32)
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == np.amin(arr)
        assert max_val == np.amax(arr)

    def test_minmax_2d_small2(self):
        arr = np.random.uniform(low=-6.0, high=3.0, size=(2, 15)).astype(np.float32)
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == np.amin(arr)
        assert max_val == np.amax(arr)

    def test_minmax_2d_shape_large(self):
        arr = np.random.uniform(low=-6.0, high=3.0, size=(2, 999)).astype(np.float32)
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == np.amin(arr)
        assert max_val == np.amax(arr)

    def test_minimax_2d_f_contiguous(self):
        arr = np.random.uniform(low=-6.0, high=3.0, size=(2, 27)).astype(np.float32)
        arr = np.asfortranarray(arr)
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == np.amin(arr)
        assert max_val == np.amax(arr)

    def test_minimax_1d_non_contiguous_short(self):
        arr = np.random.uniform(low=-6.0, high=3.0, size=(27,)).astype(np.float32)[::3]
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == np.amin(arr)
        assert max_val == np.amax(arr)

    def test_minimax_1d_non_contiguous(self):
        arr = np.random.uniform(low=-6.0, high=3.0, size=(27,)).astype(np.float32)[::2]
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == np.amin(arr)
        assert max_val == np.amax(arr)

    def test_minimax_1d_negative_stride(self):
        arr = np.random.uniform(low=-6.0, high=3.0, size=(27,)).astype(np.float32)[::-1]
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == np.amin(arr)
        assert max_val == np.amax(arr)

    def test_minimax_1d_non_contiguous_negative_stride(self):
        arr = np.random.uniform(low=-6.0, high=3.0, size=(61,)).astype(np.float32)[::-2]
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == np.amin(arr)
        assert max_val == np.amax(arr)

    def test_minimax_1d_non_contiguous_negative_stride_increasing(self):
        arr = np.arange(start = 17, step= -1, stop = -19, dtype=np.float32)[::-2]
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == np.amin(arr)
        assert max_val == np.amax(arr)

    def test_minimax_1d_non_contiguous_negative_stride_decreasing(self):
        arr = np.arange(start = -21, stop = 13, dtype=np.float32)[::-2]
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == np.amin(arr)
        assert max_val == np.amax(arr)

    def test_minimax_1d_non_contiguous_negative_stride_short(self):
        arr = np.random.uniform(low=-6.0, high=3.0, size=(27,)).astype(np.float32)[::-3]
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == np.amin(arr)
        assert max_val == np.amax(arr)

    def test_minmax_unaligned(self):
        # Allocate memory and create an unaligned array from that
        buf = np.arange(402, dtype=np.uint8)
        arr = np.frombuffer(buf.data, offset=2, count=100, dtype=np.float32)
        arr.shape = 10, 10
        assert arr.flags["ALIGNED"] == False

        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == np.amin(arr)
        assert max_val == np.amax(arr)

    def test_minmax_3d_shape(self):
        arr = np.random.uniform(low=-6.0, high=3.0, size=(2, 2, 16)).astype(np.float32)
        min_val, max_val = numpy_minmax.minmax(arr)
        assert min_val == np.amin(arr)
        assert max_val == np.amax(arr)

    @pytest.mark.parametrize("shape", [(0,), (0, 0)])
    def test_minmax_empty_array(self, shape):
        arr = np.empty(shape=shape, dtype=np.float32)
        with pytest.raises(ValueError):
            numpy_minmax.minmax(arr)
