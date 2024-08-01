import numpy as np
import pytest

import numpy_minmax


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minmax_even(dtype):
    arr = np.array([0, 1, -2, 0], dtype=dtype)
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == -2
    assert max_val == 1
    assert isinstance(min_val, dtype)
    assert isinstance(max_val, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minmax_odd(dtype):
    arr = np.array([1, -2, -5], dtype=dtype)
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == -5
    assert max_val == 1


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minmax_single_item(dtype):
    arr = np.array([1337], dtype=dtype)
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == 1337
    assert max_val == 1337


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minmax_thirteen_min_value_first(dtype):
    arr = np.arange(13, dtype=dtype)
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == 0
    assert max_val == 12


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minmax_thirteen_min_value_last_and_not_contiguous(dtype):
    arr = np.flip(np.arange(13, dtype=dtype))
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == 0
    assert max_val == 12


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minmax_999_values(dtype):
    arr = np.arange(999, dtype=dtype)
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == 0
    assert max_val == 998
    assert isinstance(min_val, dtype)
    assert isinstance(max_val, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minmax_999_positive_values(dtype):
    offset = 5
    arr = np.arange(999, dtype=dtype) + offset
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == offset
    assert max_val == 998 + offset
    assert isinstance(min_val, dtype)
    assert isinstance(max_val, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minmax_4_positive_values(dtype):
    offset = 5
    arr = np.arange(4, dtype=dtype) + offset
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == offset
    assert max_val == 3 + offset
    assert isinstance(min_val, dtype)
    assert isinstance(max_val, dtype)


def test_minmax_float64_numpy_fallback():
    arr = np.arange(17, dtype=np.float64)
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == 0.0
    assert max_val == 16.0
    assert isinstance(min_val, np.float64)
    assert isinstance(max_val, np.float64)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minmax_2d_small1(dtype):
    np.random.seed(1)
    arr = np.random.uniform(low=-6.0, high=3.0, size=(15, 2)).astype(dtype)
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == np.amin(arr)
    assert max_val == np.amax(arr)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minmax_2d_small2(dtype):
    np.random.seed(2)
    arr = np.random.uniform(low=-6.0, high=3.0, size=(2, 15)).astype(dtype)
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == np.amin(arr)
    assert max_val == np.amax(arr)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minmax_2d_shape_large(dtype):
    np.random.seed(3)
    arr = np.random.uniform(low=-6.0, high=3.0, size=(2, 999)).astype(dtype)
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == np.amin(arr)
    assert max_val == np.amax(arr)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minimax_2d_f_contiguous(dtype):
    np.random.seed(4)
    arr = np.random.uniform(low=-6.0, high=3.0, size=(2, 27)).astype(dtype)
    arr = np.asfortranarray(arr)
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == np.amin(arr)
    assert max_val == np.amax(arr)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minimax_1d_non_contiguous_short(dtype):
    np.random.seed(5)
    arr = np.random.uniform(low=-6.0, high=3.0, size=(27,)).astype(dtype)[::3]
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == np.amin(arr)
    assert max_val == np.amax(arr)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minimax_1d_non_contiguous(dtype):
    np.random.seed(6)
    arr = np.random.uniform(low=-6.0, high=3.0, size=(27,)).astype(dtype)[::2]
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == np.amin(arr)
    assert max_val == np.amax(arr)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minimax_1d_negative_stride(dtype):
    np.random.seed(7)
    arr = np.random.uniform(low=-6.0, high=3.0, size=(27,)).astype(dtype)[::-1]
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == np.amin(arr)
    assert max_val == np.amax(arr)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minimax_1d_non_contiguous_negative_stride(dtype):
    np.random.seed(7)
    arr = np.random.uniform(low=-6.0, high=3.0, size=(61,)).astype(dtype)[::-2]
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == np.amin(arr)
    assert max_val == np.amax(arr)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minimax_1d_non_contiguous_negative_stride_increasing(dtype):
    np.random.seed(8)
    arr = np.arange(start=17, step=-1, stop=-19, dtype=dtype)[::-2]
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == np.amin(arr)
    assert max_val == np.amax(arr)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minimax_1d_non_contiguous_negative_stride_decreasing(dtype):
    np.random.seed(9)
    arr = np.arange(start=-21, stop=13, dtype=dtype)[::-2]
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == np.amin(arr)
    assert max_val == np.amax(arr)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minimax_1d_non_contiguous_negative_stride_short(dtype):
    np.random.seed(10)
    arr = np.random.uniform(low=-6.0, high=3.0, size=(27,)).astype(dtype)[::-3]
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == np.amin(arr)
    assert max_val == np.amax(arr)


def test_minmax_unaligned():
    # Allocate memory and create an unaligned array from that
    buf = np.arange(402, dtype=np.uint8)
    arr = np.frombuffer(buf.data, offset=2, count=100, dtype=np.float32)
    arr.shape = 10, 10
    assert arr.flags["ALIGNED"] == False

    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == np.amin(arr)
    assert max_val == np.amax(arr)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minmax_3d_shape(dtype):
    arr = np.random.uniform(low=-6.0, high=3.0, size=(2, 2, 16)).astype(dtype)
    min_val, max_val = numpy_minmax.minmax(arr)
    assert min_val == np.amin(arr)
    assert max_val == np.amax(arr)


@pytest.mark.parametrize("shape", [(0,), (0, 0)])
@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_minmax_empty_array(dtype, shape):
    arr = np.empty(shape=shape, dtype=dtype)
    with pytest.raises(ValueError):
        numpy_minmax.minmax(arr)
