from typing import Tuple

import _numpy_minmax
import numpy as np
from numpy.typing import NDArray

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


def minmax(a: NDArray) -> Tuple:
    if 0 in a.shape:
        raise ValueError("Cannot find min/max value in empty array")
    if a.dtype == np.dtype("float32"):
        if a.flags["C_CONTIGUOUS"] or a.flags["F_CONTIGUOUS"]:
            result = _numpy_minmax.lib.minmax_contiguous_float32(
                _numpy_minmax.ffi.cast("float *", a.ctypes.data), a.size
            )
            return np.float32(result.min_val), np.float32(result.max_val)
        if a.ndim == 1:
            result = _numpy_minmax.lib.minmax_1d_strided_float32(
                _numpy_minmax.ffi.cast("float *", a.ctypes.data), a.size, a.strides[0]
            )
            return np.float32(result.min_val), np.float32(result.max_val)
        # TODO: Find multi-dim arrays that can be simplified to a single stride
    elif a.dtype == np.dtype("int16") and (
        a.flags["C_CONTIGUOUS"] or a.flags["F_CONTIGUOUS"]
    ):
        result = _numpy_minmax.lib.minmax_contiguous_int16(
            _numpy_minmax.ffi.cast("int16_t *", a.ctypes.data), a.size
        )
        return np.int16(result.min_val), np.int16(result.max_val)

    return np.amin(a), np.amax(a)
