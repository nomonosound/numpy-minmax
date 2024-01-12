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
    if a.dtype == np.dtype("float32") and (a.flags["C_CONTIGUOUS"] or a.flags["F_CONTIGUOUS"]):
        result = _numpy_minmax.lib.minmax_contiguous(
            _numpy_minmax.ffi.cast("float *", a.ctypes.data), a.size
        )
        return np.float32(result.min_val), np.float32(result.max_val)
    return np.amin(a), np.amax(a)
