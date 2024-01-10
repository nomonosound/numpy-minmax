from typing import Tuple

import _numpy_minmax
import numpy as np
from numpy.typing import NDArray

__version__ = "0.0.1"


def minmax(a: NDArray) -> Tuple:
    if a.dtype == np.dtype("float32") and a.ndim == 1:
        if a.flags["C_CONTIGUOUS"]:
            result = _numpy_minmax.lib.minmax(
                _numpy_minmax.ffi.cast("float *", a.ctypes.data),
                len(a),
            )
        else:
            # TODO: There is room for improvement here, as diplib is ~3x faster in this case
            result = _numpy_minmax.lib.minmax(
                _numpy_minmax.ffi.cast("float *", np.ascontiguousarray(a).ctypes.data),
                len(a),
            )
        return np.float32(result.min_val), np.float32(result.max_val)
    else:
        return np.amin(a), np.amax(a)
