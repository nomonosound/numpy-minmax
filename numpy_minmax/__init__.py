from typing import Tuple

import _numpy_minmax
import numpy as np
from numpy.typing import NDArray

__version__ = "0.0.1"


def minmax(
    a: NDArray[np.float32],
) -> Tuple:
    """
    TODO
    """
    assert a.dtype == np.dtype("float32"), "The given array must be float32"
    assert a.ndim == 1
    assert a.flags["C_CONTIGUOUS"], "The arrays must be C-contiguous"

    result = _numpy_minmax.lib.minmax(
        _numpy_minmax.ffi.cast("float *", a.ctypes.data),
        len(a),
    )
    return result.min_val, result.max_val
