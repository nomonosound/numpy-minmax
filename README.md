# numpy-minmax: a fast function for finding the minimum and maximum value in a numpy array

Numpy lacked an optimized minmax function, so we wrote our own.

* Written in C and takes advantage of AVX2 for speed
* Roughly 2.3x faster than the numpy amin+amax equivalent (tested with numpy 1.24-1.26)
* The fast implementation is tailored for C-contiguous 1-dimensional and 2-dimensional float32 arrays. Other types of arrays get processed with numpy.amin and numpy.amax, so no perf gain there.

# Installation

```
$ pip install numpy-minmax
```

# Usage

```py
import numpy_minmax
import numpy as np

arr = np.arange(1337, dtype=np.float32)
min_val, max_val = numpy_minmax.minmax(arr)  # 0.0, 1336.0
```

# Development

* Install dev/build/test dependencies as denoted in setup.py
* `CC=clang pip install -e .`
* `pytest`

# Acknowledgements

This library is maintained/backed by [Nomono](https://nomono.co/), a Norwegian audio AI startup.
