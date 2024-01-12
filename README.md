# numpy-minmax: a fast function for finding the minimum and maximum value in a NumPy array

NumPy lacked an optimized minmax function, so we wrote our own. At Nomono, we use it for audio processing, but it can also be applied to other kinds of data of similar shape.

* Written in C and takes advantage of AVX2 for speed
* Roughly **2.3x speedup** compared to the numpy amin+amax equivalent (tested with numpy 1.24-1.26)
* The fast implementation is tailored for C-contiguous 1-dimensional and 2-dimensional float32 arrays. Other types of arrays get processed with numpy.amin and numpy.amax, so no perf gain there.

# Installation

[![PyPI version](https://img.shields.io/pypi/v/numpy-minmax.svg?style=flat)](https://pypi.org/project/numpy-minmax/)
![python 3.8, 3.9, 3.10, 3.11](https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10%20|%203.11-blue)
![os: Linux, Windows](https://img.shields.io/badge/OS-Linux%20|%20Windows-blue)

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

# Running benchmarks
* Install diplib `pip install diplib`
* `python scripts/perf_benchmark.py`

# Acknowledgements

This library is maintained/backed by [Nomono](https://nomono.co/), a Norwegian audio AI startup.
