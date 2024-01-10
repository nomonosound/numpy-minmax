# numpy-minmax: a fast function for finding the minimum and maximum value in a numpy array

* Written in C and takes advantage of AVX2 for speed
* Roughly 2.3x faster than the numpy amin+amax equivalent (tested with numpy 1.26)
* Works for 1-dimensional float32 arrays

# Installation

```
$ pip install numpy-minmax
```

# Usage

```py
import numpy_minmax
import numpy as np

# TODO
```

# Development

* Install dev/build/test dependencies as denoted in setup.py
* `pip install -e .`
* `pytest`

# Acknowledgements

This library is maintained/backed by [Nomono](https://nomono.co/), a Norwegian audio AI startup.
