# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-03-17

### Changes

* Target numpy 2.x instead of numpy 1.x. If you still depend on numpy 1.x, you need an older version of numpy-minmax.
* Process some types of arrays with a negative stride in a scalar way instead of vectorized for improved compatibility

### Removed

* Remove support for Python 3.8

## [0.3.1] - 2024-08-15

### Changes

* Optimize (with AVX) the processing of contiguous int16 arrays. ~2.3x speedup compared to 0.3.0

## [0.3.0] - 2024-07-29

### Added

* Distribute source

### Changes

* Add support for ARM (without NEON optimizations for now) on Linux and macOS
* Update supported numpy version range to >=1.21,<2 

## [0.2.1] - 2024-03-12

### Changes

* Add support for AVX512. It will only be used if the CPU reports that it supports it.
* Compile builds for linux with clang instead of gcc, as this seems to yield tiny performance improvements

## [0.2.0] - 2024-01-17

### Changes

* Add support for Python 3.12
* Significantly speed up the processing of 1-dimensional strided arrays
* Slightly speed up the processing of ndarrays with at least 16 items

## [0.1.1] - 2024-01-15

### Changes

* Slightly speed up the processing of 2D arrays
* Speed up the processing of arrays with ndim > 2
* Speed up the processing of F-contiguous ndarrays

## [0.1.0] - 2024-01-11

Initial release
