from setuptools import setup

setup(
    cffi_modules=["numpy_minmax/_minmax_cffi.py:ffibuilder"],
)
