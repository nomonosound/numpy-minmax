import os
import platform
from cffi import FFI

ffibuilder = FFI()
ffibuilder.cdef("""
    typedef struct {
        float min_val;
        float max_val;
    } MinMaxResult;
""")
ffibuilder.cdef("MinMaxResult minmax_contiguous(float *, size_t);")
ffibuilder.cdef("MinMaxResult minmax_1d_strided(float *, size_t, long);")

script_dir = os.path.dirname(os.path.realpath(__file__))
c_file_path = os.path.join(script_dir, "_minmax.c")

with open(c_file_path, "r") as file:
    c_code = file.read()

extra_compile_args = ["-O3", "-Wall"]
if os.name == "posix":
    extra_compile_args.append("-Wextra")

# Detect architecture and set appropriate SIMD-related compile args
if platform.machine().lower() in ["x86_64", "amd64", "i386", "i686"]:
    extra_compile_args.append("-mavx")
    extra_compile_args.append("-mavx512f")

ffibuilder.set_source("_numpy_minmax", c_code, extra_compile_args=extra_compile_args)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
