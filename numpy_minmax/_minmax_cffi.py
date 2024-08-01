import os
import platform
from cffi import FFI

ffibuilder = FFI()
ffibuilder.cdef("""
    typedef struct {
        int16_t min_val;
        int16_t max_val;
    } minmax_result_int16;
""")
ffibuilder.cdef("""
    typedef struct {
        float min_val;
        float max_val;
    } minmax_result_float32;
""")
ffibuilder.cdef("minmax_result_int16 minmax_contiguous_int16(int16_t *, size_t);")
ffibuilder.cdef("minmax_result_float32 minmax_contiguous_float32(float *, size_t);")
ffibuilder.cdef("minmax_result_float32 minmax_1d_strided_float32(float *, size_t, long);")

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
