# Supplemental validation record

These are the host-specific probes used on 2026-09-09. They complement, but do not
replace, a fresh same-toolchain build of the patched upstream Q1/Q6 targets.
Paths refer to the GH200 development host described in [README.md](README.md).

## Standalone consumer of the existing cuDF build

The ignored `build/cudf-vortex-smoke/CMakeLists.txt` contains:

```cmake
cmake_minimum_required(VERSION 4.0)
project(CudfVortexSmoke LANGUAGES C CXX CUDA)
find_package(cudf CONFIG REQUIRED)
# Stand-ins for the existing benchmark targets; only the smoke target is built.
foreach(name IN ITEMS NDSH_Q01_NVBENCH NDSH_Q06_NVBENCH)
  add_executable(${name} EXCLUDE_FROM_ALL
    ../cudf-ndsh-src/cpp/benchmarks/ndsh/vortex_build_smoke.cpp)
endforeach()
include(../cudf-ndsh-src/cpp/benchmarks/ndsh/vortex.cmake)
```

From the Vortex root, the configuration and build commands were:

```sh
env NVCC_PREPEND_FLAGS= \
  NVCC_CCBIN=/home/ubuntu/micromamba/envs/cudf-cpp-min/bin/aarch64-conda-linux-gnu-g++ \
  LIBCLANG_PATH=/usr/lib/llvm-18/lib \
  cmake -S build/cudf-vortex-smoke -B build/cudf-vortex-smoke/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/home/ubuntu/micromamba/envs/cudf-cpp-min/bin/aarch64-conda-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=/home/ubuntu/micromamba/envs/cudf-cpp-min/bin/aarch64-conda-linux-gnu-g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/home/ubuntu/micromamba/envs/cudf-cpp-min/bin/aarch64-conda-linux-gnu-g++ \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.1 \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -Dcudf_DIR=/home/ubuntu/cudf/cpp/build \
  -DCMAKE_PREFIX_PATH=/home/ubuntu/micromamba/envs/cudf-cpp-min \
  -DCUDF_NDSH_WITH_VORTEX=ON \
  -DFETCHCONTENT_SOURCE_DIR_VORTEX=/home/ubuntu/vortex

env NVCC_PREPEND_FLAGS= \
  NVCC_CCBIN=/home/ubuntu/micromamba/envs/cudf-cpp-min/bin/aarch64-conda-linux-gnu-g++ \
  LIBCLANG_PATH=/usr/lib/llvm-18/lib CARGO_BUILD_JOBS=4 \
  cmake --build build/cudf-vortex-smoke/build \
  --target NDSH_VORTEX_BUILD_SMOKE --parallel 2
```

Configuration and compilation passed; the final link failed. The existing cuDF
library was built against system glibc, not the Conda glibc-2.28 sysroot. Its
transitive nanoarrow dependency also produced unresolved `cudfArrow*` references.

The full upstream configure also passed using the same environment and compiler,
CUDA, prefix, and Vortex options, replacing the source/build arguments with
`-S build/cudf-ndsh-src/cpp -B build/cudf-ndsh-build`, omitting `cudf_DIR`, and
adding `-DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=ON`. That full build was not run.

## Native relink to isolate the prebuilt-library mismatch

The following linked the already-compiled objects and archives against the
system runtime and supplied the existing nanoarrow dependency's link search path.
It did not modify the integration or either cuDF library. Run in
`build/cudf-vortex-smoke/build`:

```sh
/usr/bin/g++ -O3 -DNDEBUG \
  CMakeFiles/NDSH_VORTEX_BUILD_SMOKE.dir/home/ubuntu/vortex/build/cudf-ndsh-src/cpp/benchmarks/ndsh/vortex_build_smoke.cpp.o \
  -o NDSH_VORTEX_BUILD_SMOKE_NATIVE \
  -Wl,-rpath,/home/ubuntu/cudf/cpp/build:/home/ubuntu/cudf/cpp/build/_deps/rmm-build:/home/ubuntu/cudf/cpp/build/_deps/rapids_logger-build \
  -Wl,-rpath-link,/home/ubuntu/cudf/cpp/build/_deps/nanoarrow-build \
  /home/ubuntu/cudf/cpp/build/libcudf.so \
  _deps/vortex-build/libvortex_cxx.a \
  /home/ubuntu/cudf/cpp/build/_deps/rmm-build/librmm.so \
  /home/ubuntu/cudf/cpp/build/_deps/rapids_logger-build/librapids_logger.so \
  /usr/local/cuda-13.1/lib64/libcudart_static.a \
  _deps/vortex-build/ffi/vortex-artifacts/libvortex_ffi.a \
  -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc
./NDSH_VORTEX_BUILD_SMOKE_NATIVE
```

Result: `cuDF + Vortex C++/CUDA FFI initialization succeeded`.

## CUB/nvCOMP loading and temporary-size queries

This probe loads prebuilt cuDF first, then Vortex's actual newly built libraries.
It checks only loading and API calls, not kernel execution or decompression.
The artifact hashes below identify the tested build; locate the corresponding
`out` directories if rebuilding changes them. Run from the Vortex root:

```sh
python3 - <<'PY'
import ctypes as c
from pathlib import Path

root = Path("build/cudf-vortex-smoke/build/_deps/vortex-build/ffi/cargo-target/")
root /= "aarch64-unknown-linux-gnu/release/build"
cudf = c.CDLL("/home/ubuntu/cudf/cpp/build/libcudf.so", mode=c.RTLD_GLOBAL)
cub = c.CDLL(str((root / "vortex-cub-62fc87f0583d0b9e/out/libvortex_cub.so").resolve()))
cub.filter_temp_size_u64.argtypes = [c.POINTER(c.c_size_t), c.c_int64]
cub.filter_temp_size_u64.restype = c.c_int
size = c.c_size_t()
status = cub.filter_temp_size_u64(c.byref(size), 1000)
print("CUB status/bytes:", status, size.value)
assert status == 0 and size.value > 0

nvcomp = c.CDLL(str((root / "vortex-nvcomp-7c4911d222f928c3/out/nvcomp-sdk/lib/libnvcomp.so").resolve()))
class ZstdOpts(c.Structure):
    _fields_ = [("backend", c.c_int), ("reserved", c.c_uint8 * 60)]
fn = nvcomp.nvcompBatchedZstdDecompressGetTempSizeAsync
fn.argtypes = [c.c_size_t, c.c_size_t, ZstdOpts, c.POINTER(c.c_size_t), c.c_size_t]
fn.restype = c.c_int
status = fn(10, 65536, ZstdOpts(), c.byref(size), 655360)
print("Vortex nvCOMP status/bytes:", status, size.value)
assert status == 0 and size.value > 0

print("\n".join(sorted({
    line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
    if "libnvcomp" in line or "libvortex_cub" in line
})))
PY
```

Results: CUB returned status `0` / `1023` bytes; Vortex nvCOMP returned status `0`
/ `2035440` bytes. The process mappings confirmed both nvCOMP **5.1.0.21** (Vortex)
and **5.2.0.10** (prebuilt cuDF) loaded. The newly configured upstream build uses
nvCOMP **5.3.0.16**, whose same-process behavior remains to be tested.
