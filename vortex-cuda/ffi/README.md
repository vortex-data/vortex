# vortex-cuda-ffi

CUDA-specific C FFI helpers for cuDF interop.

This crate keeps CUDA out of the base `vortex-ffi` crate. Its public C API exports a borrowed `vx_array` as an `ArrowSchema + ArrowDeviceArray` pair.

It does not create cuDF objects itself. The caller passes the exported Arrow Device structs to cuDF and releases them after cuDF is done importing.

Use this crate as the CUDA-enabled FFI artifact. Include both headers:

```c
#include "vortex.h"
#include "vortex_cuda.h"
```

and link the CUDA FFI library (`vortex_cuda_ffi`). Do not pass Vortex handles between independently linked Rust FFI libraries.

Use `vx_cuda_session_new` to initialize CUDA once and reuse it across exports.

Use `vx_cuda_array_sink_open_file` to open a standard Vortex file sink configured to produce CUDA-readable files.
Push host-resident arrays and close the sink using the standard `vx_array_sink_*` APIs.

Use `vx_cuda_scan_path_arrow_device_stream` to read such a local file through pinned host buffers
and receive an Arrow C Device stream. Reuse the same CUDA session across scans so the pinned buffer
pool and CUDA state are reused as well.
