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

On Linux, use `vx_cuda_scan_path_arrow_device_stream_with_options` with
`vx_cuda_scan_options.flags = VX_CUDA_SCAN_FLAG_DIRECT_IO` to bypass the operating system page
cache for pooled data-plane reads. Footer and zone-map reads remain buffered on the host.

## Generated header

`build.rs` generates `cinclude/vortex_cuda.h` using cbindgen on stable Rust. Edit `src/lib.rs`
or `cbindgen.toml`, not the header, and commit regenerated output. The header keeps cbindgen's
formatting and is excluded from clang-format through generated markers.

Unchanged output is not rewritten; changed output is published atomically. CUDA CI checks
for header drift after building the FFI crate.
