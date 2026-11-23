# vortex-cuda-ffi

CUDA-specific C FFI helpers for cuDF interop.

This crate keeps CUDA out of the base `vortex-ffi` crate. Its public C API exports a borrowed `vx_array` as an `ArrowSchema + ArrowDeviceArray` pair.

The Rust FFI does not create cuDF objects itself. The caller imports the exported Arrow Device
structs into cuDF and retains the device arrays until all GPU work using their buffers completes.

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

Pooled data reads use fixed 8 MiB chunks and concurrency 32;
scan row ranges and batch sizes are unchanged.

On Linux, use `vx_cuda_scan_path_arrow_device_stream_with_options` with
`vx_cuda_scan_options.flags = VX_CUDA_SCAN_FLAG_DIRECT_IO` to bypass the operating system page
cache for pooled data-plane reads. Footer and zone-map reads remain buffered on the host.

## Dictionary-preserving batches

Use `vx_cuda_scan_path_device_batches` to preserve dictionaries with a schema per batch.
Physical encodings and dictionary index widths may differ between batches; import each batch
with its own schema. The existing fixed-schema stream API continues to decode dictionaries.
See `vortex_cuda.h` for ownership, synchronization, and error contracts.

## Late string expansion with cuDF

The optional C++ header `vortex_cudf_dictionary.hpp` combines string dictionary batches before
expanding them into owned cuDF output. It accepts differing dictionary keys and uses a numeric
payload fast path for non-null strings with uniform 1/2/4/8-byte widths. Compression and file
size are unchanged; the Rust FFI does not depend on cuDF.

Callers handle unsupported types separately and retain input buffers until GPU work completes.
See the header for contracts and `tests/` for standalone C++ tests targeting libcudf 26.06.

## Generated header

`build.rs` generates `cinclude/vortex_cuda.h` with cbindgen. Edit the Rust declarations in `src/`
or `cbindgen.toml` and commit the regenerated header. It skips clang-format; CUDA CI checks for
header drift.
