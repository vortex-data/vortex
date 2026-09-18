// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

// THIS FILE IS AUTO-GENERATED, DO NOT MAKE EDITS DIRECTLY

#include <stddef.h>
#include <stdint.h>

#include "vortex.h"

/* Link against the CUDA-enabled FFI library that provides both the base Vortex FFI and these CUDA
 * entry points. Do not pass Vortex handles between independently linked Rust FFI libraries. */

/* Definitions from the Arrow C Device data interface. Define USE_OWN_ARROW_DEVICE to skip them.
 * https://arrow.apache.org/docs/format/CDeviceDataInterface.html */
#if !defined(ARROW_C_DEVICE_DATA_INTERFACE) && !defined(USE_OWN_ARROW_DEVICE)
#define ARROW_C_DEVICE_DATA_INTERFACE

typedef int32_t ArrowDeviceType;
#define ARROW_DEVICE_CPU          1
#define ARROW_DEVICE_CUDA         2
#define ARROW_DEVICE_CUDA_HOST    3
#define ARROW_DEVICE_OPENCL       4
#define ARROW_DEVICE_VULKAN       7
#define ARROW_DEVICE_METAL        8
#define ARROW_DEVICE_VPI          9
#define ARROW_DEVICE_ROCM         10
#define ARROW_DEVICE_ROCM_HOST    11
#define ARROW_DEVICE_EXT_DEV      12
#define ARROW_DEVICE_CUDA_MANAGED 13
#define ARROW_DEVICE_ONEAPI       14
#define ARROW_DEVICE_WEBGPU       15
#define ARROW_DEVICE_HEXAGON      16

struct ArrowDeviceArray {
    struct ArrowArray array;
    int64_t device_id;
    ArrowDeviceType device_type;
    void *sync_event;
    int64_t reserved[3];
};
#endif

#if !defined(ARROW_C_DEVICE_STREAM_INTERFACE) && !defined(USE_OWN_ARROW_DEVICE)
#define ARROW_C_DEVICE_STREAM_INTERFACE
struct ArrowDeviceArrayStream {
    ArrowDeviceType device_type;
    int (*get_schema)(struct ArrowDeviceArrayStream *, struct ArrowSchema *out);
    int (*get_next)(struct ArrowDeviceArrayStream *, struct ArrowDeviceArray *out);
    const char *(*get_last_error)(struct ArrowDeviceArrayStream *);
    void (*release)(struct ArrowDeviceArrayStream *);
    void *private_data;
};
#endif

/**
 * Bypass the operating system page cache for pooled data-plane reads.
 * Footer and zone-map reads remain buffered. Supported only on Linux.
 */
#define VX_CUDA_SCAN_FLAG_DIRECT_IO (1u << 0)

/**
 * Options for scanning a CUDA-compatible Vortex file.
 *
 * Zero-initialize this struct to use buffered file I/O and layout-derived batch splitting.
 */
typedef struct vx_cuda_scan_options {
    /**
     * A bitwise combination of `VX_CUDA_SCAN_FLAG_*` values. Unknown bits are ignored.
     */
    uint32_t flags;
    /**
     * Maximum rows in each output batch. Zero uses layout-derived splitting.
     * Physical layout boundaries may produce shorter batches.
     */
    size_t batch_rows;
} vx_cuda_scan_options;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Create a CUDA Vortex session.
 *
 * Repeated [`vx_cuda_array_export_arrow_device`] calls reuse this CUDA state. Returns an owned
 * session handle, or null and an optional `vx_error` on failure.
 *
 * # Safety
 *
 * If `error_out` is non-null, it must be valid for writing one error pointer.
 */
vx_session *vx_cuda_session_new(vx_error **error_out);

/**
 * Open a Vortex file sink configured to produce CUDA-readable files.
 *
 * Push host arrays and close/abort with `vx_array_sink_*`. Only on-disk encodings and layouts
 * change; writing does not move arrays to the GPU.
 *
 * # Safety
 *
 * `session`, `path`, and `dtype` follow `vx_array_sink_open_file`'s requirements.
 * Non-null `error_out` must be writable for one error pointer.
 */
vx_array_sink *vx_cuda_array_sink_open_file(const vx_session *session,
                                            vx_view path,
                                            const vx_dtype *dtype,
                                            vx_error **error_out);

/**
 * Open a CUDA-readable Vortex file sink with a fixed row block size.
 *
 * `block_rows` controls the row granularity of CUDA-flat data blocks. Passing zero uses the default
 * writer strategy: 8,192-row blocks may be coalesced into data blocks targeting 1 MiB. Any nonzero
 * value disables byte-size coalescing and outer layout dictionaries, so passing 8,192 is not
 * equivalent to passing zero.
 *
 * Write and scan sizing are independent; scan batches preserve on-disk layout boundaries.
 *
 * # Safety
 *
 * Same requirements as [`vx_cuda_array_sink_open_file`].
 */
vx_array_sink *vx_cuda_array_sink_open_file_block_rows(const vx_session *session,
                                                       vx_view path,
                                                       const vx_dtype *dtype,
                                                       size_t block_rows,
                                                       vx_error **error_out);

/**
 * Scan a local Vortex file with buffered I/O and export an Arrow C Device stream.
 *
 * Requires CUDA-supported encodings/layouts, such as files from [`vx_cuda_array_sink_open_file`].
 * Footer/zone-map reads stay on the host; data reaches the GPU through pinned staging buffers
 * reused across scans with the same CUDA session.
 *
 * Dictionaries, including nested children, decode on CUDA for a stable plain Arrow schema,
 * without changing session policy. Decoding can increase device memory use and requires CUDA
 * support for device-resident dictionaries.
 *
 * Returns `0` with an owned `out_stream`; release it and each batch via their Arrow callbacks.
 * Returns `1` on error, writing a `vx_error` if `error_out` is non-null; free it with `vx_error_free`.
 *
 * # Safety
 *
 * `session` must be a valid borrowed handle created by `vortex-ffi`. `path` must be valid for the
 * duration of this call and contain UTF-8. `out_stream` must be a valid writable pointer. If
 * `error_out` is non-null, it must be valid for writing one error pointer.
 */
int vx_cuda_scan_path_arrow_device_stream(const vx_session *session,
                                          vx_view path,
                                          struct ArrowDeviceArrayStream *out_stream,
                                          vx_error **error_out);

/**
 * Scan a local Vortex file with bounded row batches.
 *
 * Uses [`vx_cuda_scan_path_arrow_device_stream`]'s export and ownership rules.
 * `batch_rows` caps output rows; zero uses layout splitting. Physical boundaries may shorten
 * batches. Scan and write sizing are independent; scans preserve on-disk layout boundaries.
 *
 * # Safety
 *
 * Same requirements as [`vx_cuda_scan_path_arrow_device_stream`].
 */
int vx_cuda_scan_path_arrow_device_stream_batch_rows(const vx_session *session,
                                                     vx_view path,
                                                     size_t batch_rows,
                                                     struct ArrowDeviceArrayStream *out_stream,
                                                     vx_error **error_out);

/**
 * Like [`vx_cuda_scan_path_arrow_device_stream`], with explicit scan options.
 *
 * Null or zero-initialized `options` selects buffered I/O and layout-derived batch splitting.
 *
 * # Safety
 *
 * Same requirements as [`vx_cuda_scan_path_arrow_device_stream`]; non-null `options` must
 * point to a valid [`vx_cuda_scan_options`].
 */
int vx_cuda_scan_path_arrow_device_stream_with_options(const vx_session *session,
                                                       vx_view path,
                                                       const struct vx_cuda_scan_options *options,
                                                       struct ArrowDeviceArrayStream *out_stream,
                                                       vx_error **error_out);

/**
 * Scan a local Vortex file with ordered top-level column projection.
 *
 * Same options, ownership, and file requirements as
 * [`vx_cuda_scan_path_arrow_device_stream_with_options`]. Projection precedes column I/O/decoding.
 * Names are literal and case-sensitive; unknown/duplicate names and non-struct files are rejected.
 * `ncolumns == 0` ignores `columns` and selects all. Names are copied; empty files retain the
 * projected schema. Errors leave `out_stream` unchanged.
 *
 * # Safety
 *
 * In addition to [`vx_cuda_scan_path_arrow_device_stream_with_options`]'s requirements,
 * nonzero `ncolumns` requires that many initialized, aligned [`vx_view`] values at `columns`.
 * Each name borrows `len` readable UTF-8 bytes for this call; null is allowed only for zero length.
 */
int vx_cuda_scan_path_arrow_device_stream_projected(const vx_session *session,
                                                    vx_view path,
                                                    const struct vx_cuda_scan_options *options,
                                                    const vx_view *columns,
                                                    size_t ncolumns,
                                                    struct ArrowDeviceArrayStream *out_stream,
                                                    vx_error **error_out);

/**
 * Export a borrowed Vortex array for cuDF's Arrow Device import path.
 *
 * Returns `0` with independently owned `out_schema` and `out_array`. Pass them to cuDF, then
 * release both via their Arrow callbacks after import. Returns `1` on error, writing a `vx_error`
 * if `error_out` is non-null; free it with `vx_error_free`.
 *
 * `out_array` is exported on `ARROW_DEVICE_CUDA`; struct arrays become table-shaped schemas,
 * non-struct arrays a single column field.
 *
 * Export is stream-ordered; `out_array->sync_event` is valid until `out_array` is released.
 *
 * # Safety
 *
 * `session` and `array` must be valid borrowed handles created by `vortex-ffi`. `out_schema`
 * and `out_array` must be valid writable pointers. If `error_out` is non-null, it must be valid
 * for writing one error pointer.
 */
int vx_cuda_array_export_arrow_device(const vx_session *session,
                                      const vx_array *array,
                                      FFI_ArrowSchema *out_schema,
                                      struct ArrowDeviceArray *out_array,
                                      vx_error **error_out);

/**
 * Consume a Vortex partition and scan it as an Arrow C Device stream.
 *
 * Consumes `partition` on success or failure; never free or reuse it afterward.
 * Returns `0` with an owned `out_stream` retaining the scan iterator. Release the stream and
 * each produced batch via their Arrow release callbacks.
 * Returns `1` on error, writing a `vx_error` if `error_out` is non-null; free it with `vx_error_free`.
 *
 * # Safety
 *
 * `session` must be a valid borrowed handle created by `vortex-ffi`. `partition` must be an owned
 * partition handle created by `vortex-ffi`. `out_stream` must be a valid writable pointer. If
 * `error_out` is non-null, it must be valid for writing one error pointer.
 */
int vx_cuda_partition_scan_arrow_device_stream(const vx_session *session,
                                               vx_partition *partition,
                                               struct ArrowDeviceArrayStream *out_stream,
                                               vx_error **error_out);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
