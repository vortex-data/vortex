// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// clang-format off

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
 * Dictionary-preserving CUDA scan with an owned Arrow schema per batch.
 *
 * Logical dtype and CUDA device are fixed; physical schemas, dictionary values, and index widths
 * may vary. Uses `DictionaryExport::Preserve`; plain batches remain plain.
 * Retains runtime/session resources. Use and free only on the creating thread, without concurrent
 * or reentrant calls. Free with `vx_cuda_device_batches_free`.
 */
typedef struct vx_cuda_device_batches vx_cuda_device_batches;

/**
 * Options for scanning a CUDA-compatible Vortex file.
 *
 * Zero-initialize this struct to use buffered file I/O and layout-derived batch splitting.
 */
typedef struct vx_cuda_scan_options {
  /**
   * A bitwise combination of `VX_CUDA_SCAN_FLAG_*` values. Unknown bits are rejected.
   */
  uint32_t flags;
  /**
   * Rows per batch, except for a possibly smaller final batch. Zero preserves layout boundaries.
   * Nonzero counts ignore layout boundaries and may require unsupported CUDA `Chunked`
   * concatenation.
   */
  size_t batch_rows;
} vx_cuda_scan_options;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Release Vortex's cached file mappings that no open scan uses.
 *
 * Scans keep their files mapped so later scans of the same file reuse filled page tables.
 * On Linux, both `posix_fadvise(POSIX_FADV_DONTNEED)` and `/proc/sys/vm/drop_caches`
 * skip pages mapped into a process, although memory pressure can still reclaim them.
 * Call this before using either to evict file pages, for example before a cold-cache
 * measurement. Mappings still used by open streams stay mapped until those streams
 * are released.
 */
void vx_cuda_release_file_mappings(void);

/**
 * Create a CUDA Vortex session.
 *
 * Returns an owned handle with reusable CUDA state, or null and an optional `vx_error` on failure.
 *
 * # Safety
 *
 * If `error_out` is non-null, it must be valid for writing one error pointer.
 */
vx_session *vx_cuda_session_new(vx_error **error_out);

/**
 * Borrow the session's existing CUDA memory pool as a `CUmemoryPool` / `cudaMemPool_t`.
 *
 * Hosts may allocate from this pool to share Vortex's reserve of device memory. Null means
 * Vortex is using the device allocator fallback. Errors also return null and set `error_out`;
 * a successful call clears `error_out`.
 *
 * # Safety
 *
 * `session` must be a live borrowed `vortex-ffi` handle. Keep it alive until all host allocations
 * from the pool have been freed and their GPU work, including asynchronous frees, has completed.
 * The returned pool is borrowed and must not be destroyed by the caller. Non-null `error_out`
 * must be writable for one error pointer.
 */
void *vx_cuda_session_mem_pool(const vx_session *session, vx_error **error_out);

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
 * Zero `block_rows` selects the GPU default of 8,388,608 rows. Blocks are not byte-size coalesced
 * and have no outer layout dictionaries, but retain per-block dictionary compression.
 * Write sizing is independent of scan `batch_rows`; see `vx_cuda_scan_options`.
 *
 * # Safety
 *
 * Same requirements as `vx_cuda_array_sink_open_file`.
 */
vx_array_sink *vx_cuda_array_sink_open_file_block_rows(const vx_session *session,
                                                       vx_view path,
                                                       const vx_dtype *dtype,
                                                       size_t block_rows,
                                                       vx_error **error_out);

/**
 * Scan a local CUDA-readable file with buffered I/O into an Arrow C Device stream.
 * Dictionaries export as plain values. Returns `0` on success, `1` on error.
 * Release the stream and batches via Arrow callbacks; free errors with `vx_error_free`.
 *
 * # Safety
 *
 * `session` must be a live borrowed `vortex-ffi` handle; `path` must contain readable UTF-8
 * for this call. `out_stream` and non-null `error_out` must point to writable output storage.
 */
int vx_cuda_scan_path_arrow_device_stream(const vx_session *session,
                                          vx_view path,
                                          struct ArrowDeviceArrayStream *out_stream,
                                          vx_error **error_out);

/**
 * Like `vx_cuda_scan_path_arrow_device_stream`, with `batch_rows` as in `vx_cuda_scan_options`.
 *
 * # Safety
 *
 * Same requirements as `vx_cuda_scan_path_arrow_device_stream`.
 */
int vx_cuda_scan_path_arrow_device_stream_batch_rows(const vx_session *session,
                                                     vx_view path,
                                                     size_t batch_rows,
                                                     struct ArrowDeviceArrayStream *out_stream,
                                                     vx_error **error_out);

/**
 * Like `vx_cuda_scan_path_arrow_device_stream`, with explicit scan options.
 *
 * Null or zero-initialized `options` selects buffered I/O and layout-derived batch splitting.
 *
 * # Safety
 *
 * Same requirements as `vx_cuda_scan_path_arrow_device_stream`; non-null `options` must
 * point to a valid `vx_cuda_scan_options`.
 */
int vx_cuda_scan_path_arrow_device_stream_with_options(const vx_session *session,
                                                       vx_view path,
                                                       const struct vx_cuda_scan_options *options,
                                                       struct ArrowDeviceArrayStream *out_stream,
                                                       vx_error **error_out);

/**
 * Scan a local Vortex file with ordered top-level column projection.
 *
 * Otherwise follows `vx_cuda_scan_path_arrow_device_stream_with_options`.
 * Names are literal, case-sensitive, and unique; nonempty projections require a struct file.
 * Zero `ncolumns` ignores `columns` and selects all. Unknown names fail.
 * Skips unselected column I/O where the layout allows. Errors leave `out_stream` unchanged.
 *
 * # Safety
 *
 * Same requirements as `vx_cuda_scan_path_arrow_device_stream_with_options`.
 * For nonzero `ncolumns`, `columns` must reference that many valid, aligned `vx_view` values.
 * Name bytes must be readable UTF-8 for this call; null is allowed only for zero length.
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
 * Scan a Vortex partition as an Arrow C Device stream.
 *
 * Consumes `partition`, even on error. Return codes and output ownership follow
 * `vx_cuda_scan_path_arrow_device_stream`.
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

/**
 * Open a local CUDA scan that preserves per-batch dictionaries.
 *
 * Returns an owned handle, or null on error. Options, projection, and splitting match
 * `vx_cuda_scan_path_arrow_device_stream_projected`: null options use defaults; zero `ncolumns`
 * ignores `columns` and selects all. Names must be unique, literal, case-sensitive UTF-8;
 * unknown names fail, and nonempty projections require a struct file.
 *
 * Inputs are borrowed for this call; the handle retains runtime/session resources.
 * Non-null `error_out` is cleared on success or receives an error to free with `vx_error_free`.
 *
 * # Safety
 *
 * Borrow a live `session` and readable UTF-8 `path`. Non-null `options` must be aligned and
 * initialized. For nonzero `ncolumns`, `columns` must reference that many valid, aligned views
 * with readable name bytes (null only for zero length). Non-null `error_out` must be writable.
 */
struct vx_cuda_device_batches *vx_cuda_scan_path_device_batches(const vx_session *session,
                                                                vx_view path,
                                                                const struct vx_cuda_scan_options *options,
                                                                const vx_view *columns,
                                                                size_t ncolumns,
                                                                vx_error **error_out);

/**
 * Return the next owned Arrow schema and CUDA array (`0` on success, `1` on error).
 *
 * Use each batch's own schema. EOS (including repeated calls) sets both release callbacks to null;
 * live zero-row batches are not EOS. Errors leave outputs unchanged. Null/unaligned handle or
 * output pointers are rejected before polling; scan/export failures, dtype/device mismatches,
 * and panics make the handle terminal (later calls fail). Free it after a terminal error.
 * Non-null `error_out` is cleared on success or receives an error to free with `vx_error_free`.
 *
 * Outputs outlive the handle/session. Release each independently, exactly once, via its Arrow
 * callback. Wait on `out_array->sync_event` before consuming buffers; retain the array until GPU
 * consumers finish. The event/buffers live until array release; output storage is caller-owned.
 *
 * # Safety
 *
 * Use a live handle exclusively on its creating thread, without reentrancy. Non-null, aligned
 * output pointers must reference writable storage, disjoint from each other, the handle, and
 * `error_out`. Outputs may be uninitialized; release previous contents before overwriting them.
 * Non-null `error_out` must be writable for one error pointer.
 */
int vx_cuda_device_batches_next(struct vx_cuda_device_batches *handle,
                                FFI_ArrowSchema *out_schema,
                                struct ArrowDeviceArray *out_array,
                                vx_error **error_out);

/**
 * Free the handle; null is a no-op.
 *
 * Returned outputs remain caller-owned and must be released via their Arrow callbacks.
 *
 * # Safety
 *
 * Free a live, exclusively owned handle exactly once on its creating thread, without reentrancy.
 */
void vx_cuda_device_batches_free(struct vx_cuda_device_batches *handle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

// clang-format on
