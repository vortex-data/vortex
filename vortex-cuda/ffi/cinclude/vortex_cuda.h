// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include <stdint.h>

#include "vortex.h"

/* Link against the CUDA-enabled FFI library that provides both the base Vortex FFI and these CUDA
 * entry points. Do not pass Vortex handles between independently linked Rust FFI libraries. */

#ifdef __cplusplus
extern "C" {
#endif

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

/**
 * Create a CUDA Vortex session.
 *
 * Repeated `vx_cuda_array_export_arrow_device` calls reuse this CUDA state. Returns an owned
 * session handle, or NULL and an optional `vx_error` on failure.
 */
vx_session *vx_cuda_session_new(vx_error **error_out);

/**
 * Export a borrowed Vortex array for cuDF's Arrow Device import path.
 *
 * On success returns 0 and writes independently releasable `out_schema` and `out_array`; the caller
 * passes them to cuDF and releases both via their embedded Arrow callbacks after import. On error
 * returns 1 and, when `error_out` is non-NULL, writes a `vx_error` (free with `vx_error_free`).
 *
 * `out_array` is exported on `ARROW_DEVICE_CUDA`; struct arrays become table-shaped schemas,
 * non-struct arrays a single column field.
 *
 * Export is stream-ordered; `out_array->sync_event` is valid until `out_array` is released.
 */
int vx_cuda_array_export_arrow_device(const vx_session *session,
                                      const vx_array *array,
                                      FFI_ArrowSchema *out_schema,
                                      struct ArrowDeviceArray *out_array,
                                      vx_error **error_out);

#ifdef __cplusplus
}
#endif
