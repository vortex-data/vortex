// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#ifndef CUDF_ARROW_FFI_H
#define CUDF_ARROW_FFI_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Arrow C Device Data Interface structures
// These match the Arrow specification for device data exchange

struct ArrowSchema {
    const char* format;
    const char* name;
    const char* metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema** children;
    struct ArrowSchema* dictionary;
    void (*release)(struct ArrowSchema*);
    void* private_data;
};

struct ArrowArray {
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void** buffers;
    struct ArrowArray** children;
    struct ArrowArray* dictionary;
    void (*release)(struct ArrowArray*);
    void* private_data;
};

// Arrow Device type constants
typedef int32_t ArrowDeviceType;
#define ARROW_DEVICE_CPU 1
#define ARROW_DEVICE_CUDA 2
#define ARROW_DEVICE_CUDA_HOST 3
#define ARROW_DEVICE_OPENCL 4
#define ARROW_DEVICE_VULKAN 7
#define ARROW_DEVICE_METAL 8
#define ARROW_DEVICE_VPI 9
#define ARROW_DEVICE_ROCM 10
#define ARROW_DEVICE_ROCM_HOST 11
#define ARROW_DEVICE_EXT_DEV 12
#define ARROW_DEVICE_CUDA_MANAGED 13
#define ARROW_DEVICE_ONEAPI 14
#define ARROW_DEVICE_WEBGPU 15
#define ARROW_DEVICE_HEXAGON 16

struct ArrowDeviceArray {
    struct ArrowArray array;
    int64_t device_id;
    ArrowDeviceType device_type;
    void* sync_event;
};

// Error type: NULL on success, pointer to error string on failure.
// Caller must free with cudf_err_free() when non-NULL.
typedef const char* cudf_err_t;

// Opaque context type that holds CUDA memory resources and global state.
typedef struct cudf_context cudf_context_t;

// Opaque table view type wrapping cudf::unique_table_view_t
typedef struct cudf_tableview cudf_tableview_t;

// Opaque column view type wrapping cudf::unique_column_view_t
typedef struct cudf_columnview cudf_columnview_t;

// Create a new cudf context and initialize RMM.
// On success, *ctx is set to the new context and NULL is returned.
// On failure, *ctx is unchanged and an error string is returned.
cudf_err_t cudf_context_create(cudf_context_t** ctx);

// Free a cudf context and all associated resources.
void cudf_context_free(cudf_context_t* ctx);

// Import an Arrow table from device memory into a cudf table view.
// On success, *out is set to the new table view and NULL is returned.
// On failure, *out is unchanged and an error string is returned.
cudf_err_t cudf_tableview_from_device(
    cudf_context_t* ctx,
    const struct ArrowSchema* schema,
    const struct ArrowDeviceArray* device_array,
    cudf_tableview_t** out
);

// Import an Arrow column from device memory into a cudf column view.
// On success, *out is set to the new column view and NULL is returned.
// On failure, *out is unchanged and an error string is returned.
cudf_err_t cudf_columnview_from_device(
    cudf_context_t* ctx,
    const struct ArrowSchema* schema,
    const struct ArrowDeviceArray* device_array,
    cudf_columnview_t** out
);

// Get the number of rows in a table view.
cudf_err_t cudf_tableview_num_rows(const cudf_tableview_t* tv, int64_t* count);

// Get the number of columns in a table view.
cudf_err_t cudf_tableview_num_columns(const cudf_tableview_t* tv, int32_t* count);

// Get the number of rows in a column view.
cudf_err_t cudf_columnview_size(const cudf_columnview_t* cv, int64_t* count);

// Count valid (non-null) values in a table column.
cudf_err_t cudf_tableview_count_valid(const cudf_tableview_t* tv, int32_t column_index, int64_t* valid_count);

// Count valid (non-null) values in a column view.
cudf_err_t cudf_columnview_count_valid(const cudf_columnview_t* cv, int64_t* valid_count);

// Sum values in an int64 table column.
cudf_err_t cudf_tableview_sum_int64(const cudf_tableview_t* tv, int32_t column_index, int64_t* sum);

// Sum values in an int64 column view.
cudf_err_t cudf_columnview_sum_int64(const cudf_columnview_t* cv, int64_t* sum);

// Free a table view.
void cudf_tableview_free(cudf_tableview_t* tv);

// Free a column view.
void cudf_columnview_free(cudf_columnview_t* cv);

// Free an error string.
void cudf_err_free(cudf_err_t err);

#ifdef __cplusplus
}
#endif

#endif // CUDF_ARROW_FFI_H
