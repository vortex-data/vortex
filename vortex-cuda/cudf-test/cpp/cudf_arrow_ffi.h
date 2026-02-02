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

// Error codes for cudf operations
typedef enum {
    CUDF_SUCCESS = 0,
    CUDF_ERROR_INIT_FAILED = 1,
    CUDF_ERROR_INVALID_ARGUMENT = 2,
    CUDF_ERROR_LOAD_FAILED = 3,
    CUDF_ERROR_NO_DATA = 4,
    CUDF_ERROR_OPERATION_FAILED = 5,
} CudfErrorCode;

// Result type for cudf operations
typedef struct {
    CudfErrorCode code;
    const char* error_message;  // NULL on success, caller must free with cudf_free_error
} CudfResult;

// Initialize cudf/RMM runtime
CudfResult cudf_init(void);

// Load Arrow data from device memory into cudf
// Takes a table (struct of arrays)
CudfResult cudf_load_from_arrow_device(
    const struct ArrowSchema* schema,
    const struct ArrowDeviceArray* device_array
);

// Load a single Arrow column from device memory into cudf
CudfResult cudf_load_column_from_arrow_device(
    const struct ArrowSchema* schema,
    const struct ArrowDeviceArray* device_array
);

// Get the number of rows in the loaded table
CudfResult cudf_get_row_count(int64_t* count);

// Get the number of columns in the loaded table
CudfResult cudf_get_column_count(int32_t* count);

// Count valid (non-null) values in a column
CudfResult cudf_count_valid(int32_t column_index, int64_t* valid_count);

// Sum values in an int64 column
CudfResult cudf_sum_int64(int32_t column_index, int64_t* sum);

// Free the loaded table
CudfResult cudf_free_table(void);

// Free an error message returned by a CudfResult
void cudf_free_error(const char* error_msg);

#ifdef __cplusplus
}
#endif

#endif // CUDF_ARROW_FFI_H
