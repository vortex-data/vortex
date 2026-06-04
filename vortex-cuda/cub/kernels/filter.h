// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// C header for CUB filter functions.
// Used by bindgen to generate Rust FFI bindings with dynamic library loading.

#pragma once

#include <stddef.h>
#include <stdint.h>

// i256 type
typedef struct {
    __int128_t high;
    __int128_t low;
} __int256_t;

// CUDA types - defined as opaque for bindgen
typedef int cudaError_t;
typedef void *cudaStream_t;

#ifdef __cplusplus
extern "C" {
#endif

// X-macro table: (suffix, c_type)
#define FILTER_TYPE_TABLE(X)                                                                                 \
    X(u8, uint8_t)                                                                                           \
    X(i8, int8_t)                                                                                            \
    X(u16, uint16_t)                                                                                         \
    X(i16, int16_t)                                                                                          \
    X(u32, uint32_t)                                                                                         \
    X(i32, int32_t)                                                                                          \
    X(u64, uint64_t)                                                                                         \
    X(i64, int64_t)                                                                                          \
    X(f32, float)                                                                                            \
    X(f64, double)                                                                                           \
    X(i128, __int128_t)                                                                                      \
    X(i256, __int256_t)

// Filter temp size query functions
#define DECLARE_FILTER_TEMP_SIZE(suffix, c_type)                                                             \
    cudaError_t filter_temp_size_##suffix(size_t *temp_bytes, int64_t num_items);

FILTER_TYPE_TABLE(DECLARE_FILTER_TEMP_SIZE)

#undef DECLARE_FILTER_TEMP_SIZE

// Filter execution functions (byte mask - one byte per element)
#define DECLARE_FILTER_BYTEMASK(suffix, c_type)                                                              \
    cudaError_t filter_bytemask_##suffix(void *d_temp,                                                       \
                                         size_t temp_bytes,                                                  \
                                         const c_type *d_in,                                                 \
                                         const uint8_t *d_flags,                                             \
                                         c_type *d_out,                                                      \
                                         int64_t *d_num_selected,                                            \
                                         int64_t num_items,                                                  \
                                         cudaStream_t stream);

FILTER_TYPE_TABLE(DECLARE_FILTER_BYTEMASK)

#undef DECLARE_FILTER_BYTEMASK

// Filter execution functions (bit mask - one bit per element)
//
// These functions accept packed bit mask directly, avoiding the need to
// expand bits to bytes in a separate kernel. Uses CUB's TransformInputIterator
// to read bits on-the-fly during the filter operation.
#define DECLARE_FILTER_BITMASK(suffix, c_type)                                                               \
    cudaError_t filter_bitmask_##suffix(void *d_temp,                                                        \
                                        size_t temp_bytes,                                                   \
                                        const c_type *d_in,                                                  \
                                        const uint8_t *d_bitmask,                                            \
                                        uint64_t bit_offset,                                                 \
                                        c_type *d_out,                                                       \
                                        int64_t *d_num_selected,                                             \
                                        int64_t num_items,                                                   \
                                        cudaStream_t stream);

FILTER_TYPE_TABLE(DECLARE_FILTER_BITMASK)

#undef DECLARE_FILTER_BITMASK

cudaError_t scan_exclusive_sum_i32_temp_size(size_t *temp_bytes, int64_t num_items);

cudaError_t scan_exclusive_sum_i32(void *d_temp,
                                   size_t temp_bytes,
                                   const int32_t *d_in,
                                   int32_t *d_out,
                                   int64_t num_items,
                                   cudaStream_t stream);

cudaError_t scan_exclusive_sum_i64_temp_size(size_t *temp_bytes, int64_t num_items);

cudaError_t scan_exclusive_sum_i64(void *d_temp,
                                   size_t temp_bytes,
                                   const int64_t *d_in,
                                   int64_t *d_out,
                                   int64_t num_items,
                                   cudaStream_t stream);

#ifdef __cplusplus
}
#endif
