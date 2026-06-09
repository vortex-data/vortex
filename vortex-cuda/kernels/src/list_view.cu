// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "config.cuh"

#include <stdint.h>
#include <type_traits>

namespace {

template <typename T>
__device__ bool non_negative_to_u64(T value, uint64_t *out) {
    if constexpr (std::is_signed_v<T>) {
        if (value < 0) {
            return false;
        }
    }

    *out = static_cast<uint64_t>(value);
    return true;
}

__device__ bool checked_add_u64(uint64_t lhs, uint64_t rhs, uint64_t *out) {
    if (rhs > UINT64_MAX - lhs) {
        return false;
    }

    *out = lhs + rhs;
    return true;
}

// Assumes `ListViewArray` construction invariants for basic metadata validity. This kernel only
// decides whether the views are already contiguous Arrow `List` offsets and fit cuDF's i32 limit.
template <typename OffsetT, typename SizeT>
__device__ void list_view_offsets_device(const OffsetT *const offsets,
                                         const SizeT *const sizes,
                                         int32_t *const output,
                                         uint32_t *const status,
                                         uint64_t list_len) {
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t startElem = start_elem(worker, list_len);
    const uint64_t stopElem = stop_elem(worker, list_len);

    for (uint64_t idx = startElem; idx < stopElem; idx++) {
        const uint64_t offset = static_cast<uint64_t>(offsets[idx]);
        const uint64_t end = offset + static_cast<uint64_t>(sizes[idx]);
        output[idx] = static_cast<int32_t>(offset);

        if (end < offset || end > static_cast<uint64_t>(INT32_MAX)) {
            atomicMax(status, 2u);
        }
        if (idx == 0 && offset != 0) {
            atomicMax(status, 1u);
        }

        if (idx + 1 == list_len) {
            output[list_len] = static_cast<int32_t>(end);
        } else if (static_cast<uint64_t>(offsets[idx + 1]) != end) {
            atomicMax(status, 1u);
        }
    }
}

template <typename SizeT>
__device__ void list_view_rebuild_init_scan_device(const SizeT *const sizes,
                                                   int32_t *const scan,
                                                   uint32_t *const status,
                                                   uint64_t list_len,
                                                   uint64_t scan_len) {
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t startElem = start_elem(worker, scan_len);
    const uint64_t stopElem = stop_elem(worker, scan_len);

    for (uint64_t idx = startElem; idx < stopElem; idx++) {
        if (idx >= list_len) {
            scan[idx] = 0;
            continue;
        }

        uint64_t size = 0;
        if (!non_negative_to_u64(sizes[idx], &size)) {
            atomicMax(status, 1u);
            scan[idx] = 0;
        } else if (size > static_cast<uint64_t>(INT32_MAX)) {
            atomicMax(status, 2u);
            scan[idx] = 0;
        } else {
            scan[idx] = static_cast<int32_t>(size);
        }
    }
}

template <typename SizeT>
__device__ void list_view_rebuild_validate_offsets_device(const SizeT *const sizes,
                                                          const int32_t *const output_offsets,
                                                          uint32_t *const status,
                                                          uint64_t list_len) {
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t startElem = start_elem(worker, list_len);
    const uint64_t stopElem = stop_elem(worker, list_len);

    for (uint64_t idx = startElem; idx < stopElem; idx++) {
        const int32_t offset = output_offsets[idx];
        const int32_t next_offset = output_offsets[idx + 1];
        if (offset < 0 || next_offset < 0) {
            atomicMax(status, 2u);
            continue;
        }

        uint64_t size = 0;
        if (!non_negative_to_u64(sizes[idx], &size)) {
            atomicMax(status, 1u);
            continue;
        }

        const int64_t expected_next = static_cast<int64_t>(offset) + static_cast<int64_t>(size);
        if (size > static_cast<uint64_t>(INT32_MAX) || expected_next != static_cast<int64_t>(next_offset)) {
            atomicMax(status, 2u);
        }
    }
}

template <typename OffsetT, typename SizeT>
__device__ void list_view_rebuild_primitive_device(const OffsetT *const offsets,
                                                   const SizeT *const sizes,
                                                   const int32_t *const output_offsets,
                                                   const uint8_t *const input_values,
                                                   uint8_t *const output_values,
                                                   uint32_t *const status,
                                                   uint64_t list_len,
                                                   uint64_t elements_len,
                                                   uint64_t value_width) {
    const uint64_t worker = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t startElem = start_elem(worker, list_len);
    const uint64_t stopElem = stop_elem(worker, list_len);

    for (uint64_t list_idx = startElem; list_idx < stopElem; list_idx++) {
        uint64_t input_offset = 0;
        uint64_t size = 0;
        if (!non_negative_to_u64(offsets[list_idx], &input_offset) ||
            !non_negative_to_u64(sizes[list_idx], &size)) {
            atomicMax(status, 1u);
            continue;
        }

        uint64_t input_end = 0;
        if (!checked_add_u64(input_offset, size, &input_end) || input_end > elements_len) {
            atomicMax(status, 1u);
            continue;
        }

        const uint64_t output_idx = static_cast<uint64_t>(output_offsets[list_idx]);
        for (uint64_t element_idx = 0; element_idx < size; element_idx++) {
            const uint64_t input_byte = (input_offset + element_idx) * value_width;
            const uint64_t output_byte = (output_idx + element_idx) * value_width;
            for (uint64_t byte_idx = 0; byte_idx < value_width; byte_idx++) {
                output_values[output_byte + byte_idx] = input_values[input_byte + byte_idx];
            }
        }
    }
}

} // namespace

#define GENERATE_VALIDATE_OFFSETS(SizeT, size_suffix)                                                        \
    extern "C" __global__ void list_view_rebuild_validate_offsets_##size_suffix(                             \
        const SizeT *const sizes,                                                                            \
        const int32_t *const output_offsets,                                                                 \
        uint32_t *const status,                                                                              \
        uint64_t list_len) {                                                                                 \
        list_view_rebuild_validate_offsets_device(sizes, output_offsets, status, list_len);                  \
    }

#define GENERATE_KERNEL(OffsetT, offset_suffix, SizeT, size_suffix)                                          \
    extern "C" __global__ void list_view_offsets_##offset_suffix##_##size_suffix(                            \
        const OffsetT *const offsets,                                                                        \
        const SizeT *const sizes,                                                                            \
        int32_t *const output,                                                                               \
        uint32_t *const status,                                                                              \
        uint64_t list_len) {                                                                                 \
        list_view_offsets_device(offsets, sizes, output, status, list_len);                                  \
    }                                                                                                        \
    extern "C" __global__ void list_view_rebuild_primitive_##offset_suffix##_##size_suffix(                  \
        const OffsetT *const offsets,                                                                        \
        const SizeT *const sizes,                                                                            \
        const int32_t *const output_offsets,                                                                 \
        const uint8_t *const input_values,                                                                   \
        uint8_t *const output_values,                                                                        \
        uint32_t *const status,                                                                              \
        uint64_t list_len,                                                                                   \
        uint64_t elements_len,                                                                               \
        uint64_t value_width) {                                                                              \
        list_view_rebuild_primitive_device(offsets,                                                          \
                                           sizes,                                                            \
                                           output_offsets,                                                   \
                                           input_values,                                                     \
                                           output_values,                                                    \
                                           status,                                                           \
                                           list_len,                                                         \
                                           elements_len,                                                     \
                                           value_width);                                                     \
    }

#define GENERATE_INIT_SCAN(SizeT, size_suffix)                                                               \
    extern "C" __global__ void list_view_rebuild_init_scan_##size_suffix(const SizeT *const sizes,           \
                                                                         int32_t *const scan,                \
                                                                         uint32_t *const status,             \
                                                                         uint64_t list_len,                  \
                                                                         uint64_t scan_len) {                \
        list_view_rebuild_init_scan_device(sizes, scan, status, list_len, scan_len);                         \
    }

#define GENERATE_SIZE_KERNELS(OffsetT, offset_suffix)                                                        \
    GENERATE_KERNEL(OffsetT, offset_suffix, uint8_t, u8)                                                     \
    GENERATE_KERNEL(OffsetT, offset_suffix, uint16_t, u16)                                                   \
    GENERATE_KERNEL(OffsetT, offset_suffix, uint32_t, u32)                                                   \
    GENERATE_KERNEL(OffsetT, offset_suffix, uint64_t, u64)                                                   \
    GENERATE_KERNEL(OffsetT, offset_suffix, int8_t, i8)                                                      \
    GENERATE_KERNEL(OffsetT, offset_suffix, int16_t, i16)                                                    \
    GENERATE_KERNEL(OffsetT, offset_suffix, int32_t, i32)                                                    \
    GENERATE_KERNEL(OffsetT, offset_suffix, int64_t, i64)

GENERATE_INIT_SCAN(uint8_t, u8)
GENERATE_INIT_SCAN(uint16_t, u16)
GENERATE_INIT_SCAN(uint32_t, u32)
GENERATE_INIT_SCAN(uint64_t, u64)
GENERATE_INIT_SCAN(int8_t, i8)
GENERATE_INIT_SCAN(int16_t, i16)
GENERATE_INIT_SCAN(int32_t, i32)
GENERATE_INIT_SCAN(int64_t, i64)

GENERATE_VALIDATE_OFFSETS(uint8_t, u8)
GENERATE_VALIDATE_OFFSETS(uint16_t, u16)
GENERATE_VALIDATE_OFFSETS(uint32_t, u32)
GENERATE_VALIDATE_OFFSETS(uint64_t, u64)
GENERATE_VALIDATE_OFFSETS(int8_t, i8)
GENERATE_VALIDATE_OFFSETS(int16_t, i16)
GENERATE_VALIDATE_OFFSETS(int32_t, i32)
GENERATE_VALIDATE_OFFSETS(int64_t, i64)

GENERATE_SIZE_KERNELS(uint8_t, u8)
GENERATE_SIZE_KERNELS(uint16_t, u16)
GENERATE_SIZE_KERNELS(uint32_t, u32)
GENERATE_SIZE_KERNELS(uint64_t, u64)
GENERATE_SIZE_KERNELS(int8_t, i8)
GENERATE_SIZE_KERNELS(int16_t, i16)
GENERATE_SIZE_KERNELS(int32_t, i32)
GENERATE_SIZE_KERNELS(int64_t, i64)
