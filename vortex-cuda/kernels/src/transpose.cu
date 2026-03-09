// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "types.cuh"
#include <stdint.h>

/// Number of lanes used for patching based on value type size.
///
/// For types 32-bits or smaller, we use 32 lanes. For 64-bit types, we use 16 lanes.
/// This matches the Rust implementation in types.rs.
template <typename V>
__device__ __forceinline__ constexpr uint32_t patch_lanes() {
    return sizeof(V) < 8 ? 32 : 16;
}

/// Core transpose implementation.
///
/// Transposes patches from sorted index order into lane-wise layout for parallel GPU execution.
/// This mirrors the Rust `transpose` function in vortex-cuda/src/kernel/patches/types.rs.
///
/// The kernel is launched with a single thread and populates lane_offsets, output_indices,
/// and output_values in the same manner as the Rust code.
template <typename IndexT, typename ValueT>
__device__ void transpose_impl(const IndexT *__restrict patch_indices,
                               const ValueT *__restrict patch_values,
                               uint64_t len,
                               uint64_t offset,
                               uint64_t array_len,
                               uint32_t *__restrict lane_offsets,
                               uint16_t *__restrict output_indices,
                               ValueT *__restrict output_values) {

    printf("patch_indices: %p\n", (void*) patch_indices);
    printf("patch_values: %p\n", (void*) patch_values);
    printf("len: %d\n", len);
    printf("offset: %d\n", offset);
    printf("array_len: %d\n", array_len);
    printf("lane_offsets: %p\n", (void*) lane_offsets);
    printf("output_indices: %p\n", (void*) output_indices);
    printf("output_values: %p\n", (void*) output_values);

    const uint32_t n_chunks = (array_len + 1023) / 1024; // div_ceil(array_len, 1024)
    const uint32_t n_lanes = patch_lanes<ValueT>();
    const uint32_t num_slots = n_chunks * n_lanes + 1;
    printf("num_slots: %d\n", num_slots);

    // Initialize lane_offsets to zero
    for (uint32_t i = 0; i < num_slots; i++) {
        lane_offsets[i] = 0;
    }

    // First pass: count patches per (chunk, lane) pair
    for (uint64_t i = 0; i < len; i++) {
        uint64_t index = static_cast<uint64_t>(patch_indices[i]) - offset;
        uint32_t chunk = index / 1024;
        uint32_t lane = index % n_lanes;

        lane_offsets[chunk * n_lanes + lane + 1] += 1;
    }

    // Prefix sum: convert counts to offsets
    for (uint32_t i = 1; i < num_slots; i++) {
        lane_offsets[i] += lane_offsets[i - 1];
    }

    // Second pass: scatter indices and values to their final positions
    for (uint64_t i = 0; i < len; i++) {
        uint64_t index = static_cast<uint64_t>(patch_indices[i]) - offset;
        uint32_t chunk = index / 1024;
        uint32_t lane = index % n_lanes;

        uint32_t position = lane_offsets[chunk * n_lanes + lane];
        output_indices[position] = static_cast<uint16_t>(index % 1024);
        output_values[position] = patch_values[i];
        lane_offsets[chunk * n_lanes + lane] += 1;
    }

    // Third pass: restore offsets by decrementing
    for (uint64_t i = 0; i < len; i++) {
        uint64_t index = static_cast<uint64_t>(patch_indices[i]) - offset;
        uint32_t chunk = index / 1024;
        uint32_t lane = index % n_lanes;

        lane_offsets[chunk * n_lanes + lane] -= 1;
    }
}

// Generate transpose kernel for a specific (IndexT, ValueT) combination
#define GENERATE_TRANSPOSE_KERNEL(ValueT, value_suffix, IndexT, index_suffix)                                    \
    extern "C" __global__ void transpose_##index_suffix##_##value_suffix(                                        \
        const IndexT *__restrict patch_indices,                                                                  \
        const ValueT *__restrict patch_values,                                                                   \
        uint64_t len,                                                                                            \
        uint64_t offset,                                                                                         \
        uint64_t array_len,                                                                                      \
        uint32_t *__restrict lane_offsets,                                                                       \
        uint16_t *__restrict output_indices,                                                                     \
        ValueT *__restrict output_values) {                                                                      \
        transpose_impl(patch_indices, patch_values, len, offset, array_len, lane_offsets, output_indices,        \
                       output_values);                                                                           \
    }

// Generate transpose kernels for all index types (unsigned integers) for a given value type
#define GENERATE_TRANSPOSE_FOR_ALL_INDICES(value_suffix, ValueT)                                                 \
    GENERATE_TRANSPOSE_KERNEL(ValueT, value_suffix, uint8_t, u8)                                                 \
    GENERATE_TRANSPOSE_KERNEL(ValueT, value_suffix, uint16_t, u16)                                               \
    GENERATE_TRANSPOSE_KERNEL(ValueT, value_suffix, uint32_t, u32)                                               \
    GENERATE_TRANSPOSE_KERNEL(ValueT, value_suffix, uint64_t, u64)

// Generate for all native SIMD ptypes (matches the Rust match_each_native_ptype)
FOR_EACH_NATIVE_SIMD_PTYPE(GENERATE_TRANSPOSE_FOR_ALL_INDICES)
