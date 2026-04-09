// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "types.cuh"

/// Apply patches to an output array using the transposed Patched array format.
///
/// This kernel uses a thread-per-lane model where each thread is assigned to
/// one (chunk, lane) slot and applies all patches in that slot.
template <typename ValueT>
__device__ void patched(ValueT *const output,
                        const uint32_t *const lane_offsets,
                        const uint16_t *const patch_indices,
                        const ValueT *const patch_values,
                        uint32_t n_lanes,
                        uint32_t total_lane_slots,
                        uint64_t offset,
                        uint64_t len) {
    const uint32_t lane_slot = blockIdx.x * blockDim.x + threadIdx.x;

    // Early return if this thread is beyond the number of lane slots
    if (lane_slot >= total_lane_slots) {
        return;
    }

    // Determine which chunk this lane slot belongs to
    const uint32_t chunk = lane_slot / n_lanes;

    // Get the range of patches for this lane slot
    const uint32_t start = lane_offsets[lane_slot];
    const uint32_t stop = lane_offsets[lane_slot + 1];

    // Apply all patches in this lane
    for (uint32_t p = start; p < stop; p++) {
        // Get within-chunk index and compute global position
        const uint16_t within_chunk_idx = patch_indices[p];
        const uint64_t global_idx = static_cast<uint64_t>(chunk) * 1024 + within_chunk_idx;

        // Check bounds (for sliced arrays)
        if (global_idx < offset) {
            continue;
        }

        if (global_idx >= offset + len) {
            break;
        }

        output[global_idx - offset] = patch_values[p];
    }
}

#define GENERATE_PATCHED_KERNEL(value_suffix, ValueT)                                                        \
    extern "C" __global__ void patched_##value_suffix(ValueT *const output,                                  \
                                                      const uint32_t *const lane_offsets,                    \
                                                      const uint16_t *const patch_indices,                   \
                                                      const ValueT *const patch_values,                      \
                                                      uint32_t n_lanes,                                      \
                                                      uint32_t total_lane_slots,                             \
                                                      uint64_t offset,                                       \
                                                      uint64_t len) {                                        \
        patched(output, lane_offsets, patch_indices, patch_values, n_lanes, total_lane_slots, offset, len);  \
    }

// Generate for all native SIMD ptypes
FOR_EACH_NATIVE_SIMD_PTYPE(GENERATE_PATCHED_KERNEL)
