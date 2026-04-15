// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "patches.h"

/// Load a chunk offset value, dispatching on the runtime type.
__device__ inline uint32_t load_chunk_offset(const GPUPatches &patches, uint32_t idx) {
    switch (patches.chunk_offset_type) {
        case CO_U8:  return reinterpret_cast<const uint8_t*>(patches.chunk_offsets)[idx];
        case CO_U16: return reinterpret_cast<const uint16_t*>(patches.chunk_offsets)[idx];
        case CO_U32: return reinterpret_cast<const uint32_t*>(patches.chunk_offsets)[idx];
        case CO_U64: return static_cast<uint32_t>(
            reinterpret_cast<const uint64_t*>(patches.chunk_offsets)[idx]);
    }
    return 0;
}

/// A single patch: a within-chunk index and its replacement value.
/// A sentinel patch has index == 1024, which can never match a valid
/// within-chunk position (0–1023).
template <typename T>
struct Patch {
    uint16_t index;
    T value;
};

/// Cursor for iterating over a thread's portion of patches within a chunk.
///
/// Patches are divided evenly among threads. Each thread applies its patches
/// to shared memory, then all threads sync and copy to global memory.
///
/// Usage in the generated kernel:
///
///     PatchesCursor<uint32_t> cursor(patches, blockIdx.x, thread_idx, 32);
///     auto patch = cursor.next();
///     while (patch.index != 1024) {
///         shared_out[patch.index] = patch.value;
///         patch = cursor.next();
///     }
template <typename T>
class PatchesCursor {
public:
    /// Construct a cursor for this thread's portion of patches in the chunk.
    __device__ PatchesCursor(const GPUPatches &patches, uint32_t chunk, uint32_t thread_idx, uint32_t n_threads) {
        if (patches.chunk_offsets == nullptr) {
            indices = nullptr;
            values = nullptr;
            remaining = 0;
            return;
        }

        // Get patch range for this chunk.
        // chunk_offsets has n_chunks elements; the final offset is implicit (num_patches).
        uint32_t chunk_start = load_chunk_offset(patches, chunk);
        uint32_t chunk_end = (chunk + 1 < patches.n_chunks)
            ? load_chunk_offset(patches, chunk + 1)
            : patches.num_patches;
        uint32_t num_patches = chunk_end - chunk_start;

        // Divide patches among threads (ceil division)
        uint32_t patches_per_thread = (num_patches + n_threads - 1) / n_threads;
        uint32_t my_start = min(thread_idx * patches_per_thread, num_patches);
        uint32_t my_end = min((thread_idx + 1) * patches_per_thread, num_patches);

        uint32_t start = chunk_start + my_start;
        remaining = my_end - my_start;
        indices = patches.indices + start;
        values = reinterpret_cast<const T *>(patches.values) + start;

        // Precompute base for within-chunk index calculation
        chunk_base = patches.offset + chunk * 1024;
    }

    /// Return the current patch (with within-chunk index) and advance,
    /// or a sentinel {1024, 0} if exhausted.
    __device__ Patch<T> next() {
        if (remaining == 0) {
            return {1024, T{}};
        }
        uint16_t within_chunk = static_cast<uint16_t>(*indices - chunk_base);
        Patch<T> patch = {within_chunk, *values};
        indices++;
        values++;
        remaining--;
        return patch;
    }

private:
    const uint32_t *indices;
    const T *values;
    uint8_t remaining;
    uint32_t chunk_base;
};
