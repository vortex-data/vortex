// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "fastlanes_common.cuh"
#include "patches.h"

/// Load a chunk offset value, dispatching on the runtime type.
__device__ inline uint32_t load_chunk_offset(const GPUPatches &patches, uint32_t idx) {
    switch (patches.chunk_offset_type) {
    case CO_U8:
        return reinterpret_cast<const uint8_t *>(patches.chunk_offsets)[idx];
    case CO_U16:
        return reinterpret_cast<const uint16_t *>(patches.chunk_offsets)[idx];
    case CO_U32:
        return reinterpret_cast<const uint32_t *>(patches.chunk_offsets)[idx];
    case CO_U64:
        return static_cast<uint32_t>(reinterpret_cast<const uint64_t *>(patches.chunk_offsets)[idx]);
    }
    return 0;
}

/// A single patch: a within-chunk index and its replacement value.
/// A sentinel patch has index == FL_CHUNK, which can never match a valid
/// within-chunk position (0–FL_CHUNK-1).
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
///     while (patch.index != FL_CHUNK) {
///         shared_out[patch.index] = patch.value;
///         patch = cursor.next();
///     }
template <typename T>
class PatchesCursor {
public:
    /// Construct a cursor for this thread's portion of patches in the chunk.
    __device__
    PatchesCursor(const GPUPatches &patches, uint32_t chunk, uint32_t thread_idx, uint32_t n_threads) {
        if (patches.chunk_offsets == nullptr) {
            indices = nullptr;
            values = nullptr;
            remaining = 0;
            return;
        }

        if (chunk >= patches.n_chunks) {
            indices = nullptr;
            values = nullptr;
            remaining = 0;
            return;
        }

        const uint32_t indices_base = patches.indices_base == PATCH_DERIVE_INDICES_BASE
                                          ? load_chunk_offset(patches, 0) + patches.offset_within_chunk
                                          : patches.indices_base;

        // Convert chunk_offsets entries into offsets within indices/values.
        // Ordinary sliced patches derive the base from the first chunk offset;
        // chunk-offset-only sliced views provide it explicitly.
        uint32_t patches_start_idx = load_chunk_offset(patches, chunk);
        patches_start_idx = (patches_start_idx > indices_base) ? patches_start_idx - indices_base : 0;

        // calculate the ending index.
        uint32_t patches_end_idx;
        if ((chunk + 1) < patches.n_chunks) {
            patches_end_idx = load_chunk_offset(patches, chunk + 1);
            patches_end_idx = (patches_end_idx > indices_base) ? patches_end_idx - indices_base : 0;
        } else {
            patches_end_idx = patches.num_patches;
        }
        patches_end_idx = min(patches_end_idx, patches.num_patches);
        patches_end_idx = max(patches_end_idx, patches_start_idx);

        // calculate how many patches are in the chunk
        uint32_t num_patches = patches_end_idx - patches_start_idx;

        // Divide patches among threads (ceil division)
        uint32_t patches_per_thread = (num_patches + n_threads - 1) / n_threads;
        uint32_t my_start = min(thread_idx * patches_per_thread, num_patches);
        uint32_t my_end = min((thread_idx + 1) * patches_per_thread, num_patches);

        uint32_t start = patches_start_idx + my_start;
        remaining = my_end - my_start;
        indices = patches.indices + start;
        values = reinterpret_cast<const T *>(patches.values) + start;

        // The iterator returns indices relative to the start of the chunk.
        // `chunk_base` is the index of the first element within a chunk, accounting
        // for the slice offset.
        chunk_base = chunk * FL_CHUNK + patches.offset;
        chunk_base -= min(chunk_base, patches.offset % FL_CHUNK);
    }

    /// Return the current patch (with within-chunk index) and advance,
    /// or a sentinel {1024, 0} if exhausted.
    __device__ Patch<T> next() {
        if (remaining == 0) {
            return {FL_CHUNK, T {}};
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
    uint32_t remaining;
    uint32_t chunk_base;
};
