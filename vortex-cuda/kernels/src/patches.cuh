// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "patches.h"

/// A single patch: a within-chunk index and its replacement value.
/// A sentinel patch has index == 1024, which can never match a valid
/// within-chunk position (0–1023).
template <typename T>
struct Patch {
    uint16_t index;
    T value;
};

/// Cursor for iterating over a single lane's patches within a chunk.
///
/// Usage in the generated merge-loop:
///
///     PatchesCursor<uint32_t> cursor(patches, blockIdx.x, thread_idx, 32);
///     auto patch = cursor.next();
///     for (int i = 0; i < 32; i++) {
///         auto idx = i * 32 + thread_idx;
///         if (idx == patch.index) {
///             out[idx] = patch.value;
///             patch = cursor.next();
///         } else {
///             out[idx] = shared_out[idx];
///         }
///     }
template <typename T>
class PatchesCursor {
public:
    /// Construct a cursor positioned at the patches for the given (chunk, lane).
    /// n_lanes is a compile-time constant emitted by the code generator (16 or 32).
    __device__ PatchesCursor(const GPUPatches &patches, uint32_t chunk, uint32_t lane, uint32_t n_lanes) {
        if (patches.lane_offsets == nullptr) {
            indices = nullptr;
            values = nullptr;
            remaining = 0;
            return;
        }
        auto slot = chunk * n_lanes + lane;
        auto start = patches.lane_offsets[slot];
        remaining = patches.lane_offsets[slot + 1] - start;
        indices = patches.indices + start;
        values = reinterpret_cast<const T *>(patches.values) + start;
    }

    /// Return the current patch and advance, or a sentinel {1024, 0} if exhausted.
    __device__ Patch<T> next() {
        if (remaining == 0) {
            return {1024, T {}};
        }
        Patch<T> patch = {*indices, *values};
        indices++;
        values++;
        remaining--;
        return patch;
    }

private:
    const uint16_t *indices;
    const T *values;
    uint8_t remaining;
};