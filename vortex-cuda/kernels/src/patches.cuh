// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "patches.h"
#include "fastlanes_common.cuh"

// Cursor that makes it easy to statefully iterate over a set of patches.
struct PatchesCursor {
    // The set of patches
    GPUPatches patches;

    uint32_t chunk_index;
    uint32_t lane;
    // this is number of patches for the currently configured chunk_index
    // and lane and is set whenever seek() is called
    uint32_t n_patches;

    // this is the offset into the indices.
    uint32_t start_offset;
    uint32_t offset;
}

// Reset the cursor to the start of the patches range given at this.
void PatchesCursor::seek(uint32_t chunk, uint32_t lane) {
    this.chunk_index = chunk;
    this.lane = lane;
    this.offset = 0;

    auto idx = chunk * 1024 + lane;
    auto startIdx = patches.lane_offsets[idx];
    auto stopIdx = patches.lane_offsets[idx+1];

    this.n_patches = stopIdx - startIdx;
    this.start_offset = startIdx;
}

// Advance to the next patch in the patching group.
bool PatchesCursor::next() {
    if (offset >= n_patches) {
        return false;
    }

    ++offset;
    return true;
}

template<typename T>
T PatchesCursor::get_value() {
    T* values = reinterpret_cast<T*>(patches.values);
    return values[start_offset + offset];
}

// Get the patch index of the current position in the cursor.
uint16_t PatchesCursor::get_index() {
    return patches.indices[start_offset + offset];
}

// If there is a patch stored in the GPUPatches provided for the next item in the given chunk/lane, return it.
// Otherwise, we patch the value type instead.
template<typename T>
__device__ T maybe_patch(T value, int index, int& counter, GPUPatches patches) {
    constexpr auto LANES = 128 = size_of(T);

    int chunk_index = index / 1024;
    int lane = index % LANES;

    // When does this work exactly? You're decoding many values here.
    // If this index is going to be the next thing, we skip it. Otherwise, we advance it.

    auto start = chunk * LANES + lane;
    auto stop = start + 1;

    auto startIdx = patches.lane_offsets[start];
    auto stopIdx = patches.lane_offsets[stop];

    auto n_lane_patches = stopIdx - startIdx;

    auto index = patches.indices[startIdx + count];

    // patch counter
}