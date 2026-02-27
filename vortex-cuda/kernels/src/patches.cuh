// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <stdio.h>
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

    __device__ PatchesCursor(GPUPatches p)
        : patches(p), chunk_index(0), lane(0), n_patches(0), start_offset(0), offset(0) {}

    // Reset the cursor to the start of the patches range given at this.
    __device__ void seek(uint32_t chunk, uint32_t theLane) {
        chunk_index = chunk;
        lane = theLane;
        offset = 0;
    
        auto idx = chunk * 1024 + lane;
        auto startIdx = patches.lane_offsets[idx];
        auto stopIdx = patches.lane_offsets[idx+1];
    
        n_patches = stopIdx - startIdx;
        start_offset = startIdx;
        printf("SEEK: chunk %d LANE %d, n_patches = %d\n", chunk_index, lane, n_patches);
    }
    
    // Advance to the next patch in the patching group.
    __device__ bool next() {
        if (offset >= n_patches) {
            printf("LANE %d: exhausted all %d patches\n", lane, n_patches);
            return false;
        }
    
        ++offset;
        return true;
    }
    
    template<typename T>
    __device__ T get_value() {
        T* values = reinterpret_cast<T*>(patches.values);
        return values[start_offset + offset];
    }
    
    // Get the patch index of the current position in the cursor.
    __device__ uint16_t get_index() {
        return patches.indices[start_offset + offset];
    }
};
