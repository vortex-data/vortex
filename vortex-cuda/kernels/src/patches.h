// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Type tag for chunk_offsets pointer.
typedef enum { CO_U8 = 0, CO_U16 = 1, CO_U32 = 2, CO_U64 = 3 } ChunkOffsetType;

static const uint32_t PATCH_DERIVE_INDICES_BASE = UINT32_MAX;

/// GPU-resident patches for fused exception patching during bit-unpacking.
///
/// Patches are stored in sorted order within each chunk. The chunk_offsets
/// array maps each chunk to the start of its range in the indices/values arrays.
/// The array has n_chunks elements (not n_chunks+1); the final offset is implicit
/// and equals num_patches.
///
/// A NULL chunk_offsets pointer indicates no patches are present.
typedef struct {
    void *chunk_offsets;
    ChunkOffsetType chunk_offset_type;
    uint32_t *indices;
    void *values;
    uint32_t offset;
    uint32_t offset_within_chunk;
    uint32_t num_patches;
    uint32_t n_chunks;
    // PATCH_DERIVE_INDICES_BASE means derive from chunk_offsets[0] +
    // offset_within_chunk. Chunk-offset-only sliced views need an explicit
    // base because their indices/values buffers still start at patch 0.
    uint32_t indices_base;
} GPUPatches;

#ifdef __cplusplus
}
#endif
