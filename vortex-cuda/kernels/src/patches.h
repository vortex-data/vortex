// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Type tag for chunk_offsets pointer.
typedef enum { CO_U8 = 0, CO_U16 = 1, CO_U32 = 2, CO_U64 = 3 } ChunkOffsetType;

/// GPU-resident patches for fused exception patching during bit-unpacking.
///
/// Patches are stored in sorted order within each chunk. The chunk_offsets
/// array is a CSR-style offset array of size (n_chunks + 1) that maps each
/// chunk to its range in the indices and values arrays.
///
/// A NULL chunk_offsets pointer indicates no patches are present.
typedef struct {
    void *chunk_offsets;
    ChunkOffsetType chunk_offset_type;
    uint32_t *indices;
    void *values;
    uint32_t offset;
} GPUPatches;

#ifdef __cplusplus
}
#endif