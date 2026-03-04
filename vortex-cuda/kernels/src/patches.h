// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// GPU-resident patches for fused exception patching during bit-unpacking.
///
/// Patches are stored in a lane-wise transposed layout: for each (chunk, lane) pair,
/// the corresponding patch indices and values are stored contiguously. The lane_offsets
/// array is a CSR-style offset array of size (n_chunks * n_lanes + 1) that maps each
/// (chunk, lane) slot to its range in the indices and values arrays.
///
/// A NULL lane_offsets pointer indicates no patches are present.
typedef struct {
    uint32_t *lane_offsets;
    uint16_t *indices;
    void *values;
} GPUPatches;

#ifdef __cplusplus
}
#endif