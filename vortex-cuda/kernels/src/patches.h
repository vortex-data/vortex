// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Patches that must be traversed by each individual element type here */
typedef struct {
    uint32_t n_chunks;
    uint32_t n_lanes;
    uint32_t *lane_offsets;
    uint16_t *indices;
    void *values;
} GPUPatches;

#ifdef __cplusplus
}
#endif
