// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAKE_PATCHES(V, suffix) \
typedef struct { \
    uint32_t n_chunks; \
    uint32_t n_lanes; \
    uint32_t *lane_offsets; \
    uint16_t *indices; \
    V *values; \
} GPUPatches_##suffix;

// GPUPatches_u8
MAKE_PATCHES(uint8_t, u8)

// GPUPatches_u16
MAKE_PATCHES(uint16_t, u16)

// GPUPatches_u32
MAKE_PATCHES(uint32_t, u32)

// GPUPatches_u64
MAKE_PATCHES(uint64_t, u64)


#ifdef __cplusplus
}
#endif
