// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#ifndef VORTEX_CUDA_TYPES_CUH
#define VORTEX_CUDA_TYPES_CUH

#include <stdint.h>

// 128-bit signed integer type for decimal values
struct __align__(16) int128_t {
    int64_t lo;
    int64_t hi;
};

// 256-bit signed integer type for decimal values
struct __align__(32) int256_t {
    int64_t parts[4];
};

#endif // VORTEX_CUDA_TYPES_CUH