// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <stdint.h>

// Kernel launch configuration constants.
// Must match the Rust launch config in src/kernel/mod.rs.
//
// With THREADS_PER_BLOCK=64 (set by Rust) and ELEMENTS_PER_THREAD=32:
//   elements_per_block = 64 * 32 = 2048
//   grid_dim = ceil(array_len / 2048)
constexpr uint32_t ELEMENTS_PER_THREAD = 32;

// We use `::min` from CUDA's `crt/math_functions.hpp` (declared `__host__ __device__`)
// rather than `std::min` which is host-only.

__device__ constexpr inline uint32_t start_elem(uint32_t idx, uint32_t len) {
    return ::min(idx * ELEMENTS_PER_THREAD, len);
}

__device__ constexpr inline uint32_t stop_elem(uint32_t idx, uint32_t len) {
    return ::min(start_elem(idx, len) + ELEMENTS_PER_THREAD, len);
}
