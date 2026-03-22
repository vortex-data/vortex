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

template <typename T>
__device__ constexpr inline T start_elem(T idx, T len) {
    return ::min(idx * static_cast<T>(ELEMENTS_PER_THREAD), len);
}

template <typename T>
__device__ constexpr inline T stop_elem(T idx, T len) {
    return ::min(start_elem(idx, len) + static_cast<T>(ELEMENTS_PER_THREAD), len);
}
