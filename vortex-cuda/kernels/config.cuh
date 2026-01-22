// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

// Kernel launch configuration constants.
// Must match the Rust launch config in src/kernel/mod.rs.
//
// With THREADS_PER_BLOCK=64 (set by Rust) and ELEMENTS_PER_THREAD=32:
//   elements_per_block = 64 * 32 = 2048
//   grid_dim = ceil(array_len / 2048)
constexpr uint32_t ELEMENTS_PER_THREAD = 32;
