// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <stdint.h>

// FastLanes ordering array
__constant__ int FL_ORDER[] = {0, 4, 2, 6, 1, 5, 3, 7};

// Compute the index in the FastLanes layout
#define INDEX(row, lane) (FL_ORDER[row / 8] * 16 + (row % 8) * 128 + lane)

// Create a mask with 'width' bits set
#define MASK(T, width) (((T)1 << width) - 1)

/// Bit Field Extract: extract `len` bits starting at bit position `start`.
/// Equivalent to `(val >> start) & ((1 << len) - 1)` but compiles to a
/// single `bfe.u32` instruction instead of separate `shr` + `and`.
__device__ __forceinline__ uint32_t BFE(uint32_t val, uint32_t start, uint32_t len) {
    uint32_t result;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(val), "r"(start), "r"(len));
    return result;
}

__device__ __forceinline__ uint64_t BFE(uint64_t val, uint32_t start, uint32_t len) {
    uint64_t result;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(val), "r"(start), "r"(len));
    return result;
}

/// Overloads for narrow types: promote to u32, extract, truncate back.
/// The GPU operates on 32-bit registers regardless, so this is zero-cost.
__device__ __forceinline__ uint8_t BFE(uint8_t val, uint32_t start, uint32_t len) {
    return static_cast<uint8_t>(BFE(static_cast<uint32_t>(val), start, len));
}

__device__ __forceinline__ uint16_t BFE(uint16_t val, uint32_t start, uint32_t len) {
    return static_cast<uint16_t>(BFE(static_cast<uint32_t>(val), start, len));
}

/// Funnel shift right: extract a 32-bit window from the 64-bit concatenation
/// `{hi, lo}` shifted right by `shift` bits. Compiles to `shf.r.clamp.b32`.
/// Useful for extracting values that span two packed words.
__device__ __forceinline__ uint32_t FUNNEL_SHIFT_R(uint32_t lo, uint32_t hi, uint32_t shift) {
    return __funnelshift_r(lo, hi, shift);
}
