// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <cuda_fp16.h>
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

// Type iteration macros - call MACRO(suffix, Type) for each type in category.
// These mirror the Rust match_each_*_ptype macros.

// Unsigned integers
#define FOR_EACH_UNSIGNED_INT(MACRO) \
    MACRO(u8, uint8_t) \
    MACRO(u16, uint16_t) \
    MACRO(u32, uint32_t) \
    MACRO(u64, uint64_t)

// Signed integers
#define FOR_EACH_SIGNED_INT(MACRO) \
    MACRO(i8, int8_t) \
    MACRO(i16, int16_t) \
    MACRO(i32, int32_t) \
    MACRO(i64, int64_t)

// All integers (signed + unsigned)
#define FOR_EACH_INTEGER(MACRO) \
    FOR_EACH_UNSIGNED_INT(MACRO) \
    FOR_EACH_SIGNED_INT(MACRO)

// All floating point types (requires #include <cuda_fp16.h>)
#define FOR_EACH_FLOAT(MACRO) \
    MACRO(f16, __half) \
    MACRO(f32, float) \
    MACRO(f64, double)

// Native SIMD types (integers + f32/f64, matches match_each_native_simd_ptype)
#define FOR_EACH_NATIVE_SIMD_PTYPE(MACRO) \
    FOR_EACH_INTEGER(MACRO) \
    MACRO(f32, float) \
    MACRO(f64, double)

// All native ptypes (requires #include <cuda_fp16.h>, matches match_each_native_ptype)
#define FOR_EACH_NATIVE_PTYPE(MACRO) \
    FOR_EACH_INTEGER(MACRO) \
    FOR_EACH_FLOAT(MACRO)

// Large decimal types (128-bit and 256-bit integers for decimal representation).
// Use alongside FOR_EACH_NATIVE_PTYPE for full type coverage.
#define FOR_EACH_LARGE_DECIMAL(MACRO) \
    MACRO(i128, int128_t) \
    MACRO(i256, int256_t)

// All numeric types: native ptypes + large decimals (requires #include <cuda_fp16.h>)
#define FOR_EACH_NUMERIC(MACRO) \
    FOR_EACH_NATIVE_PTYPE(MACRO) \
    FOR_EACH_LARGE_DECIMAL(MACRO)
