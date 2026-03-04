// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <metal_stdlib>
using namespace metal;

// Frame-of-Reference decoding kernel.
// Adds a reference value to each element in the array.
//
// This kernel uses thread_position_in_grid for direct indexing,
// allowing Metal to handle the grid/threadgroup sizing optimally.

// Template for FoR kernel
template <typename T>
void for_kernel_impl(
    device T* values [[buffer(0)]],
    constant T& reference [[buffer(1)]],
    constant uint64_t& array_len [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < array_len) {
        values[gid] = values[gid] + reference;
    }
}

// Explicit kernel instantiations for each integer type
// Metal does not support C++ templates with extern "C" linkage,
// so we need explicit functions for each type.

kernel void for_u8(
    device uint8_t* values [[buffer(0)]],
    constant uint8_t& reference [[buffer(1)]],
    constant uint64_t& array_len [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    for_kernel_impl(values, reference, array_len, gid);
}

kernel void for_u16(
    device uint16_t* values [[buffer(0)]],
    constant uint16_t& reference [[buffer(1)]],
    constant uint64_t& array_len [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    for_kernel_impl(values, reference, array_len, gid);
}

kernel void for_u32(
    device uint32_t* values [[buffer(0)]],
    constant uint32_t& reference [[buffer(1)]],
    constant uint64_t& array_len [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    for_kernel_impl(values, reference, array_len, gid);
}

kernel void for_u64(
    device uint64_t* values [[buffer(0)]],
    constant uint64_t& reference [[buffer(1)]],
    constant uint64_t& array_len [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    for_kernel_impl(values, reference, array_len, gid);
}

kernel void for_i8(
    device int8_t* values [[buffer(0)]],
    constant int8_t& reference [[buffer(1)]],
    constant uint64_t& array_len [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    for_kernel_impl(values, reference, array_len, gid);
}

kernel void for_i16(
    device int16_t* values [[buffer(0)]],
    constant int16_t& reference [[buffer(1)]],
    constant uint64_t& array_len [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    for_kernel_impl(values, reference, array_len, gid);
}

kernel void for_i32(
    device int32_t* values [[buffer(0)]],
    constant int32_t& reference [[buffer(1)]],
    constant uint64_t& array_len [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    for_kernel_impl(values, reference, array_len, gid);
}

kernel void for_i64(
    device int64_t* values [[buffer(0)]],
    constant int64_t& reference [[buffer(1)]],
    constant uint64_t& array_len [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    for_kernel_impl(values, reference, array_len, gid);
}
