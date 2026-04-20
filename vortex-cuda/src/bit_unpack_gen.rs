// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::io::Write;

use fastlanes::FastLanes;

/// Emit the per-row bit extraction for a single row of a general-case lane decoder.
fn write_row(output: &mut impl Write, bits: usize, bit_width: usize, row: usize) -> io::Result<()> {
    let curr_word = (row * bit_width) / bits;
    let next_word = ((row + 1) * bit_width) / bits;
    let shift = (row * bit_width) % bits;

    if next_word > curr_word {
        let remaining_bits = ((row + 1) * bit_width) % bits;
        let current_bits = bit_width - remaining_bits;
        if next_word < bit_width {
            write!(
                output,
                r#"    tmp = (src >> {shift}) & MASK(uint{bits}_t, {current_bits});
    src = in[lane + LANE_COUNT * {next_word}];
    tmp |= (src & MASK(uint{bits}_t, {remaining_bits})) << {current_bits};
    out[INDEX({row}, lane)] = tmp + reference;
"#
            )
        } else {
            write!(
                output,
                r#"    tmp = (src >> {shift}) & MASK(uint{bits}_t, {current_bits});
    out[INDEX({row}, lane)] = tmp + reference;
"#
            )
        }
    } else {
        write!(
            output,
            r#"    tmp = (src >> {shift}) & MASK(uint{bits}_t, {bit_width});
    out[INDEX({row}, lane)] = tmp + reference;
"#
        )
    }
}

/// Emit a single lane-decoder template specialization.
///
/// For `bit_width == 0` and `bit_width == bits`, emits a simple `#pragma unroll`
/// loop.  For all other bit widths, emits pre-computed per-row bit extraction
/// with register-cached `src` words — identical to the original hand-unrolled
/// codegen, preserving minimal memory loads and zero extra work.
fn generate_lane_decoder(
    output: &mut impl Write,
    bits: usize,
    lanes: usize,
    bit_width: usize,
) -> io::Result<()> {
    if bit_width == 0 {
        write!(
            output,
            r#"template <>
__device__ void _bit_unpack_{bits}_lane<0>(const uint{bits}_t *__restrict in, uint{bits}_t *__restrict out, uint{bits}_t reference, unsigned int lane) {{
    #pragma unroll
    for (int row = 0; row < {bits}; row++) {{
        out[INDEX(row, lane)] = reference;
    }}
}}
"#
        )
    } else if bit_width == bits {
        write!(
            output,
            r#"template <>
__device__ void _bit_unpack_{bits}_lane<{bit_width}>(const uint{bits}_t *__restrict in, uint{bits}_t *__restrict out, uint{bits}_t reference, unsigned int lane) {{
    unsigned int LANE_COUNT = {lanes};
    #pragma unroll
    for (int row = 0; row < {bits}; row++) {{
        out[INDEX(row, lane)] = in[LANE_COUNT * row + lane] + reference;
    }}
}}
"#
        )
    } else {
        write!(
            output,
            r#"template <>
__device__ void _bit_unpack_{bits}_lane<{bit_width}>(const uint{bits}_t *__restrict in, uint{bits}_t *__restrict out, uint{bits}_t reference, unsigned int lane) {{
    unsigned int LANE_COUNT = {lanes};
    uint{bits}_t src;
    uint{bits}_t tmp;
    src = in[lane];
"#
        )?;

        for row in 0..bits {
            write_row(output, bits, bit_width, row)?;
        }

        writeln!(output, "}}")
    }
}

/// Generate a runtime dispatch function that routes to the appropriate
/// lane-decoder template specialization via a switch statement.
///
/// This is used by `dynamic_dispatch.cu` (via `bit_unpack.cuh`) where the
/// bit width is only known at runtime.
fn generate_lane_dispatch(output: &mut impl Write, bits: usize) -> io::Result<()> {
    write!(
        output,
        r#"/// Runtime dispatch to the optimized lane decoder for the given bit width.
__device__ __noinline__ void bit_unpack_{bits}_lane(
    const uint{bits}_t *__restrict in,
    uint{bits}_t *__restrict out,
    uint{bits}_t reference,
    unsigned int lane,
    uint32_t bit_width
) {{
    switch (bit_width) {{
"#
    )?;

    for bw in 0..=bits {
        writeln!(
            output,
            "        case {bw}: _bit_unpack_{bits}_lane<{bw}>(in, out, reference, lane); break;"
        )?;
    }

    write!(
        output,
        r#"    }}
}}
"#
    )
}

/// Emit the device kernel as a single template parameterized on bit width.
///
/// This is written once per element type rather than duplicated for every
/// bit width — the compiler instantiates it for each `BW` used by the
/// `extern "C"` global-kernel wrappers below.
fn generate_device_kernel_template(
    output: &mut impl Write,
    bits: usize,
    lanes: usize,
    thread_count: usize,
) -> io::Result<()> {
    let per_thread_loop_count = lanes / thread_count;
    let shared_copy_ncount = 1024 / thread_count;

    write!(
        output,
        r#"template <int BW>
__device__ void _bit_unpack_{bits}_device(const uint{bits}_t *__restrict in, uint{bits}_t *__restrict out, uint{bits}_t reference, int thread_idx, GPUPatches& patches) {{
    __shared__ uint{bits}_t shared_out[1024];

    // Step 1: Unpack into shared memory
    #pragma unroll
    for (int i = 0; i < {per_thread_loop_count}; i++) {{
        _bit_unpack_{bits}_lane<BW>(in, shared_out, reference, thread_idx * {per_thread_loop_count} + i);
    }}
    __syncwarp();

    // Step 2: Apply patches to shared memory in parallel
    PatchesCursor<uint{bits}_t> cursor(patches, blockIdx.x, thread_idx, {thread_count});
    auto patch = cursor.next();
    while (patch.index != 1024) {{
        shared_out[patch.index] = patch.value;
        patch = cursor.next();
    }}
    __syncwarp();

    // Step 3: Copy to global memory
    #pragma unroll
    for (int i = 0; i < {shared_copy_ncount}; i++) {{
        auto idx = i * {thread_count} + thread_idx;
        out[idx] = shared_out[idx];
    }}
}}
"#
    )
}

/// Emit a thin `extern "C"` global-kernel wrapper for a single bit width.
fn generate_global_kernel(
    output: &mut impl Write,
    bits: usize,
    bit_width: usize,
    thread_count: usize,
) -> io::Result<()> {
    let func_name = format!("bit_unpack_{bits}_{bit_width}bw_{thread_count}t");

    write!(
        output,
        r#"extern "C" __global__ void {func_name}(const uint{bits}_t *__restrict full_in, uint{bits}_t *__restrict full_out, uint{bits}_t reference, GPUPatches patches) {{
    int thread_idx = threadIdx.x;
    auto in = full_in + (blockIdx.x * (128 * {bit_width} / sizeof(uint{bits}_t)));
    auto out = full_out + (blockIdx.x * 1024);
    _bit_unpack_{bits}_device<{bit_width}>(in, out, reference, thread_idx, patches);
}}
"#
    )
}

/// Generate the lane-decoder header: template specializations + runtime dispatch.
///
/// This produces a `.cuh` file that is included by `bit_unpack.cuh` (and
/// transitively by `dynamic_dispatch.cu`). It contains only `__device__`
/// functions — no `__global__` kernels — so that `dynamic_dispatch.cu` does
/// not pull in the 129 standalone bit-unpack kernel entry points.
pub fn generate_cuda_unpack_lanes<T: FastLanes>(output: &mut impl Write) -> io::Result<()> {
    let bits = T::T;
    let lanes = T::LANES;

    write!(
        output,
        r#"// AUTO-GENERATED. Do not edit by hand!
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "fastlanes_common.cuh"

template <int BW>
__device__ void _bit_unpack_{bits}_lane(const uint{bits}_t *__restrict in, uint{bits}_t *__restrict out, uint{bits}_t reference, unsigned int lane);

"#
    )?;

    // Lane-decoder template specializations (one per bit width).
    for bit_width in 0..=bits {
        generate_lane_decoder(output, bits, lanes, bit_width)?;
        writeln!(output)?;
    }

    // Runtime dispatch function (used by dynamic_dispatch.cu via bit_unpack.cuh).
    generate_lane_dispatch(output, bits)?;
    writeln!(output)?;

    Ok(())
}

/// Generate the standalone kernel file: `_device` template + `__global__` wrappers.
///
/// This produces a `.cu` file that is compiled to PTX on its own. It includes
/// the corresponding `_lanes.cuh` header for the lane decoders.
pub fn generate_cuda_unpack_kernels<T: FastLanes>(
    output: &mut impl Write,
    thread_count: usize,
) -> io::Result<()> {
    let bits = T::T;
    let lanes = T::LANES;

    write!(
        output,
        r#"// AUTO-GENERATED. Do not edit by hand!
#include "bit_unpack_{bits}_lanes.cuh"
#include "patches.cuh"

"#
    )?;

    // Device kernel template (written once, instantiated per bit width).
    generate_device_kernel_template(output, bits, lanes, thread_count)?;
    writeln!(output)?;

    // Thin extern "C" global-kernel wrappers (one per bit width).
    for bit_width in 0..=bits {
        generate_global_kernel(output, bits, bit_width, thread_count)?;
        writeln!(output)?;
    }

    Ok(())
}
