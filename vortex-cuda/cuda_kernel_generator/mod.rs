// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod indent;

use std::io;
use std::io::Write;

use fastlanes::FastLanes;
pub use indent::IndentedWriter;

fn generate_lane_decoder<T: FastLanes, W: Write>(
    output: &mut IndentedWriter<W>,
    bit_width: usize,
) -> io::Result<()> {
    let bits = <T>::T;
    let lanes = T::LANES;

    let func_name = format!("bit_unpack_{bits}_{bit_width}bw_lane");

    writeln!(
        output,
        "__device__ void _{func_name}(const uint{bits}_t *__restrict in, uint{bits}_t *__restrict out, uint{bits}_t reference, unsigned int lane) {{"
    )?;

    output.indent(|output| {
        writeln!(output, "unsigned int LANE_COUNT = {lanes};")?;
        if bit_width == 0 {
            writeln!(output)?;
            for row in 0..bits {
                writeln!(output, "out[INDEX({row}, lane)] = reference;")?;
            }
        } else if bit_width == bits {
            writeln!(output)?;
            for row in 0..bits {
                writeln!(
                    output,
                    "out[INDEX({row}, lane)] = in[LANE_COUNT * {row} + lane] + reference;",
                )?;
            }
        } else {
            writeln!(output, "uint{bits}_t src;")?;
            writeln!(output, "uint{bits}_t tmp;")?;

            writeln!(output)?;
            writeln!(output, "src = in[lane];")?;
            for row in 0..bits {
                let curr_word = (row * bit_width) / bits;
                let next_word = ((row + 1) * bit_width) / bits;
                let shift = (row * bit_width) % bits;

                if next_word > curr_word {
                    let remaining_bits = ((row + 1) * bit_width) % bits;
                    let current_bits = bit_width - remaining_bits;
                    writeln!(
                        output,
                        "tmp = (src >> {shift}) & MASK(uint{bits}_t, {current_bits});"
                    )?;

                    if next_word < bit_width {
                        writeln!(output, "src = in[lane + LANE_COUNT * {next_word}];")?;
                        writeln!(
                            output,
                            "tmp |= (src & MASK(uint{bits}_t, {remaining_bits})) << {current_bits};"
                        )?;
                    }
                } else {
                    writeln!(
                        output,
                        "tmp = (src >> {shift}) & MASK(uint{bits}_t, {bit_width});"
                    )?;
                }

                writeln!(output, "out[INDEX({row}, lane)] = tmp + reference;")?;
            }
        }
        Ok(())
    })?;

    writeln!(output, "}}")
}

/// Generate a runtime dispatch function that routes
/// to the appropriate bit-width-specific lane decoder.
fn generate_lane_dispatch<T: FastLanes, W: Write>(
    output: &mut IndentedWriter<W>,
) -> io::Result<()> {
    let bits = <T>::T;

    writeln!(
        output,
        "/// Runtime dispatch to the optimized lane decoder for the given bit width."
    )?;
    writeln!(output, "__device__ inline void bit_unpack_{bits}_lane(")?;
    writeln!(output, "    const uint{bits}_t *__restrict in,")?;
    writeln!(output, "    uint{bits}_t *__restrict out,")?;
    writeln!(output, "    uint{bits}_t reference,")?;
    writeln!(output, "    unsigned int lane,")?;
    writeln!(output, "    uint32_t bit_width")?;
    writeln!(output, ") {{")?;

    output.indent(|output| {
        writeln!(output, "switch (bit_width) {{")?;
        output.indent(|output| {
            for bw in 0..=bits {
                writeln!(
                    output,
                    "case {bw}: _bit_unpack_{bits}_{bw}bw_lane(in, out, reference, lane); break;"
                )?;
            }
            Ok(())
        })?;
        writeln!(output, "}}")
    })?;

    writeln!(output, "}}")
}

fn generate_device_kernel_for_width<T: FastLanes, W: Write>(
    output: &mut IndentedWriter<W>,
    bit_width: usize,
    thread_count: usize,
) -> io::Result<()> {
    let bits = <T>::T;
    let lanes = T::LANES;
    let per_thread_loop_count = lanes / thread_count;
    let shared_copy_ncount = 1024 / thread_count;

    let func_name = format!("bit_unpack_{bits}_{bit_width}bw_{thread_count}t");

    let local_func_params = format!(
        "(const uint{bits}_t *__restrict in, uint{bits}_t *__restrict out, uint{bits}_t reference, int thread_idx)"
    );

    writeln!(output, "__device__ void _{func_name}{local_func_params} {{")?;

    output.indent(|output| {
        writeln!(output, "__shared__ uint{bits}_t shared_out[1024];")?;

        for thread_lane in 0..per_thread_loop_count {
            writeln!(output, "_bit_unpack_{bits}_{bit_width}bw_lane(in, shared_out, reference, thread_idx * {per_thread_loop_count} + {thread_lane});")?;
        }

        writeln!(output, "for (int i = 0; i < {shared_copy_ncount}; i++) {{")?;
        output.indent(|output| {
            writeln!(output, "auto idx = i * {thread_count} + thread_idx;")?;
            writeln!(output, "out[idx] = shared_out[idx];")
        })?;
        writeln!(output, "}}")
    })?;

    writeln!(output, "}}")
}

fn generate_global_kernel_for_width<T: FastLanes, W: Write>(
    output: &mut IndentedWriter<W>,
    bit_width: usize,
    thread_count: usize,
) -> io::Result<()> {
    let bits = <T>::T;

    let func_name = format!("bit_unpack_{bits}_{bit_width}bw_{thread_count}t");
    let func_params = format!(
        "(const uint{bits}_t *__restrict full_in, uint{bits}_t *__restrict full_out, uint{bits}_t reference)"
    );

    writeln!(
        output,
        "extern \"C\" __global__ void {func_name}{func_params} {{"
    )?;

    output.indent(|output| {
        writeln!(output, "int thread_idx = threadIdx.x;")?;
        writeln!(
            output,
            "auto in = full_in + (blockIdx.x * (128 * {bit_width} / sizeof(uint{bits}_t)));"
        )?;
        writeln!(output, "auto out = full_out + (blockIdx.x * 1024);")?;

        writeln!(output, "_{func_name}(in, out, reference, thread_idx);")
    })?;

    writeln!(output, "}}")
}

/// Generate CUDA lane decoders, dispatch function, and kernel wrappers for all bit widths.
pub fn generate_cuda_unpack_for_width<T: FastLanes, W: Write>(
    output: &mut IndentedWriter<W>,
    thread_count: usize,
) -> io::Result<()> {
    let bits = <T>::T;

    writeln!(output, "// AUTO-GENERATED. Do not edit by hand!")?;
    writeln!(output, "#include <cuda.h>")?;
    writeln!(output, "#include <cuda_runtime.h>")?;
    writeln!(output, "#include <stdint.h>")?;
    writeln!(output, "#include \"fastlanes_common.cuh\"")?;
    writeln!(output)?;

    // First, emit all lane decoders.
    for bit_width in 0..=bits {
        generate_lane_decoder::<T, _>(output, bit_width)?;
        writeln!(output)?;
    }

    // Emit the runtime lane dispatch function (used by dynamic_dispatch.cu).
    generate_lane_dispatch::<T, _>(output)?;
    writeln!(output)?;

    // Emit device and global kernel wrappers.
    for bit_width in 0..=bits {
        generate_device_kernel_for_width::<T, _>(output, bit_width, thread_count)?;
        writeln!(output)?;

        generate_global_kernel_for_width::<T, _>(output, bit_width, thread_count)?;
        writeln!(output)?;
    }

    Ok(())
}
