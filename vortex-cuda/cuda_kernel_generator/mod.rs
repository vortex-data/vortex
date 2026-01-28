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
        "__device__ void _{func_name}(const uint{bits}_t *__restrict in, uint{bits}_t *__restrict out, unsigned int lane) {{"
    )?;

    output.indent(|output| {
        writeln!(output, "unsigned int LANE_COUNT = {lanes};")?;
        if bit_width == 0 {
            writeln!(output, "uint{bits}_t zero = 0ULL;")?;
            writeln!(output)?;
            for row in 0..bits {
                writeln!(output, "out[INDEX({row}, lane)] = zero;")?;
            }
        } else if bit_width == bits {
            writeln!(output)?;
            for row in 0..bits {
                writeln!(
                    output,
                    "out[INDEX({row}, lane)] = in[LANE_COUNT * {row} + lane];",
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

                writeln!(output, "out[INDEX({row}, lane)] = tmp;")?;
            }
        }
        Ok(())
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

    let func_name = format!("bit_unpack_{bits}_{bit_width}bw_{thread_count}t");

    let local_func_params = format!(
        "(const uint{bits}_t *__restrict in, uint{bits}_t *__restrict out, int thread_idx)"
    );

    writeln!(output, "__device__ void _{func_name}{local_func_params} {{")?;

    output.indent(|output| {
        for thread_lane in 0..per_thread_loop_count {
            writeln!(output, "_bit_unpack_{bits}_{bit_width}bw_lane(in, out, thread_idx * {per_thread_loop_count} + {thread_lane});")?;
        }
        Ok(())
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
    let func_params =
        format!("(const uint{bits}_t *__restrict full_in, uint{bits}_t *__restrict full_out)");

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

        writeln!(output, "_{func_name}(in, out, thread_idx);")
    })?;

    writeln!(output, "}}")
}

/// # Errors
///
/// Will return Err if writing to the underlying writer fails.
pub fn generate_cuda_unpack_for_width<T: FastLanes, W: Write>(
    output: &mut IndentedWriter<W>,
    thread_count: usize,
) -> io::Result<()> {
    writeln!(output, "// AUTO-GENERATED. Do not edit by hand!")?;
    writeln!(output, "#include <cuda.h>")?;
    writeln!(output, "#include <cuda_runtime.h>")?;
    writeln!(output, "#include <stdint.h>")?;
    writeln!(output, "#include \"fastlanes_common.cuh\"")?;
    writeln!(output)?;

    for bit_width in 0..=<T>::T {
        generate_lane_decoder::<T, _>(output, bit_width)?;
        writeln!(output)?;
        generate_device_kernel_for_width::<T, _>(output, bit_width, thread_count)?;
        writeln!(output)?;

        generate_global_kernel_for_width::<T, _>(output, bit_width, thread_count)?;
        writeln!(output)?;
    }

    Ok(())
}
