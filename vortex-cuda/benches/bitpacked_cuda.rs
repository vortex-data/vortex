// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for BitPacked decompression.

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use std::mem::size_of;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC;
use futures::executor::block_on;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaDeviceBuffer;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda::launch_cuda_kernel_with_config;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_dtype::NativePType;
use vortex_error::VortexExpect;
use vortex_fastlanes::BitPackedArray;
use vortex_session::VortexSession;

const BENCH_ARGS: &[(usize, &str)] = &[(10_000_000, "10M")];

const BIT_WIDTH: u8 = 6;

/// Creates a BitPacked array with the specified type and length.
/// Values are chosen to fit in `BIT_WIDTH` bits so no patches are needed.
fn make_bitpacked_array<T>(len: usize) -> BitPackedArray
where
    T: NativePType + From<u8>,
{
    let max_val = (1u64 << BIT_WIDTH) - 1;
    let data: Vec<T> = (0..len)
        .map(|i| <T as From<u8>>::from((i as u64 % (max_val + 1)) as u8))
        .collect();

    let primitive_array =
        PrimitiveArray::new(Buffer::from(data), Validity::NonNullable).into_array();

    BitPackedArray::encode(primitive_array.as_ref(), BIT_WIDTH)
        .vortex_expect("failed to create BitPacked array")
}

/// Launches BitPacked decompression kernel and returns elapsed GPU time.
fn launch_bitpacked_kernel_timed<T>(
    bitpacked_array: &BitPackedArray,
    cuda_ctx: &mut CudaExecutionCtx,
) -> vortex_error::VortexResult<Duration>
where
    T: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let packed = bitpacked_array.packed().clone();
    let len = bitpacked_array.len();
    let bit_width = bitpacked_array.bit_width();

    // Copy packed data to device
    let device_input =
        block_on(cuda_ctx.move_to_device(packed)?).vortex_expect("failed to move to device");

    let input_view = device_input
        .cuda_view::<T>()
        .vortex_expect("failed to get input view");

    // Allocate output buffer
    let output_slice = cuda_ctx
        .device_alloc::<T>(len.next_multiple_of(1024))
        .vortex_expect("failed to allocate output");
    let output_buf = CudaDeviceBuffer::new(output_slice);
    let output_view = output_buf.as_view::<T>();

    // Load kernel function: bit_unpack_{bits}_{bit_width}bw_{thread_count}t
    let bits = size_of::<T>() * 8;
    let thread_count = if bits == 64 { 16u32 } else { 32u32 };
    let bw_suffix = format!("{bit_width}bw");
    let tc_suffix = format!("{thread_count}t");
    let cuda_function =
        cuda_ctx.load_function(&format!("bit_unpack_{bits}"), &[&bw_suffix, &tc_suffix])?;

    let mut launch_builder = cuda_ctx.launch_builder(&cuda_function);
    launch_builder.arg(&input_view);
    launch_builder.arg(&output_view);

    let num_blocks = u32::try_from(len.div_ceil(1024))?;
    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (thread_count, 1, 1),
        shared_mem_bytes: 0,
    };

    let events =
        launch_cuda_kernel_with_config(&mut launch_builder, config, CU_EVENT_BLOCKING_SYNC)?;

    events.duration()
}

/// Benchmark BitPacked decompression for a specific type.
fn benchmark_bitpacked_typed<T>(c: &mut Criterion, type_name: &str)
where
    T: NativePType + DeviceRepr + From<u8> + Send + Sync + 'static,
{
    let mut group = c.benchmark_group("bitpacked_cuda");
    group.sample_size(10);

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<T>()) as u64));

        let bitpacked_array = make_bitpacked_array::<T>(*len);

        group.bench_with_input(
            BenchmarkId::new("bitpacked", format!("{len_str}_{type_name}")),
            &bitpacked_array,
            |b, bitpacked_array| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let kernel_time =
                            launch_bitpacked_kernel_timed::<T>(bitpacked_array, &mut cuda_ctx)
                                .vortex_expect("kernel launch failed");
                        total_time += kernel_time;
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

/// Benchmark BitPacked decompression for all types.
fn benchmark_bitpacked(c: &mut Criterion) {
    benchmark_bitpacked_typed::<u8>(c, "u8");
    benchmark_bitpacked_typed::<u16>(c, "u16");
    benchmark_bitpacked_typed::<u32>(c, "u32");
    benchmark_bitpacked_typed::<u64>(c, "u64");
}

criterion::criterion_group!(benches, benchmark_bitpacked);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
