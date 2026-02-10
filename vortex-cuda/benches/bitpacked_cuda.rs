// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for bit unpacking.

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use std::mem::size_of;
use std::ops::Add;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING;
use futures::executor::block_on;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity::NonNullable;
use vortex_buffer::Buffer;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaDeviceBuffer;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda::bitpacked_cuda_kernel;
use vortex_cuda::bitpacked_cuda_launch_config;
use vortex_cuda::launch_cuda_kernel_with_config;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_dtype::NativePType;
use vortex_error::VortexExpect;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::unpack_iter::BitPacked;
use vortex_session::VortexSession;

const N_ROWS: usize = 100_000_000;

/// Create a bit-packed array with the given bit width
fn make_bitpacked_array<T>(bit_width: u8, len: usize) -> BitPackedArray
where
    T: NativePType + Add<Output = T> + From<u8>,
{
    let max_val = (1u64 << bit_width).saturating_sub(1);

    let values: Vec<T> = (0..len)
        .map(|i| {
            let val = ((i as u64 % 256) & max_val) as u8;
            <T as From<u8>>::from(val)
        })
        .collect();

    let primitive_array = PrimitiveArray::new(Buffer::from(values), NonNullable);
    BitPackedArray::encode(primitive_array.as_ref(), bit_width)
        .vortex_expect("failed to create BitPacked array")
}

/// Launch the bit unpacking kernel and return elapsed GPU time
fn launch_bitunpack_kernel_timed_typed<T>(
    bitpacked_array: &BitPackedArray,
    cuda_ctx: &mut CudaExecutionCtx,
) -> vortex_error::VortexResult<Duration>
where
    T: BitPacked + DeviceRepr,
    T::Physical: DeviceRepr,
{
    let packed = bitpacked_array.packed().clone();
    let bit_width = bitpacked_array.bit_width();
    let len = bitpacked_array.len();

    // Move packed data to device if not already there
    let device_input = if packed.is_on_device() {
        packed
    } else {
        block_on(cuda_ctx.move_to_device(packed)?).vortex_expect("failed to move to device")
    };

    // Allocate output buffer
    let output_slice = cuda_ctx
        .device_alloc::<T>(len.next_multiple_of(1024))
        .vortex_expect("failed to allocate output");
    let output_buf = CudaDeviceBuffer::new(output_slice);

    // Get device views
    let input_view = device_input
        .cuda_view::<T::Physical>()
        .vortex_expect("failed to get input view");
    let output_view = output_buf.as_view::<T>();

    let output_width = size_of::<T>() * 8;
    let cuda_function = bitpacked_cuda_kernel(bit_width, output_width, cuda_ctx)?;
    let mut launch_builder = cuda_ctx.launch_builder(&cuda_function);

    launch_builder.arg(&input_view);
    launch_builder.arg(&output_view);

    let config = bitpacked_cuda_launch_config(output_width, len)?;

    // Launch kernel
    let events =
        launch_cuda_kernel_with_config(&mut launch_builder, config, CU_EVENT_BLOCKING_SYNC)?;

    events.duration()
}

/// Generic benchmark function for a specific type and bit width
fn benchmark_bitunpack_typed<T>(c: &mut Criterion, bit_width: u8, type_name: &str)
where
    T: BitPacked + NativePType + DeviceRepr + Add<Output = T> + From<u8>,
    T::Physical: DeviceRepr,
{
    let mut group = c.benchmark_group(format!("bitunpack_cuda_{}", type_name));
    group.sample_size(10);

    let array = make_bitpacked_array::<T>(bit_width, N_ROWS);
    let nbytes = (N_ROWS * bit_width as usize).div_ceil(8);

    group.throughput(Throughput::Bytes(nbytes as u64));

    group.bench_with_input(
        BenchmarkId::new("bitunpack", format!("{}bw", bit_width)),
        &array,
        |b, array| {
            b.iter_custom(|iters| {
                let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                    .vortex_expect("failed to create execution context");

                let mut total_time = Duration::ZERO;

                for _ in 0..iters {
                    let kernel_time =
                        launch_bitunpack_kernel_timed_typed::<T>(array, &mut cuda_ctx)
                            .vortex_expect("kernel launch failed");
                    total_time += kernel_time;
                }

                total_time
            });
        },
    );

    group.finish();
}

/// Benchmark bit unpacking for u8
fn benchmark_bitunpack_u8(c: &mut Criterion) {
    // Benchmark all meaningful bit widths for u8 (1-8)
    for bit_width in 1..=8 {
        benchmark_bitunpack_typed::<u8>(c, bit_width, "u8");
    }
}

/// Benchmark bit unpacking for u16
fn benchmark_bitunpack_u16(c: &mut Criterion) {
    // Benchmark selected bit widths for u16
    for bit_width in [1, 2, 4, 8, 12, 16] {
        benchmark_bitunpack_typed::<u16>(c, bit_width, "u16");
    }
}

/// Benchmark bit unpacking for u32
fn benchmark_bitunpack_u32(c: &mut Criterion) {
    // Benchmark selected bit widths for u32
    for bit_width in [1, 2, 4, 8, 12, 16, 20, 24, 28, 32] {
        benchmark_bitunpack_typed::<u32>(c, bit_width, "u32");
    }
}

/// Benchmark bit unpacking for u64
fn benchmark_bitunpack_u64(c: &mut Criterion) {
    // Benchmark selected bit widths for u64
    for bit_width in [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64] {
        benchmark_bitunpack_typed::<u64>(c, bit_width, "u64");
    }
}

/// Benchmark all bit unpacking operations
fn benchmark_bitunpack(c: &mut Criterion) {
    benchmark_bitunpack_u8(c);
    benchmark_bitunpack_u16(c);
    benchmark_bitunpack_u32(c);
    benchmark_bitunpack_u64(c);
}

criterion::criterion_group!(benches, benchmark_bitunpack);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
