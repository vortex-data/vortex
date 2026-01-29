// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for FoR decompression.

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use std::mem::size_of;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::CudaView;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC;
use futures::executor::block_on;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_buffer::Buffer;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_error::VortexExpect;
use vortex_fastlanes::FoRArray;
use vortex_session::VortexSession;

const BENCH_ARGS: &[(usize, &str)] = &[(10_000_000, "10M")];

/// Creates a FoR array of u8 for the given size.
fn make_for_array_u8(len: usize) -> FoRArray {
    let data: Vec<u8> = (0..len as u8).map(|i| i.wrapping_add(10)).collect();
    let primitive_array = PrimitiveArray::new(
        Buffer::from(data),
        vortex_array::validity::Validity::NonNullable,
    )
    .into_array();

    FoRArray::try_new(primitive_array, 10u8.into()).vortex_expect("failed to create FoR array")
}

/// Creates a FoR array of u16 for the given size.
fn make_for_array_u16(len: usize) -> FoRArray {
    let data: Vec<u16> = (0..len as u16).map(|i| i.wrapping_add(10)).collect();
    let primitive_array = PrimitiveArray::new(
        Buffer::from(data),
        vortex_array::validity::Validity::NonNullable,
    )
    .into_array();

    FoRArray::try_new(primitive_array, 10u16.into()).vortex_expect("failed to create FoR array")
}

/// Creates a FoR array of u32 for the given size.
fn make_for_array_u32(len: usize) -> FoRArray {
    let primitive_array = PrimitiveArray::new(
        Buffer::from((0u32..len as u32).collect::<Vec<u32>>()),
        vortex_array::validity::Validity::NonNullable,
    )
    .into_array();

    FoRArray::try_new(primitive_array, 10u32.into()).vortex_expect("failed to create FoR array")
}

/// Creates a FoR array of u64 for the given size.
fn make_for_array_u64(len: usize) -> FoRArray {
    let data: Vec<u64> = (0..len as u64).map(|i| i.wrapping_add(10)).collect();
    let primitive_array = PrimitiveArray::new(
        Buffer::from(data),
        vortex_array::validity::Validity::NonNullable,
    )
    .into_array();

    FoRArray::try_new(primitive_array, 10u64.into()).vortex_expect("failed to create FoR array")
}

/// Launches FoR decompression kernel and returns elapsed GPU time in seconds.
fn launch_for_kernel_timed_u8(
    for_array: &FoRArray,
    device_data: CudaView<'_, u8>,
    reference: u8,
    cuda_ctx: &mut CudaExecutionCtx,
) -> vortex_error::VortexResult<Duration> {
    let array_len_u64 = for_array.len() as u64;

    let events = vortex_cuda::launch_cuda_kernel!(
        execution_ctx: cuda_ctx,
        module: "for",
        ptypes: &[for_array.ptype().to_string().as_str()],
        launch_args: [device_data, reference, array_len_u64],
        event_recording: CU_EVENT_BLOCKING_SYNC,
        array_len: for_array.len()
    );

    events.duration()
}

/// Launches FoR decompression kernel and returns elapsed GPU time in seconds.
fn launch_for_kernel_timed_u16(
    for_array: &FoRArray,
    device_data: CudaView<'_, u16>,
    reference: u16,
    cuda_ctx: &mut CudaExecutionCtx,
) -> vortex_error::VortexResult<Duration> {
    let array_len_u64 = for_array.len() as u64;

    let events = vortex_cuda::launch_cuda_kernel!(
        execution_ctx: cuda_ctx,
        module: "for",
        ptypes: &[for_array.ptype().to_string().as_str()],
        launch_args: [device_data, reference, array_len_u64],
        event_recording: CU_EVENT_BLOCKING_SYNC,
        array_len: for_array.len()
    );

    events.duration()
}

/// Launches FoR decompression kernel and returns elapsed GPU time in seconds.
fn launch_for_kernel_timed_u32(
    for_array: &FoRArray,
    device_data: CudaView<'_, u32>,
    reference: u32,
    cuda_ctx: &mut CudaExecutionCtx,
) -> vortex_error::VortexResult<Duration> {
    let array_len_u64 = for_array.len() as u64;

    let events = vortex_cuda::launch_cuda_kernel!(
        execution_ctx: cuda_ctx,
        module: "for",
        ptypes: &[for_array.ptype().to_string().as_str()],
        launch_args: [device_data, reference, array_len_u64],
        event_recording: CU_EVENT_BLOCKING_SYNC,
        array_len: for_array.len()
    );

    events.duration()
}

/// Launches FoR decompression kernel and returns elapsed GPU time in seconds.
fn launch_for_kernel_timed_u64(
    for_array: &FoRArray,
    device_data: CudaView<'_, u64>,
    reference: u64,
    cuda_ctx: &mut CudaExecutionCtx,
) -> vortex_error::VortexResult<Duration> {
    let array_len_u64 = for_array.len() as u64;

    let events = vortex_cuda::launch_cuda_kernel!(
        execution_ctx: cuda_ctx,
        module: "for",
        ptypes: &[for_array.ptype().to_string().as_str()],
        launch_args: [device_data, reference, array_len_u64],
        event_recording: CU_EVENT_BLOCKING_SYNC,
        array_len: for_array.len()
    );

    events.duration()
}

/// Benchmark u8 FoR decompression
fn benchmark_for_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("FoR_cuda_u8");
    group.sample_size(10);

    for (len, label) in BENCH_ARGS {
        let for_array = make_for_array_u8(*len);

        group.throughput(Throughput::Bytes((len * size_of::<u8>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("u8_FoR", label),
            &for_array,
            |b, for_array| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    let encoded = for_array.encoded();
                    let unpacked_array = encoded.to_primitive();
                    let unpacked_slice = unpacked_array.as_slice::<u8>();

                    let reference = 10u8;
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let device_data =
                            block_on(cuda_ctx.copy_to_device(unpacked_slice.to_vec()).unwrap())
                                .vortex_expect("failed to copy to device");

                        let kernel_time = launch_for_kernel_timed_u8(
                            for_array,
                            device_data
                                .cuda_view::<u8>()
                                .vortex_expect("failed to get device view"),
                            reference,
                            &mut cuda_ctx,
                        )
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

/// Benchmark u16 FoR decompression
fn benchmark_for_u16(c: &mut Criterion) {
    let mut group = c.benchmark_group("FoR_cuda_u16");
    group.sample_size(10);

    for (len, label) in BENCH_ARGS {
        let for_array = make_for_array_u16(*len);

        group.throughput(Throughput::Bytes((len * size_of::<u16>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("u16_FoR", label),
            &for_array,
            |b, for_array| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    let encoded = for_array.encoded();
                    let unpacked_array = encoded.to_primitive();
                    let unpacked_slice = unpacked_array.as_slice::<u16>();

                    let reference = 10u16;
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let device_data =
                            block_on(cuda_ctx.copy_to_device(unpacked_slice.to_vec()).unwrap())
                                .vortex_expect("failed to copy to device");

                        let kernel_time = launch_for_kernel_timed_u16(
                            for_array,
                            device_data
                                .cuda_view::<u16>()
                                .vortex_expect("failed to get device view"),
                            reference,
                            &mut cuda_ctx,
                        )
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

/// Benchmark u32 FoR decompression
fn benchmark_for_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("FoR_cuda_u32");
    group.sample_size(10);

    for (len, label) in BENCH_ARGS {
        let for_array = make_for_array_u32(*len);

        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("u32_FoR", label),
            &for_array,
            |b, for_array| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    let encoded = for_array.encoded();
                    let unpacked_array = encoded.to_primitive();
                    let unpacked_slice = unpacked_array.as_slice::<u32>();

                    let reference = 10u32;
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let device_data =
                            block_on(cuda_ctx.copy_to_device(unpacked_slice.to_vec()).unwrap())
                                .vortex_expect("failed to copy to device");

                        let kernel_time = launch_for_kernel_timed_u32(
                            for_array,
                            device_data
                                .cuda_view::<u32>()
                                .vortex_expect("failed to get device view"),
                            reference,
                            &mut cuda_ctx,
                        )
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

/// Benchmark u64 FoR decompression
fn benchmark_for_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("FoR_cuda_u64");
    group.sample_size(10);

    for (len, label) in BENCH_ARGS {
        let for_array = make_for_array_u64(*len);

        group.throughput(Throughput::Bytes((len * size_of::<u64>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("u64_FoR", label),
            &for_array,
            |b, for_array| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    let encoded = for_array.encoded();
                    let unpacked_array = encoded.to_primitive();
                    let unpacked_slice = unpacked_array.as_slice::<u64>();

                    let reference = 10u64;
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let device_data =
                            block_on(cuda_ctx.copy_to_device(unpacked_slice.to_vec()).unwrap())
                                .vortex_expect("failed to copy to device");

                        let kernel_time = launch_for_kernel_timed_u64(
                            for_array,
                            device_data.cuda_view::<u64>().unwrap(),
                            reference,
                            &mut cuda_ctx,
                        )
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

pub fn benchmark_for_cuda(c: &mut Criterion) {
    benchmark_for_u8(c);
    benchmark_for_u16(c);
    benchmark_for_u32(c);
    benchmark_for_u64(c);
}

criterion::criterion_group!(benches, benchmark_for_cuda);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
