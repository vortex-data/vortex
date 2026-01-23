// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity::NonNullable;
use vortex_buffer::Buffer;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaDeviceBuffer;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_dtype::PType;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

const BENCH_ARGS: &[(usize, &str)] = &[(10_000_000, "10M")];

/// Creates a Dict array with u32 values and u8 codes for the given size.
fn make_dict_array_u32_u8(len: usize) -> DictArray {
    // Dictionary with 256 values
    let values: Vec<u32> = (0..256).map(|i| i * 1000).collect();
    let values_array = PrimitiveArray::new(Buffer::from(values), NonNullable);

    // Codes cycling through all dictionary values
    let codes: Vec<u8> = (0..len).map(|i| (i % 256) as u8).collect();
    let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

    DictArray::try_new(codes_array.into_array(), values_array.into_array())
        .vortex_expect("failed to create Dict array")
}

/// Creates a Dict array with u32 values and u16 codes for the given size.
fn make_dict_array_u32_u16(len: usize) -> DictArray {
    // Dictionary with 4096 values
    let values: Vec<u32> = (0..4096).map(|i| i * 100).collect();
    let values_array = PrimitiveArray::new(Buffer::from(values), NonNullable);

    // Codes cycling through all dictionary values
    let codes: Vec<u16> = (0..len).map(|i| (i % 4096) as u16).collect();
    let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

    DictArray::try_new(codes_array.into_array(), values_array.into_array())
        .vortex_expect("failed to create Dict array")
}

/// Creates a Dict array with u64 values and u8 codes for the given size.
fn make_dict_array_u64_u8(len: usize) -> DictArray {
    // Dictionary with 256 values
    let values: Vec<u64> = (0..256).map(|i| i * 1_000_000).collect();
    let values_array = PrimitiveArray::new(Buffer::from(values), NonNullable);

    // Codes cycling through all dictionary values
    let codes: Vec<u8> = (0..len).map(|i| (i % 256) as u8).collect();
    let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

    DictArray::try_new(codes_array.into_array(), values_array.into_array())
        .vortex_expect("failed to create Dict array")
}

/// Creates a Dict array with u64 values and u32 codes for the given size.
fn make_dict_array_u64_u32(len: usize) -> DictArray {
    // Dictionary with 65536 values
    let values: Vec<u64> = (0..65536).map(|i| i * 1000).collect();
    let values_array = PrimitiveArray::new(Buffer::from(values), NonNullable);

    // Codes cycling through all dictionary values
    let codes: Vec<u32> = (0..len).map(|i| (i % 65536) as u32).collect();
    let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

    DictArray::try_new(codes_array.into_array(), values_array.into_array())
        .vortex_expect("failed to create Dict array")
}

/// Launches Dict decompression kernel and returns elapsed GPU time.
fn launch_dict_kernel_timed<V: cudarc::driver::DeviceRepr, I: cudarc::driver::DeviceRepr>(
    codes_view: CudaView<'_, I>,
    codes_len: usize,
    values_view: CudaView<'_, V>,
    output_view: CudaView<'_, V>,
    value_ptype: PType,
    code_ptype: PType,
    cuda_ctx: &mut CudaExecutionCtx,
) -> vortex_error::VortexResult<Duration> {
    let codes_len_u64 = codes_len as u64;

    let events = vortex_cuda::launch_cuda_kernel!(
        execution_ctx: cuda_ctx,
        module: "dict",
        ptypes: &[value_ptype.to_string().as_str(), code_ptype.to_string().as_str()],
        launch_args: [codes_view, codes_len_u64, values_view, output_view],
        event_recording: CU_EVENT_BLOCKING_SYNC,
        array_len: codes_len
    );

    events.duration()
}

/// Benchmark u32 values with u8 codes
fn benchmark_dict_u32_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dict_cuda_u32_u8");
    group.sample_size(10);

    for (len, label) in BENCH_ARGS {
        let dict_array = make_dict_array_u32_u8(*len);

        // Throughput is based on output size (values read from dictionary)
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("u32_values_u8_codes", label),
            &dict_array,
            |b, dict_array| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    // Get values and codes arrays
                    let values: Vec<u32> = (0..256).map(|i| i * 1000).collect();
                    let codes: Vec<u8> = (0..*len).map(|i| (i % 256) as u8).collect();

                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let values_device =
                            block_on(cuda_ctx.copy_to_device(values.clone()).unwrap())
                                .vortex_expect("failed to copy values to device");

                        let codes_device =
                            block_on(cuda_ctx.copy_to_device(codes.clone()).unwrap())
                                .vortex_expect("failed to copy codes to device");

                        let output_slice = cuda_ctx
                            .device_alloc::<u32>(dict_array.len())
                            .vortex_expect("failed to allocate output");
                        let output_device = CudaDeviceBuffer::new(output_slice);

                        let kernel_time = launch_dict_kernel_timed(
                            codes_device
                                .cuda_view::<u8>()
                                .vortex_expect("failed to get codes view"),
                            dict_array.len(),
                            values_device
                                .cuda_view::<u32>()
                                .vortex_expect("failed to get values view"),
                            output_device.as_view(),
                            PType::U32,
                            PType::U8,
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

/// Benchmark u32 values with u16 codes
fn benchmark_dict_u32_u16(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dict_cuda_u32_u16");
    group.sample_size(10);

    for (len, label) in BENCH_ARGS {
        let dict_array = make_dict_array_u32_u16(*len);

        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("u32_values_u16_codes", label),
            &dict_array,
            |b, dict_array| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    let values: Vec<u32> = (0..4096).map(|i| i * 100).collect();
                    let codes: Vec<u16> = (0..*len).map(|i| (i % 4096) as u16).collect();

                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let values_device =
                            block_on(cuda_ctx.copy_to_device(values.clone()).unwrap())
                                .vortex_expect("failed to copy values to device");

                        let codes_device =
                            block_on(cuda_ctx.copy_to_device(codes.clone()).unwrap())
                                .vortex_expect("failed to copy codes to device");

                        let output_slice = cuda_ctx
                            .device_alloc::<u32>(dict_array.len())
                            .vortex_expect("failed to allocate output");
                        let output_device = CudaDeviceBuffer::new(output_slice);

                        let kernel_time = launch_dict_kernel_timed(
                            codes_device
                                .cuda_view::<u16>()
                                .vortex_expect("failed to get codes view"),
                            dict_array.len(),
                            values_device
                                .cuda_view::<u32>()
                                .vortex_expect("failed to get values view"),
                            output_device.as_view(),
                            PType::U32,
                            PType::U16,
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

/// Benchmark u64 values with u8 codes
fn benchmark_dict_u64_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dict_cuda_u64_u8");
    group.sample_size(10);

    for (len, label) in BENCH_ARGS {
        let dict_array = make_dict_array_u64_u8(*len);

        group.throughput(Throughput::Bytes((len * size_of::<u64>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("u64_values_u8_codes", label),
            &dict_array,
            |b, dict_array| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    let values: Vec<u64> = (0..256).map(|i| i * 1_000_000).collect();
                    let codes: Vec<u8> = (0..*len).map(|i| (i % 256) as u8).collect();

                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let values_device =
                            block_on(cuda_ctx.copy_to_device(values.clone()).unwrap())
                                .vortex_expect("failed to copy values to device");

                        let codes_device =
                            block_on(cuda_ctx.copy_to_device(codes.clone()).unwrap())
                                .vortex_expect("failed to copy codes to device");

                        let output_slice = cuda_ctx
                            .device_alloc::<u64>(dict_array.len())
                            .vortex_expect("failed to allocate output");
                        let output_device = CudaDeviceBuffer::new(output_slice);

                        let kernel_time = launch_dict_kernel_timed(
                            codes_device
                                .cuda_view::<u8>()
                                .vortex_expect("failed to get codes view"),
                            dict_array.len(),
                            values_device
                                .cuda_view::<u64>()
                                .vortex_expect("failed to get values view"),
                            output_device.as_view(),
                            PType::U64,
                            PType::U8,
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

/// Benchmark u64 values with u32 codes
fn benchmark_dict_u64_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dict_cuda_u64_u32");
    group.sample_size(10);

    for (len, label) in BENCH_ARGS {
        let dict_array = make_dict_array_u64_u32(*len);

        group.throughput(Throughput::Bytes((len * size_of::<u64>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("u64_values_u32_codes", label),
            &dict_array,
            |b, dict_array| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    let values: Vec<u64> = (0..65536).map(|i| i * 1000).collect();
                    let codes: Vec<u32> = (0..*len).map(|i| (i % 65536) as u32).collect();

                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let values_device =
                            block_on(cuda_ctx.copy_to_device(values.clone()).unwrap())
                                .vortex_expect("failed to copy values to device");

                        let codes_device =
                            block_on(cuda_ctx.copy_to_device(codes.clone()).unwrap())
                                .vortex_expect("failed to copy codes to device");

                        let output_slice = cuda_ctx
                            .device_alloc::<u64>(dict_array.len())
                            .vortex_expect("failed to allocate output");
                        let output_device = CudaDeviceBuffer::new(output_slice);

                        let kernel_time = launch_dict_kernel_timed(
                            codes_device
                                .cuda_view::<u32>()
                                .vortex_expect("failed to get codes view"),
                            dict_array.len(),
                            values_device
                                .cuda_view::<u64>()
                                .vortex_expect("failed to get values view"),
                            output_device.as_view(),
                            PType::U64,
                            PType::U32,
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

pub fn benchmark_dict_cuda(c: &mut Criterion) {
    benchmark_dict_u32_u8(c);
    benchmark_dict_u32_u16(c);
    benchmark_dict_u64_u8(c);
    benchmark_dict_u64_u32(c);
}

criterion::criterion_group!(benches, benchmark_dict_cuda);

#[cfg(cuda_available)]
criterion::criterion_main!(benches);

#[cfg(not(cuda_available))]
fn main() {}
