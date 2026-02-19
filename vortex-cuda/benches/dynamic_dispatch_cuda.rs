// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::expect_used)]

use std::mem::size_of;
use std::sync::Arc;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::DevicePtr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags;
use futures::executor::block_on;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity::NonNullable;
use vortex_buffer::Buffer;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaDeviceBuffer;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda::dynamic_dispatch::DynamicDispatchPlan;
use vortex_cuda::dynamic_dispatch::ScalarOp;
use vortex_cuda::dynamic_dispatch::SourceOp;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_fastlanes::BitPackedArray;
use vortex_session::VortexSession;

const BENCH_ARGS: &[(usize, &str)] = &[
    (1_000_000, "1M"),
    (10_000_000, "10M"),
    (100_000_000, "100M"),
];

const REFERENCE_VALUE: u32 = 100_000;

/// Bit width used for the bitpack+FoR benchmarks.
const BIT_WIDTH: u8 = 6;

/// ALP decode factors for the ALP benchmarks.
const ALP_F: f32 = 10.0;
const ALP_E: f32 = 1.0;

/// Create a BitPackedArray of u32 values with the given bit width and length.
fn make_bitpacked_array_u32(bit_width: u8, len: usize) -> BitPackedArray {
    let max_val = (1u64 << bit_width).saturating_sub(1);
    let values: Vec<u32> = (0..len)
        .map(|i| (i as u64 % (max_val + 1)) as u32)
        .collect();
    let primitive = PrimitiveArray::new(Buffer::from(values), NonNullable);
    BitPackedArray::encode(primitive.as_ref(), bit_width)
        .vortex_expect("failed to create BitPacked array")
}

/// Launch the dynamic_dispatch kernel and return GPU-timed duration.
fn run_dynamic_dispatch_timed(
    cuda_ctx: &mut CudaExecutionCtx,
    input_ptr: u64,
    output_ptr: u64,
    array_len: usize,
    device_plan: &Arc<cudarc::driver::CudaSlice<DynamicDispatchPlan>>,
) -> VortexResult<Duration> {
    let cuda_function = cuda_ctx.load_function("dynamic_dispatch", &["u32"])?;
    let array_len_u64 = array_len as u64;
    let plan_ptr = device_plan.device_ptr(cuda_ctx.stream()).0;

    let stream = cuda_ctx.stream();
    let ctx = stream.context();
    let start_event = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| vortex_err!("failed to create start event: {:?}", e))?;
    start_event
        .record(stream)
        .map_err(|e| vortex_err!("failed to record start event: {:?}", e))?;

    let mut launch_builder = cuda_ctx.stream().launch_builder(&cuda_function);
    launch_builder.arg(&input_ptr);
    launch_builder.arg(&output_ptr);
    launch_builder.arg(&array_len_u64);
    launch_builder.arg(&plan_ptr);

    let num_blocks = array_len.div_ceil(2048) as u32;
    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_builder
            .launch(config)
            .map_err(|e| vortex_err!("dynamic_dispatch kernel launch failed: {}", e))?;
    }

    let stream = cuda_ctx.stream();
    let ctx = stream.context();
    let end_event = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| vortex_err!("failed to create end event: {:?}", e))?;
    end_event
        .record(stream)
        .map_err(|e| vortex_err!("failed to record end event: {:?}", e))?;

    let elapsed_ms = start_event
        .elapsed_ms(&end_event)
        .map_err(|e| vortex_err!("failed to get elapsed time: {:?}", e))?;

    Ok(Duration::from_secs_f32(elapsed_ms / 1000.0))
}

/// Run a fused dynamic_dispatch launch on a bitpacked array, returning GPU time.
fn run_dynamic_dispatch_bitpacked_timed(
    cuda_ctx: &mut CudaExecutionCtx,
    bitpacked_array: &BitPackedArray,
    device_plan: &Arc<cudarc::driver::CudaSlice<DynamicDispatchPlan>>,
) -> VortexResult<Duration> {
    let packed = bitpacked_array.packed().clone();
    let len = bitpacked_array.len();

    // Move packed data to device.
    let device_input = if packed.is_on_device() {
        packed
    } else {
        block_on(cuda_ctx.move_to_device(packed)?).vortex_expect("failed to move to device")
    };

    let input_ptr = device_input
        .cuda_view::<u32>()
        .vortex_expect("failed to get input view")
        .device_ptr(cuda_ctx.stream())
        .0;

    // Allocate output buffer (padded to 1024-element chunks).
    let output_slice = cuda_ctx
        .device_alloc::<u32>(len.next_multiple_of(1024))
        .vortex_expect("failed to allocate output");
    let output_buf = CudaDeviceBuffer::new(output_slice);
    let output_ptr = output_buf.as_view::<u32>().device_ptr(cuda_ctx.stream()).0;

    // Ensure all previous works on the stream completed.
    cuda_ctx
        .stream()
        .synchronize()
        .map_err(|e| vortex_err!("failed to synchronize stream: {:?}", e))?;

    run_dynamic_dispatch_timed(cuda_ctx, input_ptr, output_ptr, len, device_plan)
}

fn bench_bitunpack_for_dynamic_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitunpack_for");
    group.sample_size(10);

    let plan = DynamicDispatchPlan::new(
        SourceOp::bitunpack(BIT_WIDTH),
        &[ScalarOp::frame_of_ref(REFERENCE_VALUE as u64)],
    );

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let bitpacked = make_bitpacked_array_u32(BIT_WIDTH, *len);

        group.bench_with_input(
            BenchmarkId::new("dynamic_dispatch_u32", len_str),
            &bitpacked,
            |b, array| {
                let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                    .vortex_expect("failed to create execution context");

                let device_plan = Arc::new(
                    cuda_ctx
                        .stream()
                        .clone_htod(std::slice::from_ref(&plan))
                        .expect("failed to copy plan to device"),
                );

                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let kernel_time = run_dynamic_dispatch_bitpacked_timed(
                            &mut cuda_ctx,
                            array,
                            &device_plan,
                        )
                        .vortex_expect("bitunpack+for dynamic_dispatch failed");
                        total_time += kernel_time;
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

// Benchmark: BitUnpack + FoR + ALP — single fused dynamic dispatch launch
fn bench_bitunpack_for_alp_dynamic_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitunpack_for_alp");
    group.sample_size(10);

    let plan = DynamicDispatchPlan::new(
        SourceOp::bitunpack(BIT_WIDTH),
        &[
            ScalarOp::frame_of_ref(REFERENCE_VALUE as u64),
            ScalarOp::alp(ALP_F, ALP_E),
        ],
    );

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let bitpacked = make_bitpacked_array_u32(BIT_WIDTH, *len);

        group.bench_with_input(
            BenchmarkId::new("dynamic_dispatch_u32", len_str),
            &bitpacked,
            |b, array| {
                let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                    .vortex_expect("failed to create execution context");

                let device_plan = Arc::new(
                    cuda_ctx
                        .stream()
                        .clone_htod(std::slice::from_ref(&plan))
                        .expect("failed to copy plan to device"),
                );

                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let kernel_time = run_dynamic_dispatch_bitpacked_timed(
                            &mut cuda_ctx,
                            array,
                            &device_plan,
                        )
                        .vortex_expect("bitunpack+for+alp dynamic_dispatch failed");
                        total_time += kernel_time;
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

fn benchmark_nested_decode(c: &mut Criterion) {
    bench_bitunpack_for_dynamic_dispatch(c);
    bench_bitunpack_for_alp_dynamic_dispatch(c);
}

criterion::criterion_group!(benches, benchmark_nested_decode);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
