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
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags;
use vortex_cuda::CudaDeviceBuffer;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

const BENCH_ARGS: &[(usize, &str)] = &[(1_000_000, "1M"), (10_000_000, "10M")];

const REFERENCE_VALUE: u32 = 100_000;

/// Number of FoR passes (matches the 3 separate kernel launches in other benchmarks).
const NUM_PASSES: u32 = 3;

/// Matches `ScalarOp` enum in `scalar_decode.cu`.
const SCALAR_OP_FOR_ADD: u32 = 0;

/// Matches `DecodeOp` struct layout in `scalar_decode.cu` (16-byte aligned).
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct DecodeOp {
    op: u32,
    _pad: u32,
    param: u64,
}

// SAFETY: DecodeOp is a plain-old-data struct with a fixed layout that matches
// the CUDA-side DecodeOp struct. It contains only primitive fields and is safe
// to pass to GPU memory.
unsafe impl DeviceRepr for DecodeOp {}

/// Helper: launch a single FoR kernel on a device buffer.
fn launch_for_kernel(
    cuda_ctx: &mut CudaExecutionCtx,
    device_buf: &CudaDeviceBuffer,
    output_len: usize,
) -> VortexResult<()> {
    let cuda_function = cuda_ctx.load_function_ptype("for", &[PType::U32])?;
    let mut launch_builder = cuda_ctx.launch_builder(&cuda_function);

    let device_view = device_buf.as_view::<u32>();
    let reference = REFERENCE_VALUE;
    let array_len_u64 = output_len as u64;

    launch_builder.arg(&device_view);
    launch_builder.arg(&reference);
    launch_builder.arg(&array_len_u64);

    let num_blocks = output_len.div_ceil(2048) as u32;
    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_builder
            .launch(config)
            .map_err(|e| vortex_err!("FoR kernel launch failed: {}", e))?;
    }
    Ok(())
}

/// Runs three separate FoR kernel launches with CUDA event timing.
fn run_three_separate_for_timed(
    cuda_ctx: &mut CudaExecutionCtx,
    input_data: &[u32],
    output_len: usize,
) -> VortexResult<Duration> {
    // Pre-copy data to device (not timed)
    let device_data = cuda_ctx
        .stream()
        .clone_htod(input_data)
        .map_err(|e| vortex_err!("failed to copy data to device: {}", e))?;
    let device_buf = CudaDeviceBuffer::new(device_data);

    // Record start event
    let stream = cuda_ctx.stream();
    let ctx = stream.context();
    let start_event = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| vortex_err!("failed to create start event: {:?}", e))?;
    start_event
        .record(stream)
        .map_err(|e| vortex_err!("failed to record start event: {:?}", e))?;

    // Launch 3 separate FoR kernels (in-place on same buffer)
    for _ in 0..3 {
        launch_for_kernel(cuda_ctx, &device_buf, output_len)?;
    }

    // Record end event
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

/// Benchmark: Three separate FoR kernel launches.
fn bench_three_separate_for(c: &mut Criterion) {
    let mut group = c.benchmark_group("three_for_separate");
    group.sample_size(10);

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let data: Vec<u32> = (0..*len).map(|i| i as u32).collect();
        let output_len = *len;

        group.bench_with_input(
            BenchmarkId::new("3x_for_u32", len_str),
            &data,
            |b, input_data| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let kernel_time =
                            run_three_separate_for_timed(&mut cuda_ctx, input_data, output_len)
                                .vortex_expect("three separate FoR failed");
                        total_time += kernel_time;
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

fn run_scalar_decode_timed(
    cuda_ctx: &mut CudaExecutionCtx,
    input_data: &[u32],
    output_len: usize,
    device_ops: &Arc<cudarc::driver::CudaSlice<DecodeOp>>,
) -> VortexResult<Duration> {
    let device_data = cuda_ctx
        .stream()
        .clone_htod(input_data)
        .map_err(|e| vortex_err!("failed to copy data to device: {}", e))?;
    let device_buf = CudaDeviceBuffer::new(device_data);

    let cuda_function = cuda_ctx.load_function("scalar_decode", &["u32"])?;
    let device_ptr = device_buf.device_ptr();
    let array_len_u64 = output_len as u64;
    let ops_ptr = device_ops.device_ptr(cuda_ctx.stream()).0;
    let num_ops = NUM_PASSES;

    // Ensure H2D copy is complete before we start timing.
    cuda_ctx
        .stream()
        .synchronize()
        .map_err(|e| vortex_err!("failed to synchronize stream: {:?}", e))?;

    let stream = cuda_ctx.stream();
    let ctx = stream.context();
    let start_event = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| vortex_err!("failed to create start event: {:?}", e))?;
    start_event
        .record(stream)
        .map_err(|e| vortex_err!("failed to record start event: {:?}", e))?;

    let mut launch_builder = cuda_ctx.stream().launch_builder(&cuda_function);
    launch_builder.arg(&device_ptr);
    launch_builder.arg(&array_len_u64);
    launch_builder.arg(&ops_ptr);
    launch_builder.arg(&num_ops);

    let num_blocks = output_len.div_ceil(2048) as u32;
    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        launch_builder
            .launch(config)
            .map_err(|e| vortex_err!("scalar_decode kernel launch failed: {}", e))?;
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

/// Benchmark: Single scalar_decode kernel that applies all 3 FoR ops in one kernel.
fn bench_scalar_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("three_for_scalar_decode");
    group.sample_size(10);

    let ops = vec![
        DecodeOp {
            op: SCALAR_OP_FOR_ADD,
            _pad: 0,
            param: REFERENCE_VALUE as u64,
        };
        NUM_PASSES as usize
    ];

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let data: Vec<u32> = (0..*len).map(|i| i as u32).collect();
        let output_len = *len;

        group.bench_with_input(
            BenchmarkId::new("3x_for_u32", len_str),
            &data,
            |b, input_data| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    let device_ops = Arc::new(
                        cuda_ctx
                            .stream()
                            .clone_htod(ops.as_slice())
                            .expect("failed to copy ops to device"),
                    );

                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let kernel_time = run_scalar_decode_timed(
                            &mut cuda_ctx,
                            input_data,
                            output_len,
                            &device_ops,
                        )
                        .vortex_expect("scalar_decode failed");
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
    bench_three_separate_for(c);
    bench_scalar_decode(c);
}

criterion::criterion_group!(benches, benchmark_nested_decode);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
