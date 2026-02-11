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
use vortex_cuda::bitpacked_cuda_kernel;
use vortex_cuda::bitpacked_cuda_launch_config;
use vortex_cuda::dynamic_dispatch_op::DynamicOp;
use vortex_cuda::dynamic_dispatch_op::DynamicOpCode_BITUNPACK;
use vortex_cuda::dynamic_dispatch_op::DynamicOpCode_FOR;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_fastlanes::BitPackedArray;
use vortex_session::VortexSession;

const BENCH_ARGS: &[(usize, &str)] = &[(1_000_000, "1M"), (10_000_000, "10M")];

const REFERENCE_VALUE: u32 = 100_000;

/// Bit width used for the bitpack+FoR benchmarks.
const BIT_WIDTH: u8 = 6;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Helper: launch a single FoR kernel on a device buffer (in-place).
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
    device_ops: &Arc<cudarc::driver::CudaSlice<DynamicOp>>,
    num_ops: u8,
) -> VortexResult<Duration> {
    let cuda_function = cuda_ctx.load_function("dynamic_dispatch", &["u32"])?;
    let array_len_u64 = array_len as u64;
    let ops_ptr = device_ops.device_ptr(cuda_ctx.stream()).0;

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
    launch_builder.arg(&ops_ptr);
    launch_builder.arg(&num_ops);

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

// ============================================================================
// Benchmark: BitUnpack + FoR — two separate kernel launches
// ============================================================================

/// Run bitunpack then FoR as two separate kernel launches, returning GPU time.
fn run_bitunpack_for_separate_timed(
    cuda_ctx: &mut CudaExecutionCtx,
    bitpacked_array: &BitPackedArray,
) -> VortexResult<Duration> {
    let packed = bitpacked_array.packed().clone();
    let bit_width = bitpacked_array.bit_width();
    let len = bitpacked_array.len();

    // Move packed data to device.
    let device_input = if packed.is_on_device() {
        packed
    } else {
        block_on(cuda_ctx.move_to_device(packed)?).vortex_expect("failed to move to device")
    };

    // Allocate output buffer (padded to 1024-element chunks).
    let output_slice = cuda_ctx
        .device_alloc::<u32>(len.next_multiple_of(1024))
        .vortex_expect("failed to allocate output");
    let output_buf = CudaDeviceBuffer::new(output_slice);

    let input_view = device_input
        .cuda_view::<u32>()
        .vortex_expect("failed to get input view");
    let output_view = output_buf.as_view::<u32>();

    // Ensure H2D copy is done before we start timing.
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

    // --- Kernel 1: BitUnpack ---
    {
        let output_width = u32::BITS as usize;
        let cuda_function = bitpacked_cuda_kernel(bit_width, output_width, cuda_ctx)?;
        let mut launch_builder = cuda_ctx.launch_builder(&cuda_function);
        launch_builder.arg(&input_view);
        launch_builder.arg(&output_view);

        let config = bitpacked_cuda_launch_config(output_width, len)?;
        unsafe {
            launch_builder
                .launch(config)
                .map_err(|e| vortex_err!("bit_unpack kernel launch failed: {}", e))?;
        }
    }

    // --- Kernel 2: FoR (in-place on output_buf) ---
    launch_for_kernel(cuda_ctx, &output_buf, len)?;

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

fn bench_bitunpack_for_separate(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitunpack_for");
    group.sample_size(10);

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let bitpacked = make_bitpacked_array_u32(BIT_WIDTH, *len);

        group.bench_with_input(
            BenchmarkId::new("separate_u32", len_str),
            &bitpacked,
            |b, array| {
                b.iter_custom(|iters| {
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context");

                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let kernel_time = run_bitunpack_for_separate_timed(&mut cuda_ctx, array)
                            .vortex_expect("bitunpack+for separate failed");
                        total_time += kernel_time;
                    }

                    total_time
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark: BitUnpack + FoR — single fused dynamic scalar_decode launch
// ============================================================================

/// Run bitunpack+FoR as a single fused dynamic_dispatch launch, returning GPU time.
fn run_bitunpack_for_fused_timed(
    cuda_ctx: &mut CudaExecutionCtx,
    bitpacked_array: &BitPackedArray,
    device_ops: &Arc<cudarc::driver::CudaSlice<DynamicOp>>,
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

    // ops = [BITUNPACK(bit_width), FOR(reference)]
    let num_ops: u8 = 2;

    // Ensure all previous works on the stream completed.
    cuda_ctx
        .stream()
        .synchronize()
        .map_err(|e| vortex_err!("failed to synchronize stream: {:?}", e))?;

    run_dynamic_dispatch_timed(cuda_ctx, input_ptr, output_ptr, len, device_ops, num_ops)
}

fn bench_bitunpack_for_dynamic_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitunpack_for");
    group.sample_size(10);

    // ops = [BITUNPACK(bit_width=BIT_WIDTH), FOR(REFERENCE_VALUE)]
    let ops = vec![
        DynamicOp {
            op: DynamicOpCode_BITUNPACK,
            param: BIT_WIDTH as u64,
        },
        DynamicOp {
            op: DynamicOpCode_FOR,
            param: REFERENCE_VALUE as u64,
        },
    ];

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let bitpacked = make_bitpacked_array_u32(BIT_WIDTH, *len);

        group.bench_with_input(
            BenchmarkId::new("dynamic_dispatch_u32", len_str),
            &bitpacked,
            |b, array| {
                let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                    .vortex_expect("failed to create execution context");

                // Force PTX JIT compilation before any measurement.
                cuda_ctx
                    .load_function("dynamic_dispatch", &["u32"])
                    .vortex_expect("failed to preload dynamic_dispatch kernel");

                let device_ops = Arc::new(
                    cuda_ctx
                        .stream()
                        .clone_htod(ops.as_slice())
                        .expect("failed to copy ops to device"),
                );

                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;

                    for _ in 0..iters {
                        let kernel_time =
                            run_bitunpack_for_fused_timed(&mut cuda_ctx, array, &device_ops)
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

fn benchmark_nested_decode(c: &mut Criterion) {
    bench_bitunpack_for_separate(c);
    bench_bitunpack_for_dynamic_dispatch(c);
}

criterion::criterion_group!(benches, benchmark_nested_decode);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
