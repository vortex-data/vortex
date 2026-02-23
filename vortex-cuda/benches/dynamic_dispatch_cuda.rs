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
use vortex::array::IntoArray;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::scalar::Scalar;
use vortex::array::validity::Validity::NonNullable;
use vortex::buffer::Buffer;
use vortex::encodings::fastlanes::BitPackedArray;
use vortex::encodings::fastlanes::FoRArray;
use vortex::encodings::runend::RunEndArray;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::VortexSession;
use vortex_cuda::CudaDeviceBuffer;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda::dynamic_dispatch;
use vortex_cuda::dynamic_dispatch::DynamicDispatchPlan;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

const BENCH_ARGS: &[(usize, &str)] = &[
    (1_000_000, "1M"),
    (10_000_000, "10M"),
    (100_000_000, "100M"),
];

/// Launch the dynamic_dispatch kernel and return GPU-timed duration.
fn run_timed(
    cuda_ctx: &mut CudaExecutionCtx,
    output_ptr: u64,
    array_len: usize,
    device_plan: &Arc<cudarc::driver::CudaSlice<DynamicDispatchPlan>>,
    shared_mem_bytes: u32,
) -> VortexResult<Duration> {
    let cuda_function = cuda_ctx.load_function("dynamic_dispatch", &["u32"])?;
    let array_len_u64 = array_len as u64;
    let plan_ptr = device_plan.device_ptr(cuda_ctx.stream()).0;

    let stream = cuda_ctx.stream();
    let ctx = stream.context();
    let start_event = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| vortex_err!("{e:?}"))?;
    start_event
        .record(stream)
        .map_err(|e| vortex_err!("{e:?}"))?;

    let mut launch_builder = cuda_ctx.stream().launch_builder(&cuda_function);
    launch_builder.arg(&output_ptr);
    launch_builder.arg(&array_len_u64);
    launch_builder.arg(&plan_ptr);

    let num_blocks = array_len.div_ceil(2048) as u32;
    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes,
    };

    unsafe {
        launch_builder
            .launch(config)
            .map_err(|e| vortex_err!("kernel launch failed: {e}"))?;
    }

    let stream = cuda_ctx.stream();
    let ctx = stream.context();
    let end_event = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| vortex_err!("{e:?}"))?;
    end_event.record(stream).map_err(|e| vortex_err!("{e:?}"))?;

    let elapsed_ms = start_event
        .elapsed_ms(&end_event)
        .map_err(|e| vortex_err!("{e:?}"))?;

    Ok(Duration::from_secs_f32(elapsed_ms / 1000.0))
}

/// Benchmark runner: builds a dynamic plan and launches the kernel.
struct BenchRunner {
    _plan: DynamicDispatchPlan,
    smem_bytes: u32,
    output_ptr: u64,
    len: usize,
    // Keep alive
    _device_plan: Arc<cudarc::driver::CudaSlice<DynamicDispatchPlan>>,
    _output_buf: CudaDeviceBuffer,
    _plan_buffers: Vec<vortex::array::buffer::BufferHandle>,
}

impl BenchRunner {
    fn new(array: &vortex::array::ArrayRef, len: usize, cuda_ctx: &CudaExecutionCtx) -> Self {
        let (plan, plan_buffers) =
            dynamic_dispatch::build_plan(array, cuda_ctx).vortex_expect("build_plan");
        let smem_bytes = plan.shared_mem_bytes::<u32>();

        let device_plan = Arc::new(
            cuda_ctx
                .stream()
                .clone_htod(std::slice::from_ref(&plan))
                .expect("htod plan"),
        );

        let output_slice = cuda_ctx
            .device_alloc::<u32>(len.next_multiple_of(1024))
            .expect("alloc output");
        let output_buf = CudaDeviceBuffer::new(output_slice);
        let output_ptr = output_buf.as_view::<u32>().device_ptr(cuda_ctx.stream()).0;

        Self {
            _plan: plan,
            smem_bytes,
            output_ptr,
            len,
            _device_plan: device_plan,
            _output_buf: output_buf,
            _plan_buffers: plan_buffers,
        }
    }

    fn run(&self, cuda_ctx: &mut CudaExecutionCtx) -> Duration {
        cuda_ctx.stream().synchronize().unwrap();
        run_timed(
            cuda_ctx,
            self.output_ptr,
            self.len,
            &self._device_plan,
            self.smem_bytes,
        )
        .unwrap()
    }
}

// ---------------------------------------------------------------------------
// Benchmark: FoR(BitPacked)
// ---------------------------------------------------------------------------
fn bench_for_bitpacked(c: &mut Criterion) {
    let mut group = c.benchmark_group("for_bitpacked_6bw");
    group.sample_size(10);

    let bit_width: u8 = 6;
    let reference = 100_000u32;

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        // FoR(BitPacked): residuals 0..max_val, reference adds 100_000
        let max_val = (1u64 << bit_width).saturating_sub(1);
        let residuals: Vec<u32> = (0..*len)
            .map(|i| (i as u64 % (max_val + 1)) as u32)
            .collect();
        let prim = PrimitiveArray::new(Buffer::from(residuals), NonNullable);
        let bp = BitPackedArray::encode(prim.as_ref(), bit_width).vortex_expect("bitpack");
        let for_arr =
            FoRArray::try_new(bp.into_array(), Scalar::from(reference)).vortex_expect("for");
        let array = for_arr.to_array();

        group.bench_with_input(
            BenchmarkId::new("dynamic_dispatch_u32", len_str),
            len,
            |b, &n| {
                let mut cuda_ctx =
                    CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

                let bench_runner = BenchRunner::new(&array, n, &cuda_ctx);

                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;
                    for _ in 0..iters {
                        total_time += bench_runner.run(&mut cuda_ctx);
                    }
                    total_time
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Dict(codes=BitPacked, values=Primitive)
// ---------------------------------------------------------------------------
fn bench_dict_bp_codes(c: &mut Criterion) {
    let mut group = c.benchmark_group("dict_256vals_bp8bw_codes");
    group.sample_size(10);

    let dict_size: usize = 256;
    let dict_bit_width: u8 = 8;
    let dict_values: Vec<u32> = (0..dict_size as u32).map(|i| i * 1000 + 42).collect();

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let codes: Vec<u32> = (0..*len).map(|i| (i % dict_size) as u32).collect();
        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let codes_bp = BitPackedArray::encode(codes_prim.as_ref(), dict_bit_width)
            .vortex_expect("bitpack codes");
        let values_prim = PrimitiveArray::new(Buffer::from(dict_values.clone()), NonNullable);
        let dict = DictArray::new(codes_bp.into_array(), values_prim.into_array());
        let array = dict.to_array();

        group.bench_with_input(
            BenchmarkId::new("dynamic_dispatch_u32", len_str),
            len,
            |b, &n| {
                let mut cuda_ctx =
                    CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

                let bench_runner = BenchRunner::new(&array, n, &cuda_ctx);

                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;
                    for _ in 0..iters {
                        total_time += bench_runner.run(&mut cuda_ctx);
                    }
                    total_time
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: RunEnd(ends=Prim, values=Prim)
// ---------------------------------------------------------------------------
fn bench_runend(c: &mut Criterion) {
    let mut group = c.benchmark_group("runend_100runs");
    group.sample_size(10);

    let num_runs: usize = 100;

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let run_len = *len / num_runs;
        let ends: Vec<u32> = (1..=num_runs).map(|i| (i * run_len) as u32).collect();
        let values: Vec<u32> = (0..num_runs).map(|i| (i * 7 + 42) as u32).collect();

        let ends_arr = PrimitiveArray::new(Buffer::from(ends), NonNullable).into_array();
        let values_arr = PrimitiveArray::new(Buffer::from(values), NonNullable).into_array();
        let re = RunEndArray::new(ends_arr, values_arr);
        let array = re.to_array();

        group.bench_with_input(
            BenchmarkId::new("dynamic_dispatch_u32", len_str),
            len,
            |b, &n| {
                let mut cuda_ctx =
                    CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

                let bench_runner = BenchRunner::new(&array, n, &cuda_ctx);

                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;
                    for _ in 0..iters {
                        total_time += bench_runner.run(&mut cuda_ctx);
                    }
                    total_time
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Dict(codes=BitPacked, values=FoR(BitPacked))
// ---------------------------------------------------------------------------
fn bench_dict_bp_codes_bp_for_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("dict_64vals_bp6bw_codes_for_bp6bw_values");
    group.sample_size(10);

    let dict_size: usize = 64;
    let dict_bit_width: u8 = 6;
    let dict_reference = 1_000_000u32;
    let codes_bit_width: u8 = 6;

    // Dict values: residuals 0..63 bitpacked, FoR adds 1_000_000
    let dict_residuals: Vec<u32> = (0..dict_size as u32).collect();
    let dict_prim = PrimitiveArray::new(Buffer::from(dict_residuals), NonNullable);
    let dict_bp =
        BitPackedArray::encode(dict_prim.as_ref(), dict_bit_width).vortex_expect("bitpack dict");
    let dict_for = FoRArray::try_new(dict_bp.into_array(), Scalar::from(dict_reference))
        .vortex_expect("for dict");

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let codes: Vec<u32> = (0..*len).map(|i| (i % dict_size) as u32).collect();
        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let codes_bp = BitPackedArray::encode(codes_prim.as_ref(), codes_bit_width)
            .vortex_expect("bitpack codes");

        let dict = DictArray::new(codes_bp.into_array(), dict_for.to_array());
        let array = dict.to_array();

        group.bench_with_input(
            BenchmarkId::new("dynamic_dispatch_u32", len_str),
            len,
            |b, &n| {
                let mut cuda_ctx =
                    CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

                let bench_runner = BenchRunner::new(&array, n, &cuda_ctx);

                b.iter_custom(|iters| {
                    let mut total_time = Duration::ZERO;
                    for _ in 0..iters {
                        total_time += bench_runner.run(&mut cuda_ctx);
                    }
                    total_time
                });
            },
        );
    }

    group.finish();
}

fn benchmark_dynamic_dispatch(c: &mut Criterion) {
    bench_for_bitpacked(c);
    bench_dict_bp_codes(c);
    bench_runend(c);
    bench_dict_bp_codes_bp_for_values(c);
}

criterion::criterion_group!(benches, benchmark_dynamic_dispatch);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
