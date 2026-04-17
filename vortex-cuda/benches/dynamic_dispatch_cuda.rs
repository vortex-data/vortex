// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::expect_used)]

use std::mem::size_of;
use std::sync::Arc;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags;
use vortex::array::IntoArray;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::scalar::Scalar;
use vortex::array::validity::Validity::NonNullable;
use vortex::buffer::Buffer;
use vortex::dtype::PType;
use vortex::encodings::alp::ALP;
use vortex::encodings::alp::ALPArrayExt;
use vortex::encodings::alp::ALPArraySlotsExt;
use vortex::encodings::alp::ALPFloat;
use vortex::encodings::alp::Exponents;
use vortex::encodings::alp::alp_encode;
use vortex::encodings::fastlanes::BitPackedData;
use vortex::encodings::fastlanes::FoR;
use vortex::encodings::fastlanes::FoRArrayExt;
use vortex::encodings::fastlanes::FoRData;
use vortex::encodings::runend::RunEnd;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::VortexSession;
use vortex_cuda::CudaDeviceBuffer;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda::dynamic_dispatch::CudaDispatchPlan;
use vortex_cuda::dynamic_dispatch::DispatchPlan;
use vortex_cuda::dynamic_dispatch::MaterializedPlan;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

const BENCH_ARGS: &[(usize, &str)] = &[(10_000_000, "10M"), (100_000_000, "100M")];

/// Launch the dynamic_dispatch kernel and return GPU-timed duration.
///
/// This deliberately does not use `CudaDispatchPlan::execute` because the
/// benchmark pre-allocates the output buffer and device plan once, then reuses
/// them across iterations.
fn run_timed(
    cuda_ctx: &mut CudaExecutionCtx,
    array_len: usize,
    output_buf: &CudaDeviceBuffer,
    device_plan: &Arc<CudaSlice<u8>>,
    shared_mem_bytes: u32,
) -> VortexResult<Duration> {
    let cuda_function = cuda_ctx.load_function("dynamic_dispatch", &[PType::U32])?;
    let array_len_u64 = array_len as u64;
    let output_view = output_buf.as_view::<u32>();
    let (output_ptr, record_output) = output_view.device_ptr(cuda_ctx.stream());
    let (plan_ptr, record_plan) = device_plan.device_ptr(cuda_ctx.stream());

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
    drop((record_output, record_plan));

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
    _plan: CudaDispatchPlan,
    smem_bytes: u32,
    len: usize,
    device_plan: Arc<CudaSlice<u8>>,
    output_buf: CudaDeviceBuffer,
    _plan_buffers: Vec<vortex::array::buffer::BufferHandle>,
}

impl BenchRunner {
    fn new(array: &vortex::array::ArrayRef, len: usize, cuda_ctx: &CudaExecutionCtx) -> Self {
        let plan = match DispatchPlan::new(array).vortex_expect("build_dyn_dispatch_plan") {
            DispatchPlan::Fused(plan) => plan,
            _ => unreachable!("encoding not fusable"),
        };
        let MaterializedPlan {
            dispatch_plan,
            device_buffers,
            shared_mem_bytes,
            ..
        } = plan.materialize(cuda_ctx).vortex_expect("materialize plan");

        let device_plan = Arc::new(
            cuda_ctx
                .stream()
                .clone_htod(dispatch_plan.as_bytes())
                .expect("htod plan"),
        );

        Self {
            _plan: dispatch_plan,
            smem_bytes: shared_mem_bytes,
            len,
            device_plan,
            output_buf: CudaDeviceBuffer::new(
                cuda_ctx
                    .device_alloc::<u32>(len.next_multiple_of(1024))
                    .expect("alloc output"),
            ),
            _plan_buffers: device_buffers,
        }
    }

    fn run(&self, cuda_ctx: &mut CudaExecutionCtx) -> Duration {
        cuda_ctx.stream().synchronize().unwrap();
        run_timed(
            cuda_ctx,
            self.len,
            &self.output_buf,
            &self.device_plan,
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
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let bp =
            BitPackedData::encode(&prim.into_array(), bit_width, &mut ctx).vortex_expect("bitpack");
        let array = FoR::try_new(bp.into_array(), Scalar::from(reference))
            .vortex_expect("for")
            .into_array();

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

    let dict_size: usize = 256;
    let dict_bit_width: u8 = 8;
    let dict_values: Vec<u32> = (0..dict_size as u32).map(|i| i * 1000 + 42).collect();

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let codes: Vec<u32> = (0..*len).map(|i| (i % dict_size) as u32).collect();
        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let codes_bp = BitPackedData::encode(&codes_prim.into_array(), dict_bit_width, &mut ctx)
            .vortex_expect("bitpack codes");
        let values_prim = PrimitiveArray::new(Buffer::from(dict_values.clone()), NonNullable);
        let dict = DictArray::new(codes_bp.into_array(), values_prim.into_array());
        let array = dict.into_array();

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

    let num_runs: usize = 100;

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let run_len = *len / num_runs;
        let ends: Vec<u32> = (1..=num_runs).map(|i| (i * run_len) as u32).collect();
        let values: Vec<u32> = (0..num_runs).map(|i| (i * 7 + 42) as u32).collect();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let ends_arr = PrimitiveArray::new(Buffer::from(ends), NonNullable).into_array();
        let values_arr = PrimitiveArray::new(Buffer::from(values), NonNullable).into_array();
        let re = RunEnd::new(ends_arr, values_arr, &mut ctx);
        let array = re.into_array();

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

    let dict_size: usize = 64;
    let dict_bit_width: u8 = 6;
    let dict_reference = 1_000_000u32;
    let codes_bit_width: u8 = 6;

    // Dict values: residuals 0..63 bitpacked, FoR adds 1_000_000
    let dict_residuals: Vec<u32> = (0..dict_size as u32).collect();
    let dict_prim = PrimitiveArray::new(Buffer::from(dict_residuals), NonNullable);
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let dict_bp = BitPackedData::encode(&dict_prim.into_array(), dict_bit_width, &mut ctx)
        .vortex_expect("bitpack dict");
    let dict_for =
        FoR::try_new(dict_bp.into_array(), Scalar::from(dict_reference)).vortex_expect("for dict");

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let codes: Vec<u32> = (0..*len).map(|i| (i % dict_size) as u32).collect();
        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let codes_bp = BitPackedData::encode(&codes_prim.into_array(), codes_bit_width, &mut ctx)
            .vortex_expect("bitpack codes");

        let dict = DictArray::new(codes_bp.into_array(), dict_for.clone().into_array());
        let array = dict.into_array();

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
// Benchmark: ALP(FoR(BitPacked)) for f32
// ---------------------------------------------------------------------------
fn bench_alp_for_bitpacked(c: &mut Criterion) {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut group = c.benchmark_group("alp_for_bp_6bw_f32");

    let exponents = Exponents { e: 2, f: 0 };
    let bit_width: u8 = 6;

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<f32>()) as u64));

        // Generate f32 values that ALP-encode without patches.
        let floats: Vec<f32> = (0..*len)
            .map(|i| <f32 as ALPFloat>::decode_single(10 + (i as i32 % 64), exponents))
            .collect();
        let float_prim = PrimitiveArray::new(Buffer::from(floats), NonNullable);

        // Encode: ALP → FoR → BitPacked
        let alp =
            alp_encode(float_prim.as_view(), Some(exponents), &mut ctx).vortex_expect("alp_encode");
        assert!(alp.patches().is_none());
        let for_arr = FoRData::encode(
            alp.encoded()
                .clone()
                .execute::<PrimitiveArray>(&mut ctx)
                .vortex_expect("to primitive"),
        )
        .vortex_expect("for encode");
        let bp = BitPackedData::encode(for_arr.encoded(), bit_width, &mut ctx)
            .vortex_expect("bitpack encode");

        let tree = ALP::new(
            FoR::try_new(bp.into_array(), for_arr.reference_scalar().clone())
                .vortex_expect("for_new")
                .into_array(),
            exponents,
            None,
        );
        let array = tree.into_array();

        group.bench_with_input(
            BenchmarkId::new("dynamic_dispatch_f32", len_str),
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
    bench_alp_for_bitpacked(c);
}

criterion::criterion_group! {
    name = benches;
    config = Criterion::default().without_plots()
        .sample_size(10)
        .warm_up_time(Duration::from_nanos(1))
        .measurement_time(Duration::from_nanos(1))
        .nresamples(10);
    targets = benchmark_dynamic_dispatch
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
