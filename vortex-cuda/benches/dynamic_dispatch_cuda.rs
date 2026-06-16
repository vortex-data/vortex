// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::expect_used)]

mod bench_config;

use std::marker::PhantomData;
use std::mem::size_of;
use std::os::raw::c_void;
use std::ptr;
use std::sync::Arc;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::CudaFunction;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags;
use futures::executor::block_on;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::buffer::BufferHandle;
use vortex::array::scalar::Scalar;
use vortex::array::validity::Validity::NonNullable;
use vortex::buffer::Buffer;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::encodings::alp::ALP;
use vortex::encodings::alp::ALPArrayExt;
use vortex::encodings::alp::ALPArraySlotsExt;
use vortex::encodings::alp::ALPFloat;
use vortex::encodings::alp::Exponents;
use vortex::encodings::alp::alp_encode;
use vortex::encodings::fastlanes::BitPackedArray;
use vortex::encodings::fastlanes::BitPackedArrayExt;
use vortex::encodings::fastlanes::BitPackedData;
use vortex::encodings::fastlanes::FoR;
use vortex::encodings::fastlanes::FoRArrayExt;
use vortex::encodings::fastlanes::FoRData;
use vortex::encodings::runend::RunEnd;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::VortexSession;
use vortex_cuda::CudaBufferExt;
use vortex_cuda::CudaDeviceBuffer;
use vortex_cuda::CudaDispatchMode;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda::CudaSessionExt;
use vortex_cuda::dynamic_dispatch::CudaDispatchPlan;
use vortex_cuda::dynamic_dispatch::DispatchPlan;
use vortex_cuda::dynamic_dispatch::MaterializedPlan;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::bench_config::BENCH_SIZES;

/// Launch the dynamic_dispatch kernel and return GPU-timed duration.
///
/// This deliberately does not use `CudaDispatchPlan::execute` because the
/// benchmark pre-allocates the output buffer and device plan once, then reuses
/// them across iterations.
fn run_timed<T: DeviceRepr + NativePType>(
    cuda_ctx: &mut CudaExecutionCtx,
    array_len: usize,
    output_buf: &CudaDeviceBuffer,
    device_plan: &Arc<CudaSlice<u8>>,
    shared_mem_bytes: u32,
) -> VortexResult<Duration> {
    let cuda_function = cuda_ctx.load_function("dynamic_dispatch", &[T::PTYPE])?;
    let array_len_u64 = array_len as u64;
    let output_view = output_buf.as_view::<T>();
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
///
/// `T` is the unsigned integer type matching the output element width
/// (e.g. `u32` for f32/i32/u32, `u64` for f64/i64/u64).
struct BenchRunner<T> {
    _plan: CudaDispatchPlan,
    smem_bytes: u32,
    len: usize,
    device_plan: Arc<CudaSlice<u8>>,
    output_buf: CudaDeviceBuffer,
    _plan_buffers: Vec<BufferHandle>,
    _phantom: PhantomData<T>,
}

impl<T: DeviceRepr + NativePType> BenchRunner<T> {
    fn new(array: &ArrayRef, len: usize, cuda_ctx: &mut CudaExecutionCtx) -> Self {
        let plan = match DispatchPlan::new(array, CudaDispatchMode::DynDispatchOnly)
            .vortex_expect("build_dyn_dispatch_plan")
        {
            DispatchPlan::Fused(plan) => plan,
            _ => unreachable!("encoding not fusable"),
        };
        let MaterializedPlan {
            dispatch_plan,
            device_buffers,
            shared_mem_bytes,
            ..
        } = block_on(plan.materialize(cuda_ctx)).vortex_expect("materialize plan");

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
                    .device_alloc::<T>(len.next_multiple_of(1024))
                    .expect("alloc output"),
            ),
            _plan_buffers: device_buffers,
            _phantom: PhantomData,
        }
    }

    fn run(&self, cuda_ctx: &mut CudaExecutionCtx) -> Duration {
        cuda_ctx.stream().synchronize().unwrap();
        run_timed::<T>(
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
    let mut group = c.benchmark_group("cuda");

    let bit_width: u8 = 6;
    let reference = 100_000u32;

    for (len, len_str) in BENCH_SIZES {
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
            BenchmarkId::new("cuda/for_bitpacked_6bw/dispatch_u32", len_str),
            len,
            |b, &n| {
                let mut cuda_ctx =
                    CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

                let bench_runner = BenchRunner::<u32>::new(&array, n, &mut cuda_ctx);

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
    let mut group = c.benchmark_group("cuda");

    let dict_size: usize = 256;
    let dict_bit_width: u8 = 8;
    let dict_values: Vec<u32> = (0..dict_size as u32).map(|i| i * 1000 + 42).collect();

    for (len, len_str) in BENCH_SIZES {
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
            BenchmarkId::new("cuda/dict_256vals_bp8bw_codes/dispatch_u32", len_str),
            len,
            |b, &n| {
                let mut cuda_ctx =
                    CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

                let bench_runner = BenchRunner::<u32>::new(&array, n, &mut cuda_ctx);

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
    let mut group = c.benchmark_group("cuda");

    let num_runs: usize = 100;

    for (len, len_str) in BENCH_SIZES {
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
            BenchmarkId::new("cuda/runend_100runs/dispatch_u32", len_str),
            len,
            |b, &n| {
                let mut cuda_ctx =
                    CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

                let bench_runner = BenchRunner::<u32>::new(&array, n, &mut cuda_ctx);

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
// Benchmark: Dict(codes=BitPacked, values=ALP(FoR(BitPacked))) — f32
// ---------------------------------------------------------------------------
fn bench_dict_bp_codes_alp_for_bp_values_dynanmic_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    let exponents = Exponents { e: 2, f: 0 };
    let values_bit_width: u8 = 6;
    let dict_size = 64usize;
    let codes_bit_width: u8 = 6;

    for (len, len_str) in BENCH_SIZES {
        group.throughput(Throughput::Bytes((len * size_of::<f32>()) as u64));

        let dict_floats: Vec<f32> = (0..dict_size)
            .map(|i| <f32 as ALPFloat>::decode_single(10 + i as i32, exponents))
            .collect();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        // values: ALP → FoR → BitPacked.
        let float_prim = PrimitiveArray::new(Buffer::from(dict_floats), NonNullable);
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
        let bp = BitPackedData::encode(for_arr.encoded(), values_bit_width, &mut ctx)
            .vortex_expect("bitpack values");
        let values_tree = ALP::new(
            FoR::try_new(bp.into_array(), for_arr.reference_scalar().clone())
                .vortex_expect("for_new")
                .into_array(),
            exponents,
            None,
        );

        // codes: BitPacked u32 indices into the dict.
        let codes: Vec<u32> = (0..*len).map(|i| (i % dict_size) as u32).collect();
        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let codes_bp = BitPackedData::encode(&codes_prim.into_array(), codes_bit_width, &mut ctx)
            .vortex_expect("bitpack codes");

        let dict = DictArray::new(codes_bp.into_array(), values_tree.into_array());
        let array = dict.into_array();

        group.bench_with_input(
            BenchmarkId::new(
                "cuda/dict_64vals_bp6bw_codes_alp_for_bp6bw_f32_values/dispatch_f32",
                len_str,
            ),
            len,
            |b, &n| {
                let mut cuda_ctx =
                    CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

                let bench_runner = BenchRunner::<u32>::new(&array, n, &mut cuda_ctx);

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

mod standalone {
    use super::*;

    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    struct NullGpuPatches {
        chunk_offsets: *mut c_void,
        chunk_offset_type: u32,
        indices: *mut u32,
        values: *mut c_void,
        offset: u32,
        offset_within_chunk: u32,
        num_patches: u32,
        n_chunks: u32,
    }

    unsafe impl DeviceRepr for NullGpuPatches {}

    impl NullGpuPatches {
        const NULL: Self = Self {
            chunk_offsets: ptr::null_mut(),
            chunk_offset_type: 2,
            indices: ptr::null_mut(),
            values: ptr::null_mut(),
            offset: 0,
            offset_within_chunk: 0,
            num_patches: 0,
            n_chunks: 0,
        };
    }

    pub(super) struct DictAlpForBitpackedRunner {
        values_packed: BufferHandle,
        codes_packed: BufferHandle,
        values_for_buf: CudaDeviceBuffer,
        values_alp_buf: CudaDeviceBuffer,
        codes_buf: CudaDeviceBuffer,
        output_buf: CudaDeviceBuffer,
        bit_unpack_fn: CudaFunction,
        alp_fn: CudaFunction,
        dict_fn: CudaFunction,
        values_reference: i32,
        alp_f: f32,
        alp_e: f32,
        values_len: usize,
        codes_len: usize,
    }

    /// Decompresses the following tree in a chain of standalone kernels without fusing:
    /// Dict<f32>
    /// ├── codes: BitPacked<u32>
    /// │   └── packed codes, bit_width = 6
    /// └── values: ALP<f32>
    ///     └── FoR<i32>
    ///         └── BitPacked<i32>
    ///             └── packed values, bit_width = 6
    impl DictAlpForBitpackedRunner {
        pub(super) fn new(
            values_bp: &BitPackedArray,
            values_reference: i32,
            exponents: Exponents,
            codes_bp: &BitPackedArray,
            len: usize,
            cuda_session: &CudaSession,
            cuda_ctx: &mut CudaExecutionCtx,
        ) -> Self {
            assert_eq!(values_bp.bit_width(), 6);
            assert_eq!(codes_bp.bit_width(), 6);

            let values_packed = block_on(cuda_ctx.ensure_on_device(values_bp.packed().clone()))
                .vortex_expect("values packed");
            let codes_packed = block_on(cuda_ctx.ensure_on_device(codes_bp.packed().clone()))
                .vortex_expect("codes packed");

            let values_len = values_bp.len();
            let values_for_buf = CudaDeviceBuffer::new(
                cuda_ctx
                    .device_alloc::<i32>(values_len.next_multiple_of(1024))
                    .vortex_expect("alloc values for"),
            );
            let values_alp_buf = CudaDeviceBuffer::new(
                cuda_ctx
                    .device_alloc::<f32>(values_len.next_multiple_of(1024))
                    .vortex_expect("alloc values alp"),
            );
            let codes_buf = CudaDeviceBuffer::new(
                cuda_ctx
                    .device_alloc::<u32>(len.next_multiple_of(1024))
                    .vortex_expect("alloc codes"),
            );
            let output_buf = CudaDeviceBuffer::new(
                cuda_ctx
                    .device_alloc::<f32>(len)
                    .vortex_expect("alloc output"),
            );

            cuda_ctx.stream().synchronize().expect("setup sync");

            Self {
                values_packed,
                codes_packed,
                values_for_buf,
                values_alp_buf,
                codes_buf,
                output_buf,
                bit_unpack_fn: cuda_session
                    .load_function_with_suffixes("bit_unpack_32", &["6bw", "32t"])
                    .vortex_expect("load bit_unpack"),
                alp_fn: cuda_session
                    .load_function_with_suffixes("alp", &["i32", "f32", "32t"])
                    .vortex_expect("load alp"),
                dict_fn: cuda_ctx
                    .load_function("dict", &[PType::F32, PType::U32])
                    .vortex_expect("load dict"),
                values_reference,
                alp_f: <f32 as ALPFloat>::F10[exponents.f as usize],
                alp_e: <f32 as ALPFloat>::IF10[exponents.e as usize],
                values_len,
                codes_len: len,
            }
        }

        // Setup owns all allocations and H2D copies; this times only the standalone
        // kernel sequence needed to mirror the fused dynamic-dispatch plan.
        pub(super) fn run(&self, cuda_ctx: &mut CudaExecutionCtx) -> Duration {
            cuda_ctx.stream().synchronize().unwrap();

            let values_packed_view = self.values_packed.cuda_view::<u32>().unwrap();
            let codes_packed_view = self.codes_packed.cuda_view::<u32>().unwrap();
            let values_for_view = self.values_for_buf.as_view::<i32>();
            let values_alp_view = self.values_alp_buf.as_view::<f32>();
            let codes_view = self.codes_buf.as_view::<u32>();
            let output_view = self.output_buf.as_view::<f32>();
            let patches = NullGpuPatches::NULL;

            cuda_ctx.stream().synchronize().unwrap();

            let stream = cuda_ctx.stream();
            let ctx = stream.context();
            let start_event = ctx
                .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .map_err(|e| vortex_err!("{e:?}"))
                .unwrap();
            start_event
                .record(stream)
                .map_err(|e| vortex_err!("{e:?}"))
                .unwrap();

            {
                let mut launch = cuda_ctx.stream().launch_builder(&self.bit_unpack_fn);
                launch.arg(&values_packed_view);
                launch.arg(&values_for_view);
                launch.arg(&self.values_reference);
                launch.arg(&patches);
                unsafe {
                    launch
                        .launch(LaunchConfig {
                            grid_dim: (self.values_len.div_ceil(1024) as u32, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .unwrap();
                }
            }

            {
                let mut launch = cuda_ctx.stream().launch_builder(&self.alp_fn);
                let values_len = self.values_len as u64;
                launch.arg(&values_for_view);
                launch.arg(&values_alp_view);
                launch.arg(&self.alp_f);
                launch.arg(&self.alp_e);
                launch.arg(&values_len);
                launch.arg(&patches);
                unsafe {
                    launch
                        .launch(LaunchConfig {
                            grid_dim: (self.values_len.div_ceil(1024) as u32, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .unwrap();
                }
            }

            {
                let mut launch = cuda_ctx.stream().launch_builder(&self.bit_unpack_fn);
                let reference = 0u32;
                launch.arg(&codes_packed_view);
                launch.arg(&codes_view);
                launch.arg(&reference);
                launch.arg(&patches);
                unsafe {
                    launch
                        .launch(LaunchConfig {
                            grid_dim: (self.codes_len.div_ceil(1024) as u32, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .unwrap();
                }
            }

            {
                let mut launch = cuda_ctx.stream().launch_builder(&self.dict_fn);
                let codes_len = self.codes_len as u64;
                launch.arg(&codes_view);
                launch.arg(&codes_len);
                launch.arg(&values_alp_view);
                launch.arg(&output_view);
                unsafe {
                    launch
                        .launch(LaunchConfig {
                            grid_dim: (self.codes_len.div_ceil(2048) as u32, 1, 1),
                            block_dim: (64, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .unwrap();
                }
            }

            let stream = cuda_ctx.stream();
            let ctx = stream.context();
            let end_event = ctx
                .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .map_err(|e| vortex_err!("{e:?}"))
                .unwrap();
            end_event
                .record(stream)
                .map_err(|e| vortex_err!("{e:?}"))
                .unwrap();

            let elapsed_ms = start_event
                .elapsed_ms(&end_event)
                .map_err(|e| vortex_err!("{e:?}"))
                .unwrap();

            Duration::from_secs_f32(elapsed_ms / 1000.0)
        }
    }
}

fn bench_dict_bp_codes_alp_for_bp_values_composed_standalone(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    for (len, len_str) in BENCH_SIZES {
        group.throughput(Throughput::Bytes((len * size_of::<f32>()) as u64));

        let dict_size = 64usize;
        let codes_bit_width: u8 = 6;
        let values_bit_width: u8 = 6;
        let exponents = Exponents { e: 2, f: 0 };

        let dict_floats: Vec<f32> = (0..dict_size)
            .map(|i| <f32 as ALPFloat>::decode_single(10 + (i as i32 % 64), exponents))
            .collect();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let float_prim = PrimitiveArray::new(Buffer::from(dict_floats), NonNullable);
        let alp =
            alp_encode(float_prim.as_view(), Some(exponents), &mut ctx).vortex_expect("alp_encode");
        assert!(alp.patches().is_none());
        let alp_encoded = alp
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .vortex_expect("to primitive");
        let for_arr = FoRData::encode(alp_encoded.clone()).vortex_expect("for encode");
        let bp = BitPackedData::encode(for_arr.encoded(), values_bit_width, &mut ctx)
            .vortex_expect("bitpack values");
        let values_bp = bp;
        let values_reference: i32 = for_arr
            .reference_scalar()
            .try_into()
            .vortex_expect("values reference");
        let values_for = FoR::try_new(
            values_bp.clone().into_array(),
            for_arr.reference_scalar().clone(),
        )
        .vortex_expect("for_new")
        .into_array();
        let _values_alp = ALP::new(values_for, exponents, None).into_array();

        let codes: Vec<u32> = (0..*len).map(|i| (i % dict_size) as u32).collect();
        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let codes_bp = BitPackedData::encode(&codes_prim.into_array(), codes_bit_width, &mut ctx)
            .vortex_expect("bitpack codes");

        group.bench_with_input(
            BenchmarkId::new(
                "cuda/dict_64vals_bp6bw_codes_alp_for_bp6bw_f32_values/composed_standalone_f32",
                len_str,
            ),
            &(values_bp, values_reference, codes_bp),
            |b, (values_bp, values_reference, codes_bp)| {
                b.iter_custom(|iters| {
                    let session = VortexSession::empty();
                    let cuda_session = session.cuda_session();
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&session)
                        .vortex_expect("ctx")
                        .with_dispatch_mode(CudaDispatchMode::StandaloneOnly);
                    let bench_runner = standalone::DictAlpForBitpackedRunner::new(
                        values_bp,
                        *values_reference,
                        exponents,
                        codes_bp,
                        *len,
                        cuda_session,
                        &mut cuda_ctx,
                    );

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
// Benchmark: ALP(FoR(BitPacked)) — f64
// ---------------------------------------------------------------------------
fn bench_alp_for_bitpacked_f64(c: &mut Criterion) {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut group = c.benchmark_group("cuda");

    let exponents = Exponents { e: 2, f: 0 };
    let bit_width: u8 = 6;

    for (len, len_str) in BENCH_SIZES {
        group.throughput(Throughput::Bytes((len * size_of::<f64>()) as u64));

        // Generate f64 values that ALP-encode without patches.
        let floats: Vec<f64> = (0..*len)
            .map(|i| <f64 as ALPFloat>::decode_single(10 + (i as i64 % 64), exponents))
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
            BenchmarkId::new("cuda/alp_for_bp_6bw_f64/dispatch_f64", len_str),
            len,
            |b, &n| {
                let mut cuda_ctx =
                    CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

                let bench_runner = BenchRunner::<u64>::new(&array, n, &mut cuda_ctx);

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
    let mut group = c.benchmark_group("cuda");

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

    for (len, len_str) in BENCH_SIZES {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let codes: Vec<u32> = (0..*len).map(|i| (i % dict_size) as u32).collect();
        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let codes_bp = BitPackedData::encode(&codes_prim.into_array(), codes_bit_width, &mut ctx)
            .vortex_expect("bitpack codes");

        let dict = DictArray::new(codes_bp.into_array(), dict_for.clone().into_array());
        let array = dict.into_array();

        group.bench_with_input(
            BenchmarkId::new(
                "cuda/dict_64vals_bp6bw_codes_for_bp6bw_values/dispatch_u32",
                len_str,
            ),
            len,
            |b, &n| {
                let mut cuda_ctx =
                    CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

                let bench_runner = BenchRunner::<u32>::new(&array, n, &mut cuda_ctx);

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
    let mut group = c.benchmark_group("cuda");

    let exponents = Exponents { e: 2, f: 0 };
    let bit_width: u8 = 6;

    for (len, len_str) in BENCH_SIZES {
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
            BenchmarkId::new("cuda/alp_for_bp_6bw_f32/dispatch_f32", len_str),
            len,
            |b, &n| {
                let mut cuda_ctx =
                    CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

                let bench_runner = BenchRunner::<u32>::new(&array, n, &mut cuda_ctx);

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
// Benchmark: Dict with narrower BitPacked codes (exercises widen_inplace)
// ---------------------------------------------------------------------------

/// Dict(codes=BitPacked<u8>, values=Prim<u32>) — widens u8 → u32 in smem.
fn bench_dict_bp_u8_codes_u32_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    let dict_size: usize = 4; // 2-bit codes
    let bit_width: u8 = 2;
    let dict_values: Vec<u32> = (0..dict_size as u32).map(|i| i * 1000 + 42).collect();

    for (len, len_str) in BENCH_SIZES {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let codes: Vec<u8> = (0..*len).map(|i| (i % dict_size) as u8).collect();
        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let codes_bp = BitPackedData::encode(&codes_prim.into_array(), bit_width, &mut ctx)
            .vortex_expect("bitpack u8 codes");
        let values_prim = PrimitiveArray::new(Buffer::from(dict_values.clone()), NonNullable);
        let dict = DictArray::new(codes_bp.into_array(), values_prim.into_array());
        let array = dict.into_array();

        group.bench_with_input(
            BenchmarkId::new("cuda/dict_widen_u8_to_u32/dispatch_u32", len_str),
            len,
            |b, &n| {
                let mut cuda_ctx =
                    CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

                let bench_runner = BenchRunner::<u32>::new(&array, n, &mut cuda_ctx);

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

/// Dict(codes=BitPacked<u16>, values=Prim<u32>) — widens u16 → u32 in smem.
fn bench_dict_bp_u16_codes_u32_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    let dict_size: usize = 8; // 3-bit codes
    let bit_width: u8 = 3;
    let dict_values: Vec<u32> = (0..dict_size as u32).map(|i| i * 5000 + 100).collect();

    for (len, len_str) in BENCH_SIZES {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let codes: Vec<u16> = (0..*len).map(|i| (i % dict_size) as u16).collect();
        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let codes_bp = BitPackedData::encode(&codes_prim.into_array(), bit_width, &mut ctx)
            .vortex_expect("bitpack u16 codes");
        let values_prim = PrimitiveArray::new(Buffer::from(dict_values.clone()), NonNullable);
        let dict = DictArray::new(codes_bp.into_array(), values_prim.into_array());
        let array = dict.into_array();

        group.bench_with_input(
            BenchmarkId::new("cuda/dict_widen_u16_to_u32/dispatch_u32", len_str),
            len,
            |b, &n| {
                let mut cuda_ctx =
                    CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

                let bench_runner = BenchRunner::<u32>::new(&array, n, &mut cuda_ctx);

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

/// Dict(codes=BitPacked<u32>, values=Prim<u32>) — same-width baseline, no widen.
fn bench_dict_bp_u32_codes_u32_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    let dict_size: usize = 8; // 3-bit codes
    let bit_width: u8 = 3;
    let dict_values: Vec<u32> = (0..dict_size as u32).map(|i| i * 5000 + 100).collect();

    for (len, len_str) in BENCH_SIZES {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let codes: Vec<u32> = (0..*len).map(|i| (i % dict_size) as u32).collect();
        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let codes_bp = BitPackedData::encode(&codes_prim.into_array(), bit_width, &mut ctx)
            .vortex_expect("bitpack u32 codes");
        let values_prim = PrimitiveArray::new(Buffer::from(dict_values.clone()), NonNullable);
        let dict = DictArray::new(codes_bp.into_array(), values_prim.into_array());
        let array = dict.into_array();

        group.bench_with_input(
            BenchmarkId::new("cuda/dict_nowiden_u32_to_u32/dispatch_u32", len_str),
            len,
            |b, &n| {
                let mut cuda_ctx =
                    CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

                let bench_runner = BenchRunner::<u32>::new(&array, n, &mut cuda_ctx);

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
    bench_dict_bp_codes_alp_for_bp_values_dynanmic_dispatch(c);
    bench_dict_bp_codes_alp_for_bp_values_composed_standalone(c);
    bench_alp_for_bitpacked(c);
    bench_alp_for_bitpacked_f64(c);
    bench_dict_bp_u8_codes_u32_values(c);
    bench_dict_bp_u16_codes_u32_values(c);
    bench_dict_bp_u32_codes_u32_values(c);
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_dynamic_dispatch
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
