// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Focused comparison bench for the cuda-oxide dynamic-dispatch POC.
//!
//! Runs the production `.cu` dynamic_dispatch kernel on the same synthetic case
//! and timing shape used by `vortex-cuda/cuda-oxide-poc`:
//!
//! - case: FoR(BitPacked<u32>)
//! - len: 16 * 1024 * 1024
//! - bit width: 6
//! - reference: 100_000
//! - warmup launches: 20
//! - timed launches: 200

#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]
#![expect(clippy::cast_possible_truncation)]

use std::mem::size_of;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags;
use futures::executor::block_on;
use vortex::array::IntoArray;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::buffer::BufferHandle;
use vortex::array::scalar::Scalar;
use vortex::array::validity::Validity::NonNullable;
use vortex::buffer::Buffer;
use vortex::dtype::NativePType;
use vortex::encodings::fastlanes::BitPackedData;
use vortex::encodings::fastlanes::FoR;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::VortexSession;
use vortex_cuda::CudaDeviceBuffer;
use vortex_cuda::CudaDispatchMode;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda::dynamic_dispatch::DispatchPlan;
use vortex_cuda::dynamic_dispatch::MaterializedPlan;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

const LEN: usize = 16 * 1024 * 1024;
const BIT_WIDTH: u8 = 6;
const REFERENCE: u32 = 100_000;
const WARMUP_LAUNCHES: usize = 20;
const TIMED_LAUNCHES: usize = 200;

struct BenchRunner<T> {
    len: usize,
    smem_bytes: u32,
    device_plan: Arc<CudaSlice<u8>>,
    output_buf: CudaDeviceBuffer,
    _plan_buffers: Vec<BufferHandle>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DeviceRepr + NativePType> BenchRunner<T> {
    fn new(cuda_ctx: &mut CudaExecutionCtx) -> Self {
        let max_val = (1u64 << BIT_WIDTH).saturating_sub(1);
        let residuals: Vec<u32> = (0..LEN)
            .map(|i| (i as u64 % (max_val + 1)) as u32)
            .collect();
        let prim = PrimitiveArray::new(Buffer::from(residuals), NonNullable);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let bp =
            BitPackedData::encode(&prim.into_array(), BIT_WIDTH, &mut ctx).vortex_expect("bitpack");
        let array = FoR::try_new(bp.into_array(), Scalar::from(REFERENCE))
            .vortex_expect("for")
            .into_array();

        let plan = match DispatchPlan::new(&array, CudaDispatchMode::DynDispatchOnly)
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
            len: LEN,
            smem_bytes: shared_mem_bytes,
            device_plan,
            output_buf: CudaDeviceBuffer::new(
                cuda_ctx
                    .device_alloc::<T>(LEN.next_multiple_of(1024))
                    .expect("alloc output"),
            ),
            _plan_buffers: device_buffers,
            _phantom: std::marker::PhantomData,
        }
    }

    fn launch_once(&self, cuda_ctx: &mut CudaExecutionCtx) -> VortexResult<()> {
        let cuda_function = cuda_ctx.load_function("dynamic_dispatch", &[T::PTYPE])?;
        let array_len_u64 = self.len as u64;
        let output_view = self.output_buf.as_view::<T>();
        let (output_ptr, record_output) = output_view.device_ptr(cuda_ctx.stream());
        let (plan_ptr, record_plan) = self.device_plan.device_ptr(cuda_ctx.stream());

        let mut launch_builder = cuda_ctx.stream().launch_builder(&cuda_function);
        launch_builder.arg(&output_ptr);
        launch_builder.arg(&array_len_u64);
        launch_builder.arg(&plan_ptr);

        let config = LaunchConfig {
            grid_dim: (self.len.div_ceil(2048) as u32, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: self.smem_bytes,
        };

        unsafe {
            launch_builder
                .launch(config)
                .map_err(|e| vortex_err!("kernel launch failed: {e}"))?;
        }
        drop((record_output, record_plan));
        Ok(())
    }

    fn run_timed_many(&self, cuda_ctx: &mut CudaExecutionCtx) -> VortexResult<f32> {
        for _ in 0..WARMUP_LAUNCHES {
            self.launch_once(cuda_ctx)?;
        }
        cuda_ctx.stream().synchronize().unwrap();

        let ctx = cuda_ctx.stream().context().clone();
        let start = ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| vortex_err!("{e:?}"))?;
        let stop = ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| vortex_err!("{e:?}"))?;

        start
            .record(cuda_ctx.stream())
            .map_err(|e| vortex_err!("{e:?}"))?;
        for _ in 0..TIMED_LAUNCHES {
            self.launch_once(cuda_ctx)?;
        }
        stop.record(cuda_ctx.stream())
            .map_err(|e| vortex_err!("{e:?}"))?;

        start.elapsed_ms(&stop).map_err(|e| vortex_err!("{e:?}"))
    }
}

#[cuda_available]
fn main() {
    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
        .vortex_expect("ctx")
        .with_dispatch_mode(CudaDispatchMode::DynDispatchOnly);
    let runner = BenchRunner::<u32>::new(&mut cuda_ctx);
    let total_ms = runner.run_timed_many(&mut cuda_ctx).unwrap();
    let avg_us = total_ms * 1_000.0 / TIMED_LAUNCHES as f32;

    // Match the cuda-oxide POC's approximate accounting: packed input read + u32 output write.
    let packed_bytes_per_launch = LEN * BIT_WIDTH as usize / 8;
    let output_bytes_per_launch = LEN * size_of::<u32>();
    let bytes_per_launch = packed_bytes_per_launch + output_bytes_per_launch;
    let total_gib = bytes_per_launch as f64 * TIMED_LAUNCHES as f64 / 1024.0_f64.powi(3);
    let gib_per_s = total_gib / (total_ms as f64 / 1_000.0);

    println!(
        "cu dynamic_dispatch benchmark: case=for_bitpacked_u32 len={} launches={} total={:.3} ms avg={:.3} us throughput={:.2} GiB/s",
        LEN, TIMED_LAUNCHES, total_ms, avg_us, gib_per_s
    );
}

#[cuda_not_available]
fn main() {}
