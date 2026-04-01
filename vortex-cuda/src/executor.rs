// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaFunction;
use cudarc::driver::CudaSlice;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchArgs;
use cudarc::driver::LaunchConfig;
use futures::future::BoxFuture;
use tracing::debug;
use tracing::trace;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::DynArray;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::struct_::StructArrayParts;
use vortex::array::buffer::BufferHandle;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::CudaSession;
use crate::ExportDeviceArray;
use crate::hybrid_dispatch;
use crate::kernel::DefaultLaunchStrategy;
use crate::kernel::LaunchStrategy;
use crate::kernel::LaunchStrategyExt;
use crate::kernel::launch_cuda_kernel_impl;
use crate::kernel::launch_cuda_kernel_with_config;
use crate::session::CudaSessionExt;
use crate::stream::VortexCudaStream;

/// CUDA kernel events recorded before and after kernel launch.
#[derive(Debug)]
pub struct CudaKernelEvents {
    /// Event recorded before kernel launch.
    pub before_launch: CudaEvent,
    /// Event recorded after kernel launch.
    pub after_launch: CudaEvent,
}

impl CudaKernelEvents {
    pub fn duration(&self) -> VortexResult<Duration> {
        self.before_launch
            .elapsed_ms(&self.after_launch) // synchronizes
            .map_err(|e| vortex_err!("failed to get elapsed time: {}", e))
            .map(|f| Duration::from_secs_f32(f / 1000.0))
    }
}

/// CUDA execution context.
///
/// Provides access to the CUDA context and stream for kernel execution.
/// Handles memory allocation and data transfers between host and device.
pub struct CudaExecutionCtx {
    stream: VortexCudaStream,
    ctx: ExecutionCtx,
    cuda_session: CudaSession,
    strategy: Arc<dyn LaunchStrategy>,
}

impl CudaExecutionCtx {
    /// Creates a new CUDA execution context.
    pub(crate) fn new(stream: VortexCudaStream, ctx: ExecutionCtx) -> Self {
        let cuda_session = ctx.session().cuda_session().clone();
        Self {
            stream,
            ctx,
            cuda_session,
            strategy: Arc::new(DefaultLaunchStrategy),
        }
    }

    /// Get a mutable handle to the CPU execution context.
    pub fn execution_ctx(&mut self) -> &mut ExecutionCtx {
        &mut self.ctx
    }

    /// Set the launch strategy for the execution context.
    ///
    /// This can only be set on setup (an "owned" context) and not from within
    /// a kernel execution.
    pub fn with_launch_strategy(mut self, launch_strategy: Arc<dyn LaunchStrategy>) -> Self {
        self.strategy = launch_strategy;
        self
    }

    /// Perform an external kernel launch, with events created and logged via the configured
    /// [`LaunchStrategy`].
    ///
    /// We use CUB and NVCOMP routines, and those don't match the normal `cudarc` entrypoints, so
    /// to inject the configured launch strategy we need to bracket it ourselves.
    pub fn launch_external<F: FnMut() -> VortexResult<()>>(
        &self,
        len: usize,
        function: F,
    ) -> VortexResult<()> {
        self.strategy
            .as_ref()
            .with_strategy(&self.stream, len, function)
    }

    /// Launch a Kernel function with args setup done by the provided `build_args` closure.
    ///
    /// Kernels launched this way will use the default launch configuration, which provides no
    /// shared memory bytes, and uses grid parameters based on the ideal thread block size for
    /// the given `len`.
    pub fn launch_kernel<'a, F>(
        &'a mut self,
        function: &'a CudaFunction,
        len: usize,
        build_args: F,
    ) -> VortexResult<()>
    where
        F: FnOnce(&mut LaunchArgs<'a>),
    {
        let mut launcher = self.launch_builder(function);
        build_args(&mut launcher);

        let events = launch_cuda_kernel_impl(&mut launcher, self.strategy.event_flags(), len)?;
        self.strategy.on_complete(&events, len)?;

        drop(events);

        Ok(())
    }

    /// Launch a function with args provided by the `build_args` closure, with an explicit
    /// [`LaunchConfig`], for kernels which need specific grid and shared memory configuration.
    pub fn launch_kernel_config<'a, F>(
        &'a mut self,
        function: &'a CudaFunction,
        cfg: LaunchConfig,
        len: usize,
        build_args: F,
    ) -> VortexResult<()>
    where
        F: FnOnce(&mut LaunchArgs<'a>),
    {
        let mut launcher = self.launch_builder(function);
        build_args(&mut launcher);

        let events =
            launch_cuda_kernel_with_config(&mut launcher, cfg, self.strategy.event_flags())?;
        self.strategy.on_complete(&events, len)?;

        drop(events);

        Ok(())
    }

    /// Loads a CUDA kernel function by module name and ptype(s).
    ///
    /// # Arguments
    ///
    /// * `module_name` - Name of the module (`kernels/{module_name}.ptx`)
    /// * `ptypes` - List of ptype strings for the kernel name
    ///
    /// # Errors
    ///
    /// Returns an error if kernel loading fails.
    pub fn load_function(&self, module_name: &str, ptypes: &[PType]) -> VortexResult<CudaFunction> {
        let type_suffixes: Vec<String> = ptypes.iter().map(|ptype| ptype.to_string()).collect();
        self.load_function_with_suffixes(
            module_name,
            type_suffixes
                .iter()
                .map(|t| t.as_str())
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    /// Loads a CUDA kernel function by module name and type suffixes.
    ///
    /// This is a lower-level version of [`load_function`][Self::load_function] that accepts
    /// string suffixes directly, useful for types that don't have a `PType` (e.g., i128, i256).
    ///
    /// # Arguments
    ///
    /// * `module_name` - Name of the module (`kernels/{module_name}.ptx`)
    /// * `type_suffixes` - List of type suffix strings for the kernel name
    ///
    /// # Errors
    ///
    /// Returns an error if kernel loading fails.
    pub(crate) fn load_function_with_suffixes(
        &self,
        module_name: &str,
        type_suffixes: &[&str],
    ) -> VortexResult<CudaFunction> {
        self.cuda_session
            .load_function_with_suffixes(module_name, type_suffixes)
    }

    /// Returns a launch builder for a CUDA kernel function.
    ///
    /// Arguments can be added to the kernel launch with `.arg(buffer)`.
    ///
    /// # Arguments
    ///
    /// * `func` - CUDA kernel function to launch
    pub fn launch_builder<'a>(&'a self, func: &'a CudaFunction) -> LaunchArgs<'a> {
        self.stream.launch_builder(func)
    }

    /// Allocates a typed buffer on the GPU.
    pub fn device_alloc<T: DeviceRepr + Send + Sync + 'static>(
        &self,
        len: usize,
    ) -> VortexResult<CudaSlice<T>> {
        self.stream.device_alloc(len)
    }

    /// Copies host data to the device.
    ///
    /// For **pageable** host memory the source is staged synchronously; for
    /// **pinned** memory the transfer is async. In both cases `data` is
    /// kept alive by the returned future until the copy completes.
    pub fn copy_to_device<T, D>(
        &self,
        data: D,
    ) -> VortexResult<BoxFuture<'static, VortexResult<BufferHandle>>>
    where
        T: DeviceRepr + Debug + Send + Sync + 'static,
        D: AsRef<[T]> + Send + 'static,
    {
        self.stream.copy_to_device(data)
    }

    /// Ensures a buffer is resident on the device, copying from host if necessary.
    ///
    /// If the buffer is already on the device it is returned as-is. Otherwise
    /// copies from host to device.
    pub async fn ensure_on_device(&self, handle: BufferHandle) -> VortexResult<BufferHandle> {
        if handle.is_on_device() {
            return Ok(handle);
        }
        let host_buffer = handle
            .as_host_opt()
            .ok_or_else(|| vortex_err!("Buffer is not on host"))?
            .clone();
        self.stream.copy_to_device(host_buffer)?.await
    }

    /// Synchronous variant of [`ensure_on_device`](Self::ensure_on_device).
    ///
    /// Safe to call from within an async executor (no nested `block_on`).
    /// The copy is enqueued on the stream and completes before any subsequent
    /// work on the same stream.
    pub fn ensure_on_device_sync(&self, handle: BufferHandle) -> VortexResult<BufferHandle> {
        if handle.is_on_device() {
            return Ok(handle);
        }
        let host_buffer = handle
            .as_host_opt()
            .ok_or_else(|| vortex_err!("Buffer is not on host"))?
            .clone();
        self.stream.copy_to_device_sync(host_buffer.as_ref())
    }

    /// Returns a reference to the underlying [`VortexCudaStream`].
    ///
    /// Through [`Deref`][std::ops::Deref], this also provides access to the
    /// inner [`Arc<CudaStream>`] and all of cudarc's stream methods.
    pub fn stream(&self) -> &VortexCudaStream {
        &self.stream
    }

    /// Returns the Vortex session backing this CUDA execution context.
    #[cfg(feature = "unstable_encodings")]
    pub(crate) fn session(&self) -> &vortex::session::VortexSession {
        self.ctx.session()
    }

    /// Returns a reference to the CUDA session.
    pub(crate) fn cuda_session(&self) -> &CudaSession {
        &self.cuda_session
    }

    /// Get a handle to the exporter that can convert arrays into `ArrowDeviceArray`.
    pub fn exporter(&self) -> &Arc<dyn ExportDeviceArray> {
        self.cuda_session.export_device_array()
    }

    pub fn synchronize_stream(&self) -> VortexResult<()> {
        self.stream
            .synchronize()
            .map_err(|e| vortex_err!("cuda error: {e}"))
    }
}

/// Support trait for CUDA-accelerated decompression of arrays.
///
/// # Execution model
///
/// Work is enqueued onto a single CUDA stream and executes in FIFO order.
/// Kernel launches are synchronous fire-and-forget: They enqueue work and
/// return immediately. The returned [`Canonical`] may reference device buffers
/// with in-flight writes.
///
/// ## Pageable vs. page-locked (pinned) host memory
///
/// Whether the H2D transfer is asynchronous depends on whether the source
/// memory is page-locked:
///
/// - **Page-locked memory** (allocated via `cuMemAllocHost` / `cudaMallocHost`):
///   the GPU's DMA engine holds a stable physical address for the allocation and
///   can transfer directly without CPU involvement. The call returns immediately
///   and the copy proceeds in parallel with subsequent CPU work.
///
/// - **Pageable memory** (ordinary `malloc` / Rust allocator): CUDA must first
///   stage the data through an internal page-locked bounce buffer, performing a
///   CPU `memcpy` into that buffer before the DMA can begin. The `memcpy_htod_async`
///   call blocks until the staging copy finishes, making the transfer effectively
///   synchronous from the caller's perspective, though the DMA itself still runs
///   on the stream.
///
///
/// ## Synchronisation
///
/// To insert an explicit sync point, use `await_stream_callback`, which completes
/// when all preceding stream work — including in-flight kernels — has finished.
#[async_trait]
pub trait CudaExecute: 'static + Send + Sync + Debug {
    /// Executes the array on CUDA, returning a canonical array.
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails on the GPU.
    async fn execute(&self, array: ArrayRef, ctx: &mut CudaExecutionCtx)
    -> VortexResult<Canonical>;
}

/// Extension trait for executing arrays on CUDA.
#[async_trait]
pub trait CudaArrayExt: DynArray {
    /// Recursively walks the encoding tree, dispatching each layer to its
    /// registered [`CudaExecute`] implementation and returning a canonical array
    /// on the device.
    ///
    /// See [`CudaExecute`] for details on the execution model.
    ///
    /// Falls back to CPU execution if no CUDA support is registered for the
    /// encoding.
    async fn execute_cuda(self, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical>;
}

#[async_trait]
impl CudaArrayExt for ArrayRef {
    #[allow(clippy::unwrap_in_result, clippy::unwrap_used)]
    async fn execute_cuda(self, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical> {
        if self.encoding_id() == Struct::ID {
            let len = self.len();
            let StructArrayParts {
                fields,
                struct_fields,
                validity,
                ..
            } = self.try_into::<Struct>().unwrap().into_parts();

            let mut cuda_fields = Vec::with_capacity(fields.len());
            for field in fields.iter() {
                cuda_fields.push(field.clone().execute_cuda(ctx).await?.into_array());
            }

            return Ok(Canonical::Struct(StructArray::new(
                struct_fields.names().clone(),
                cuda_fields,
                len,
                validity,
            )));
        }

        if self.is_canonical() || self.is_empty() {
            trace!(encoding = ?self.encoding_id(), "skipping canonical");
            return self.execute(&mut ctx.ctx);
        }

        // Try all GPU execution strategies: fused dynamic dispatch, partial
        // fusion with subtree fallbacks, and single-kernel fallback.
        // If none succeed, fall back to CPU execution.
        match hybrid_dispatch::try_gpu_dispatch(&self, ctx).await {
            Ok(canonical) => return Ok(canonical),
            Err(e) => {
                debug!(
                    encoding = %self.encoding_id(),
                    error = %e,
                    "No GPU execution path available, falling back to CPU"
                );
            }
        }

        // TODO(0ax1): Double check whether we need to move buffers back to the host explicitly.
        self.execute(&mut ctx.ctx)
    }
}
