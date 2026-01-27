// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaFunction;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchArgs;
use cudarc::driver::result::memcpy_htod_async;
use futures::future::BoxFuture;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Buffer;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::CudaDeviceBuffer;
use crate::CudaSession;
use crate::session::CudaSessionExt;
use crate::stream::await_stream_callback;

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
    stream: Arc<CudaStream>,
    ctx: ExecutionCtx,
    cuda_session: CudaSession,
}

impl CudaExecutionCtx {
    /// Creates a new CUDA execution context.
    pub(crate) fn new(stream: Arc<CudaStream>, ctx: ExecutionCtx) -> Self {
        let cuda_session = ctx.session().cuda_session().clone();
        Self {
            stream,
            ctx,
            cuda_session,
        }
    }

    /// Allocates a typed buffer on the GPU.
    ///
    /// Note: Allocation is async in case the CUDA driver supports this.
    ///
    /// The condition for alloc to be async is support for memory pools:
    /// `CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED`.
    ///
    /// Any kernel submitted to the stream after alloc can safely use the
    /// memory, as operations on the stream are ordered sequentially.
    pub fn device_alloc<T: DeviceRepr>(&self, len: usize) -> VortexResult<CudaSlice<T>> {
        // SAFETY: No safety guarantees for allocations on the GPU.
        unsafe {
            self.stream
                .alloc::<T>(len)
                .map_err(|e| vortex_err!("Failed to allocate device memory: {}", e))
        }
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
    pub fn load_function_ptype(
        &self,
        module_name: &str,
        ptypes: &[PType],
    ) -> VortexResult<CudaFunction> {
        let type_suffixes: Vec<String> = ptypes.iter().map(|ptype| ptype.to_string()).collect();
        self.load_function(
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
    /// This is a lower-level version of `load_function` that accepts string suffixes
    /// directly, useful for types that don't have a `PType` (e.g., i128, i256).
    ///
    /// # Arguments
    ///
    /// * `module_name` - Name of the module (`kernels/{module_name}.ptx`)
    /// * `type_suffixes` - List of type suffix strings for the kernel name
    ///
    /// # Errors
    ///
    /// Returns an error if kernel loading fails.
    pub fn load_function(
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

    /// Copies host data to the device asynchronously.
    ///
    /// Allocates device memory, schedules an async copy, and returns a future
    /// that completes when the copy is finished. The source data is moved into
    /// the future to ensure it remains valid until the copy completes.
    ///
    /// # Arguments
    ///
    /// * `data` - The host data to copy.
    ///
    /// # Returns
    ///
    /// A future that resolves to the device buffer handle when the copy completes.
    pub fn copy_to_device<T, D>(
        &self,
        data: D,
    ) -> VortexResult<BoxFuture<'static, VortexResult<BufferHandle>>>
    where
        T: DeviceRepr + Send + Sync + 'static,
        D: AsRef<[T]> + Send + 'static,
    {
        let host_slice: &[T] = data.as_ref();
        let mut cuda_slice: CudaSlice<T> = self.device_alloc(host_slice.len())?;
        let device_ptr = cuda_slice.device_ptr_mut(&self.stream).0;

        unsafe {
            memcpy_htod_async(device_ptr, host_slice, self.stream.cu_stream())
                .map_err(|e| vortex_err!("Failed to schedule async copy to device: {}", e))?;
        }

        let cuda_buf = CudaDeviceBuffer::new(cuda_slice);
        let stream = Arc::clone(&self.stream);

        Ok(Box::pin(async move {
            await_stream_callback(&stream).await?;

            // Keep source memory alive until copy completes.
            let _keep_alive = data;

            Ok(BufferHandle::new_device(Arc::new(cuda_buf)))
        }))
    }

    /// Moves a host buffer handle to the device asynchronously.
    ///
    /// # Arguments
    ///
    /// * `handle` - The host buffer to move. Must be a host buffer.
    ///
    /// # Returns
    ///
    /// A future that resolves to the device buffer handle when the copy completes.
    pub fn move_to_device<T: DeviceRepr + Send + Sync + 'static>(
        &self,
        handle: BufferHandle,
    ) -> VortexResult<BoxFuture<'static, VortexResult<BufferHandle>>> {
        let host_buffer = handle
            .as_host_opt()
            .ok_or_else(|| vortex_err!("Buffer is not on host"))?;

        let buffer: Buffer<T> = Buffer::from_byte_buffer(host_buffer.clone());
        self.copy_to_device(buffer)
    }

    /// Returns a reference to the underlying CUDA stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

/// Support trait for CUDA-accelerated decompression of arrays.
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
pub trait CudaArrayExt: Array {
    /// Recursively executes the array on CUDA, returning a canonical array.
    ///
    /// If no CUDA support is registered for the encoding, falls back to CPU execution
    /// and logs a debug message.
    async fn execute_cuda(self, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical>;
}

#[async_trait]
impl CudaArrayExt for ArrayRef {
    async fn execute_cuda(self, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical> {
        if self.is_canonical() || self.is_empty() {
            return self.execute(&mut ctx.ctx);
        }

        let Some(support) = ctx.cuda_session.kernel(&self.encoding_id()) else {
            tracing::debug!(
                encoding = %self.encoding_id(),
                "No CUDA support registered for encoding, falling back to CPU execution"
            );
            return self.execute(&mut ctx.ctx);
        };

        tracing::debug!(
            encoding = %self.encoding_id(),
            "Executing array on CUDA device"
        );

        support.execute(self, ctx).await
    }
}

#[cfg(feature = "_test-harness")]
impl CudaExecutionCtx {
    pub fn synchronize_stream(&self) -> VortexResult<()> {
        self.stream
            .synchronize()
            .map_err(|e| vortex_err!("cuda error: {e}"))
    }
}
