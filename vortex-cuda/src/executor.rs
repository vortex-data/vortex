// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::mem::size_of;
use std::sync::Arc;

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
    pub fn load_function(&self, module_name: &str, ptypes: &[PType]) -> VortexResult<CudaFunction> {
        self.cuda_session.load_function(module_name, ptypes)
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

    /// Copies host data to the device, returning a [`CudaDeviceBuffer`].
    pub fn copy_buffer_to_device<T: DeviceRepr>(
        &self,
        data: &[T],
    ) -> VortexResult<CudaDeviceBuffer<T>> {
        let cuda_slice = self
            .stream
            .clone_htod(data)
            .map_err(|e| vortex_err!("Failed to copy to device: {}", e))?;
        Ok(CudaDeviceBuffer::new(cuda_slice))
    }

    /// Copies a host buffer to the device asynchronously.
    ///
    /// Allocates device memory, schedules an async copy, and returns a future
    /// that completes when the copy is finished.
    ///
    /// # Arguments
    ///
    /// * `handle` - The host buffer to copy. Must be a host buffer.
    ///
    /// # Returns
    ///
    /// A future that resolves to the device buffer handle when the copy completes.
    pub fn copy_buffer_to_device_async<T: DeviceRepr + Send + Sync + 'static>(
        &self,
        handle: BufferHandle,
    ) -> VortexResult<BoxFuture<'static, VortexResult<BufferHandle>>> {
        let host_buffer = handle
            .as_host_opt()
            .ok_or_else(|| vortex_err!("Buffer is not on host"))?;

        let mut cuda_slice: CudaSlice<T> = self.device_alloc(host_buffer.len() / size_of::<T>())?;
        let device_ptr = cuda_slice.device_ptr_mut(&self.stream).0;

        let typed_buffer: Buffer<T> = Buffer::from_byte_buffer(host_buffer.clone());
        let src_slice: &[T] = typed_buffer.as_slice();

        unsafe {
            memcpy_htod_async(device_ptr, src_slice, self.stream.cu_stream())
                .map_err(|e| vortex_err!("Failed to schedule async copy to device: {}", e))?;
        }

        let cuda_buf = CudaDeviceBuffer::new(cuda_slice);
        let stream = Arc::clone(&self.stream);

        Ok(Box::pin(async move {
            // Await async copy completion using callback-based async wait.
            await_stream_callback(&stream).await?;

            // Keep source memory alive until copy completes.
            let _keep_alive = handle;

            Ok(BufferHandle::new_device(Arc::new(cuda_buf)))
        }))
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
