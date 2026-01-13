// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DeviceRepr;
use cudarc::driver::ValidAsZeroBits;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::session::CudaSession;

/// CUDA execution context.
///
/// Provides access to the CUDA context and stream for kernel execution.
/// Handles memory allocation and data transfers between host and device.
pub struct CudaExecutionCtx {
    session: Arc<CudaSession>,
    array_ctx: vortex_array::ExecutionCtx,
    stream: Arc<CudaStream>,
}

impl CudaExecutionCtx {
    /// Creates a new CUDA execution context.
    pub fn new(
        stream: Arc<CudaStream>,
        session: Arc<CudaSession>,
        array_ctx: vortex_array::ExecutionCtx,
    ) -> VortexResult<Self> {
        Ok(Self {
            session,
            array_ctx,
            stream,
        })
    }

    /// Allocates a typed buffer on the GPU.
    pub fn alloc<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> VortexResult<CudaSlice<T>> {
        // SAFETY: No safety guarantees for allocations on the GPU.
        unsafe {
            self.stream
                // Note that alloc is async in case the device and driver support this.
                //
                // The condition for alloc to be async is support for memory pools:
                // `CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED`. Any kernel
                // submitted to the stream after alloc can safely use the memory,
                // as operations on the stream are ordered sequentially.
                .alloc::<T>(len)
                .map_err(|e| vortex_err!("Failed to allocate device memory: {}", e))
        }
    }

    /// Copies data from host to device.
    pub fn to_device<T: DeviceRepr>(&self, data: &[T]) -> VortexResult<CudaSlice<T>> {
        self.stream
            .clone_htod(data)
            .map_err(|e| vortex_err!("Failed to copy to device: {}", e))
    }

    /// Copies data from device to host.
    ///
    /// Returns a `Buffer<T>` with the specified alignment.
    pub fn to_host<T: DeviceRepr>(
        &self,
        buffer: &CudaSlice<T>,
        alignment: Alignment,
    ) -> VortexResult<Buffer<T>> {
        let len = buffer.len();
        let mut host_buffer = BufferMut::<T>::with_capacity_aligned(len, alignment);

        self.stream
            .memcpy_dtoh(buffer, unsafe {
                // SAFETY: We allocated with sufficient capacity and fill the entire buffer.
                host_buffer.set_len(len);
                host_buffer.as_mut_slice()
            })
            .map_err(|e| vortex_err!("Failed to copy from device: {}", e))?;

        Ok(host_buffer.freeze())
    }

    /// Synchronizes the stream
    ///
    /// On `synchronize` the host waits for all pending operations of the stream to complete.
    #[cfg(test)]
    pub fn synchronize(&self) -> VortexResult<()> {
        self.stream
            .synchronize()
            .map_err(|e| vortex_err!("Failed to synchronize device: {}", e))
    }
}

/// Support trait for CUDA-accelerated execution of arrays.
///
/// Implementations provide CUDA-specific execution for array encodings.
#[async_trait]
pub trait CudaExecute: 'static + Send + Sync + Debug {
    /// Executes the array on CUDA, returning a canonical array.
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails on the GPU.
    async fn execute_canonical(
        &self,
        array: ArrayRef,
        ctx: &CudaExecutionCtx,
    ) -> VortexResult<Canonical>;
}

/// Extension trait for executing arrays on CUDA.
#[async_trait]
pub trait CudaArrayExt: Array {
    /// Recursively executes the array on CUDA, returning a canonical array.
    ///
    /// If no CUDA support is registered for the encoding, falls back to CPU execution
    /// and logs a debug message.
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails.
    async fn execute_cuda(self, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical>;
}

#[async_trait]
impl CudaArrayExt for ArrayRef {
    async fn execute_cuda(self, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical> {
        // Short-circuit if already canonical
        if self.is_canonical() {
            return Ok(self.to_canonical());
        }

        let Some(support) = ctx.session.kernel(&self.encoding_id()) else {
            tracing::debug!(
                encoding = %self.encoding().id(),
                "No CUDA support registered for encoding, falling back to CPU execution"
            );
            return self.clone().execute(&mut ctx.array_ctx);
        };

        tracing::debug!(
            encoding = %self.encoding().id(),
            "Executing array on CUDA device"
        );

        support.execute_canonical(self, ctx).await
    }
}
