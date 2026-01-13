// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DeviceRepr;
use cudarc::driver::ValidAsZeroBits;
use dashmap::DashMap;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::session::CudaSession;

/// CUDA execution context.
///
/// Provides access to the CUDA context and stream for kernel execution.
/// Handles memory allocation and data transfers between host and device.
pub struct CudaExecutionCtx {
    context: Arc<CudaContext>,
    session: Arc<CudaSession>,
    array_ctx: vortex_array::ExecutionCtx,
    stream_counter: Arc<AtomicU64>,
    streams: Arc<DashMap<u64, Arc<CudaStream>>>,
}

impl CudaExecutionCtx {
    /// Creates a new CUDA execution context.
    pub fn new(
        context: Arc<CudaContext>,
        session: Arc<CudaSession>,
        array_ctx: vortex_array::ExecutionCtx,
    ) -> Self {
        Self {
            context,
            session,
            array_ctx,
            stream_counter: Arc::new(AtomicU64::new(0)),
            streams: Arc::new(DashMap::new()),
        }
    }

    /// Allocates a typed buffer on the GPU.
    pub fn alloc<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> VortexResult<CudaSlice<T>> {
        self.context
            .default_stream()
            .alloc_zeros::<T>(len)
            .map_err(|e| vortex_err!("Failed to allocate device memory: {}", e))
    }

    /// Copies data from host to device.
    pub fn to_device<T: DeviceRepr>(&self, data: &[T]) -> VortexResult<CudaSlice<T>> {
        self.context
            .default_stream()
            .clone_htod(data)
            .map_err(|e| vortex_err!("Failed to copy to device: {}", e))
    }

    /// Copies data from device to host.
    pub fn to_host<T: DeviceRepr>(&self, buffer: &CudaSlice<T>) -> VortexResult<Vec<T>> {
        self.context
            .default_stream()
            .clone_dtoh(buffer)
            .map_err(|e| vortex_err!("Failed to copy from device: {}", e))
    }

    /// Creates a new CUDA stream with a unique index.
    ///
    /// Returns both the stream and its assigned index.
    pub fn new_stream(&self) -> VortexResult<(u64, Arc<CudaStream>)> {
        let idx = self.stream_counter.fetch_add(1, Ordering::SeqCst);
        let stream = self
            .context
            .new_stream()
            .map_err(|e| vortex_err!("Failed to create CUDA stream: {}", e))?;
        self.streams.insert(idx, stream.clone());
        Ok((idx, stream))
    }

    /// Returns a reference to the CUDA context.
    pub fn context(&self) -> Arc<CudaContext> {
        self.context.clone()
    }

    /// Retrieves a previously created stream by its index.
    pub fn stream(&self, idx: u64) -> Option<Arc<CudaStream>> {
        self.streams.get(&idx).map(|entry| entry.clone())
    }

    /// Synchronizes the stream
    ///
    /// On `synchronize` the host waits for all pending operations of the stream to complete.
    pub fn synchronize(&self) -> VortexResult<()> {
        self.context
            .default_stream()
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

        let Some(support) = ctx.session.executor(&self.encoding_id()) else {
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
