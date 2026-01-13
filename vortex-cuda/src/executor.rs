// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DeviceRepr;
use cudarc::driver::ValidAsZeroBits;
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
#[derive(Clone)]
pub struct ExecutionCtx {
    pub context: Arc<CudaContext>,
    pub session: Arc<CudaSession>,
}

impl ExecutionCtx {
    /// Creates a new CUDA execution context.
    pub(crate) fn new(context: Arc<CudaContext>, session: Arc<CudaSession>) -> Self {
        Self { context, session }
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

    /// Returns a reference to the default CUDA stream.
    pub fn stream(&self) -> Arc<CudaStream> {
        self.context.default_stream().clone()
    }

    /// Returns a reference to the CUDA context.
    pub fn context(&self) -> Arc<CudaContext> {
        self.context.clone()
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
pub trait CudaSupport: 'static + Send + Sync + Debug {
    /// Executes the array on CUDA, returning a canonical array.
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails on the GPU.
    async fn execute_canonical(
        &self,
        array: &ArrayRef,
        ctx: &ExecutionCtx,
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
    async fn execute_cuda(&self, ctx: &ExecutionCtx) -> VortexResult<Canonical>;
}

#[async_trait]
impl CudaArrayExt for ArrayRef {
    async fn execute_cuda(&self, ctx: &ExecutionCtx) -> VortexResult<Canonical> {
        // Short-circuit if already canonical
        if self.is_canonical() {
            return Ok(self.to_canonical());
        }

        let Some(support) = ctx.session.get_executor(&self.encoding_id()) else {
            tracing::debug!(
                encoding = %self.encoding().id(),
                "No CUDA support registered for encoding, falling back to CPU execution"
            );
            return Ok(self.to_canonical());
        };

        tracing::debug!(
            encoding = %self.encoding().id(),
            "Executing array on CUDA device"
        );

        support.execute_canonical(self, ctx).await
    }
}

/// CUDA executor for array execution.
///
/// Manages CUDA device initialization and execution of arrays on GPU.
pub struct CudaExecutor {
    context: Arc<CudaContext>,
    session: Arc<CudaSession>,
}

impl Debug for CudaExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaExecutor")
            .field("device_id", &0usize)
            .finish()
    }
}

impl CudaExecutor {
    /// Creates a new CUDA executor for device 0.
    ///
    /// # Arguments
    ///
    /// * `session` - The CUDA session containing registered kernel implementations
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA device initialization fails.
    pub async fn try_new(session: Arc<CudaSession>) -> VortexResult<Self> {
        Self::try_new_with_device(session, 0).await
    }

    /// Creates a new CUDA executor for the specified device.
    ///
    /// # Arguments
    ///
    /// * `session` - The CUDA session containing registered kernel implementations
    /// * `device_id` - The CUDA device ID to use
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA device initialization fails.
    pub async fn try_new_with_device(
        session: Arc<CudaSession>,
        device_id: usize,
    ) -> VortexResult<Self> {
        let context = CudaContext::new(device_id)
            .map_err(|e| vortex_err!("Failed to initialize CUDA device {}: {}", device_id, e))?;

        tracing::info!(device_id = device_id, "CUDA executor initialized");

        Ok(Self { context, session })
    }

    /// Creates a new execution context for this executor.
    pub fn create_context(&self) -> ExecutionCtx {
        ExecutionCtx::new(self.context.clone(), self.session.clone())
    }

    /// Executes an array to canonical form on GPU.
    ///
    /// # Arguments
    ///
    /// * `array` - The array to execute
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails on the GPU.
    pub async fn execute_canonical(&self, array: ArrayRef) -> VortexResult<Canonical> {
        let ctx = self.create_context();
        array.execute_cuda(&ctx).await
    }

    /// Synchronizes the GPU device, waiting for all pending operations.
    ///
    /// # Errors
    ///
    /// Returns an error if synchronization fails.
    pub fn synchronize(&self) -> VortexResult<()> {
        self.context
            .default_stream()
            .synchronize()
            .map_err(|e| vortex_err!("Failed to synchronize CUDA device: {}", e))
    }

    /// Returns a reference to the CUDA context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Returns a reference to the CUDA session.
    pub fn session(&self) -> &Arc<CudaSession> {
        &self.session
    }
}
