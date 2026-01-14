// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use vortex_array::vtable::ArrayId;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_utils::aliases::dash_map::DashMap;

use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

/// CUDA session for GPU accelerated execution.
///
/// Maintains a registry of CUDA kernel implementations for array encodings.
/// Holds the CUDA context for all GPU operations.
#[derive(Clone, Debug)]
pub struct CudaSession {
    context: Arc<CudaContext>,
    kernels: Arc<DashMap<ArrayId, &'static dyn CudaExecute>>,
}

impl CudaSession {
    /// Creates a new CUDA session with the provided context.
    pub fn new(context: Arc<CudaContext>) -> Self {
        Self {
            context,
            kernels: Arc::new(DashMap::default()),
        }
    }

    /// Creates a new CUDA execution context.
    pub fn new_execution_ctx(
        &self,
        array_ctx: vortex_array::ExecutionCtx,
    ) -> VortexResult<CudaExecutionCtx> {
        let stream = self
            .context
            .new_stream()
            .map_err(|e| vortex_err!("Failed to create CUDA stream: {}", e))?;
        CudaExecutionCtx::new(stream, Arc::new(self.clone()), array_ctx)
    }

    /// Registers CUDA support for an array encoding.
    ///
    /// # Arguments
    ///
    /// * `array_id` - The encoding ID to register support for
    /// * `executor` - A static reference to the CUDA support implementation
    pub fn register_kernel(&self, array_id: ArrayId, executor: &'static dyn CudaExecute) {
        self.kernels.insert(array_id, executor);
    }

    /// Retrieves the CUDA support implementation for an encoding, if registered.
    ///
    /// # Arguments
    ///
    /// * `array_id` - The encoding ID to look up
    pub fn kernel(&self, array_id: &ArrayId) -> Option<&'static dyn CudaExecute> {
        self.kernels.get(array_id).map(|entry| *entry.value())
    }
}

/// Extension trait for accessing the CUDA session from a Vortex session.
pub trait CudaSessionExt: SessionExt {
    /// Returns the CUDA session.
    fn cuda(&self) -> Option<Ref<'_, CudaSession>> {
        self.get_opt::<CudaSession>()
    }
}
impl<S: SessionExt> CudaSessionExt for S {}
