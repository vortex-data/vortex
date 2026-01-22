// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use vortex_array::VortexSessionExecute;
use vortex_array::vtable::ArrayId;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_utils::aliases::dash_map::DashMap;

use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::kernel::KernelLoader;

/// CUDA session for GPU accelerated execution.
///
/// Maintains a registry of CUDA kernel implementations for array encodings.
/// Holds the CUDA context for all GPU operations and caches compiled PTX modules.
#[derive(Clone, Debug)]
pub struct CudaSession {
    context: Arc<CudaContext>,
    kernels: Arc<DashMap<ArrayId, &'static dyn CudaExecute>>,
    kernel_loader: Arc<KernelLoader>,
}

impl CudaSession {
    /// Creates a new CUDA session with the provided context.
    pub fn new(context: Arc<CudaContext>) -> Self {
        Self {
            context,
            kernels: Arc::new(DashMap::default()),
            kernel_loader: Arc::new(KernelLoader::new()),
        }
    }

    /// Creates a new CUDA execution context.
    pub fn create_execution_ctx(
        vortex_session: vortex_session::VortexSession,
    ) -> VortexResult<CudaExecutionCtx> {
        let stream = vortex_session
            .cuda_session()
            .context
            .new_stream()
            .map_err(|e| vortex_err!("Failed to create CUDA stream: {}", e))?;
        Ok(CudaExecutionCtx::new(
            stream,
            vortex_session.create_execution_ctx(),
        ))
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

    /// Loads a CUDA kernel function by module name and ptypes.
    ///
    /// The kernel name is generated as `{module_name}_{ptype[0]}_{ptype[1]}...`
    ///
    /// # Arguments
    ///
    /// * `module_name` - Name of the module (`kernels/{module_name}.ptx`)
    /// * `ptypes` - List of ptype strings to generate kernel name
    ///
    /// # Errors
    ///
    /// Returns an error if PTX file cannot be read or kernel cannot be loaded.
    pub fn load_function(
        &self,
        module_name: &str,
        ptypes: &[PType],
    ) -> VortexResult<cudarc::driver::CudaFunction> {
        self.kernel_loader
            .load_function(module_name, ptypes, &self.context)
    }
}

impl Default for CudaSession {
    /// Creates a default CUDA session using device 0.
    ///
    /// # Panics
    ///
    /// Panics if CUDA device 0 cannot be initialized.
    fn default() -> Self {
        #[expect(clippy::expect_used)]
        let context = CudaContext::new(0).expect("Failed to initialize CUDA device 0");
        Self::new(context)
    }
}

/// Extension trait for accessing the CUDA session from a Vortex session.
pub trait CudaSessionExt: SessionExt {
    /// Returns the CUDA session.
    fn cuda_session(&self) -> Ref<'_, CudaSession> {
        self.get::<CudaSession>()
    }
}
impl<S: SessionExt> CudaSessionExt for S {}
