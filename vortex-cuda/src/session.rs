// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use vortex::array::ArrayId;
use vortex::array::VortexSessionExecute;
use vortex::error::VortexResult;
use vortex::session::Ref;
use vortex::session::SessionExt;
use vortex::utils::aliases::dash_map::DashMap;

use crate::ExportDeviceArray;
use crate::arrow::CanonicalDeviceArrayExport;
use crate::executor::CudaExecute;
pub use crate::executor::CudaExecutionCtx;
use crate::initialize_cuda;
use crate::kernel::KernelLoader;
use crate::stream::VortexCudaStream;
use crate::stream_pool::VortexCudaStreamPool;

/// Default maximum number of streams in the pool.
const DEFAULT_STREAM_POOL_CAPACITY: usize = 4;

/// CUDA session for GPU accelerated execution.
///
/// Maintains a registry of CUDA kernel implementations for array encodings.
/// Holds the CUDA context for all GPU operations and caches compiled PTX modules.
#[derive(Clone, Debug)]
pub struct CudaSession {
    context: Arc<CudaContext>,
    kernels: Arc<DashMap<ArrayId, &'static dyn CudaExecute>>,
    export_device_array: Arc<dyn ExportDeviceArray>,
    kernel_loader: Arc<KernelLoader>,
    stream_pool: Arc<VortexCudaStreamPool>,
}

impl CudaSession {
    /// Creates a new CUDA session with the provided context and default stream pool capacity.
    pub fn new(context: Arc<CudaContext>) -> Self {
        Self::with_stream_pool_capacity(context, DEFAULT_STREAM_POOL_CAPACITY)
    }

    /// Creates a new CUDA session with the provided context and stream pool capacity.
    pub fn with_stream_pool_capacity(
        context: Arc<CudaContext>,
        stream_pool_capacity: usize,
    ) -> Self {
        let stream_pool = Arc::new(VortexCudaStreamPool::new(
            Arc::clone(&context),
            stream_pool_capacity,
        ));
        Self {
            context,
            kernels: Arc::new(DashMap::default()),
            kernel_loader: Arc::new(KernelLoader::new()),
            export_device_array: Arc::new(CanonicalDeviceArrayExport),
            stream_pool,
        }
    }

    /// Creates a new CUDA execution context.
    pub fn create_execution_ctx(
        vortex_session: &vortex::session::VortexSession,
    ) -> VortexResult<CudaExecutionCtx> {
        let stream = vortex_session.cuda_session().stream()?;
        Ok(CudaExecutionCtx::new(
            stream,
            vortex_session.create_execution_ctx(),
        ))
    }

    /// Returns a CUDA stream from the pool.
    ///
    /// The pool reuses existing streams in round-robin fashion.
    pub fn stream(&self) -> VortexResult<VortexCudaStream> {
        self.stream_pool.stream()
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

    /// Loads a CUDA kernel function by module name and type suffixes.
    ///
    /// This is a lower-level version of `load_function` that accepts string suffixes
    /// directly, useful for types that don't have a `PType` (e.g., i128, i256).
    ///
    /// The kernel name is generated as `{module_name}_{suffix[0]}_{suffix[1]}...`
    ///
    /// # Arguments
    ///
    /// * `module_name` - Name of the module (`kernels/{module_name}.ptx`)
    /// * `type_suffixes` - List of type suffix strings to generate kernel name
    ///
    /// # Errors
    ///
    /// Returns an error if PTX file cannot be read or kernel cannot be loaded.
    pub fn load_function_with_suffixes(
        &self,
        module_name: &str,
        type_suffixes: &[&str],
    ) -> VortexResult<cudarc::driver::CudaFunction> {
        self.kernel_loader
            .load_function(module_name, type_suffixes, &self.context)
    }

    /// Get a handle to the exporter that converts Vortex arrays to `ArrowDeviceArray`.
    pub fn export_device_array(&self) -> &Arc<dyn ExportDeviceArray> {
        &self.export_device_array
    }
}

impl Default for CudaSession {
    /// Creates a default CUDA session using device 0, with all GPU array kernels preloaded.
    ///
    /// # Panics
    ///
    /// Panics if CUDA device 0 cannot be initialized.
    fn default() -> Self {
        #[expect(clippy::expect_used)]
        let context = CudaContext::new(0).expect("Failed to initialize CUDA device 0");
        let this = Self::new(context);
        initialize_cuda(&this);
        this
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
