// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA kernel loading and management.

use std::env;
use std::fmt::Debug;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::CudaFunction;
use cudarc::driver::CudaModule;
use cudarc::nvrtc::Ptx;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::dash_map::DashMap;

/// Loader for CUDA kernels with PTX caching.
///
/// Handles loading PTX files, compiling modules, and loading functions.
#[derive(Debug)]
pub struct KernelLoader {
    /// Cache of loaded CUDA modules, keyed by module name
    modules: DashMap<String, Arc<CudaModule>>,
}

impl KernelLoader {
    /// Creates a new kernel loader.
    pub fn new() -> Self {
        Self {
            modules: DashMap::default(),
        }
    }

    /// Loads CUDA function by module name and ptype(s).
    ///
    /// # Arguments
    ///
    /// * `module_name` - Name of the module (`kernels/{module_name}.ptx`)
    /// * `ptypes` - List of ptype strings for argument passed to the kernel (`kernel_i32`)
    /// * `cuda_context` - CUDA context for loading the module
    pub fn load_function(
        &self,
        module_name: &str,
        ptypes: &[PType],
        cuda_context: &Arc<CudaContext>,
    ) -> VortexResult<CudaFunction> {
        // Kernel name pattern: `<module>_<type_1>_..<type_n>`.
        let kernel_name = if ptypes.is_empty() {
            module_name.to_string()
        } else {
            format!(
                "{}_{}",
                module_name,
                ptypes
                    .iter()
                    .map(|ptype| ptype.to_string())
                    .collect::<Vec<_>>()
                    .join("_")
            )
        };

        // Check if module is already cached
        let module = if let Some(entry) = self.modules.get(module_name) {
            Arc::clone(entry.value())
        } else {
            let ptx_path = Self::ptx_path_for_module(module_name)?;

            // Compile and load the CUDA module.
            let module = cuda_context
                .load_module(Ptx::from_file(&ptx_path))
                .map_err(|e| vortex_err!("Failed to load CUDA module: {}", e))?;

            // Cache the module
            self.modules
                .insert(module_name.to_string(), Arc::clone(&module));

            module
        };

        // Load the CUDA function from the compiled module.
        module
            .load_function(&kernel_name)
            .map_err(|e| vortex_err!("Failed to load kernel function '{}': {}", kernel_name, e))
    }

    /// Returns the PTX file path for a given module name.
    ///
    /// Constructs the path based on the crate's manifest directory.
    ///
    /// # Arguments
    ///
    /// * `module_name` - Name of the module
    ///
    /// # Returns
    ///
    /// The full path to the PTX file
    fn ptx_path_for_module(module_name: &str) -> VortexResult<PathBuf> {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR")
            .map_err(|e| vortex_err!("Failed to get manifest dir: {}", e))?;
        Ok(Path::new(&manifest_dir)
            .join("kernels")
            .join(format!("{}.ptx", module_name)))
    }
}
