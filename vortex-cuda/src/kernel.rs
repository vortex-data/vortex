// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA kernel loading and management.

use std::fmt::Debug;
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
            // Derive PTX path from module name
            let ptx_path = format!("kernels/{}.ptx", module_name);

            let ptx_content = std::fs::read_to_string(&ptx_path)
                .map_err(|e| vortex_err!("Failed to read PTX file from '{}': {}", ptx_path, e))?;

            // Compile and load the CUDA module.
            let module = cuda_context
                .load_module(Ptx::from_src(&ptx_content))
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
}
