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
use cudarc::driver::LaunchArgs;
use cudarc::driver::sys::CUevent_flags;
use cudarc::nvrtc::Ptx;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::dash_map::DashMap;

mod arrays;
mod encodings;

pub use arrays::DictExecutor;
pub use encodings::*;

use crate::CudaKernelEvents;

/// Convenience macro to launch a CUDA kernel.
///
/// The kernel gets launched on the stream of the execution context.
///
/// The kernel launch config:
/// LaunchConfig {
///     grid_dim: (array.len() / 2048, 1, 1),
///     block_dim: (64, 1, 1),
///     shared_mem_bytes: 0,
/// };
/// 64 threads are used per block which corresponds to 2 warps.
/// Each block handles 2048 elements. Each thread handles 32 elements.
/// The last block and thread are allowed to have less elements.
///
/// Note: A macro is necessary to unroll the launch builder arguments.
///
/// # Returns
///
/// A pair of CUDA events submitted before and after the kernel.
/// Depending on `CUevent_flags` these events can contain timestamps. Use
/// `CU_EVENT_DISABLE_TIMING` for minimal overhead and `CU_EVENT_DEFAULT` to
/// enable timestamps.
#[macro_export]
macro_rules! launch_cuda_kernel {
    (
        execution_ctx: $ctx:expr,
        module: $module:expr,
        ptypes: $ptypes:expr,
        launch_args: [$($arg:expr),* $(,)?],
        event_recording: $event_recording:expr,
        array_len: $len:expr
    ) => {{
        let cuda_function = $ctx.load_function($module, $ptypes)?;
        let mut launch_builder = $ctx.launch_builder(&cuda_function);

        $(
            launch_builder.arg(&$arg);
        )*

        $crate::launch_cuda_kernel_impl(&mut launch_builder, $event_recording, $len)?
    }};
}

/// Launches a CUDA kernel with the passed launch builder.
///
/// # Arguments
///
/// * `launch_builder` - Configured launch builder
/// * `array_len` - Length of the array to process
///
/// # Returns
///
/// A pair of CUDA events submitted before and after the kernel.
/// Depending on `CUevent_flags` these events can contain timestamps. Use
/// `CU_EVENT_DISABLE_TIMING` for minimal overhead and `CU_EVENT_DEFAULT` to
/// enable timestamps.
pub fn launch_cuda_kernel_impl(
    launch_builder: &mut LaunchArgs,
    event_flags: CUevent_flags,
    array_len: usize,
) -> VortexResult<CudaKernelEvents> {
    let num_chunks = u32::try_from(array_len.div_ceil(2048))?;

    let config = cudarc::driver::LaunchConfig {
        grid_dim: (num_chunks, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };

    launch_builder.record_kernel_launch(event_flags);

    unsafe {
        launch_builder
            .launch(config)
            .map_err(|e| vortex_err!("Failed to launch kernel: {}", e))
            .and_then(|events| {
                events
                    .ok_or_else(|| vortex_err!("CUDA events not recorded"))
                    .map(|(before_launch, after_launch)| CudaKernelEvents {
                        before_launch,
                        after_launch,
                    })
            })
    }
}

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
