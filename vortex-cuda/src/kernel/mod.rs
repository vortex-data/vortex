// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA kernel loading and management.

use std::fmt::Debug;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::CudaFunction;
use cudarc::driver::CudaModule;
use cudarc::driver::CudaStream;
use cudarc::driver::LaunchArgs;
use cudarc::driver::LaunchConfig;
use cudarc::driver::sys::CUevent_flags;
use cudarc::nvrtc::Ptx;
use tracing::trace;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::utils::aliases::dash_map::DashMap;

mod arrays;
mod encodings;
mod filter;
mod patches;
mod slice;

pub(crate) use arrays::ConstantNumericExecutor;
pub(crate) use arrays::DictExecutor;
pub(crate) use arrays::SharedExecutor;
pub use encodings::ZstdKernelPrep;
pub use encodings::zstd_kernel_prepare;
pub(crate) use encodings::*;
pub(crate) use filter::FilterExecutor;
pub(crate) use patches::types::load_patches_to_gpu;
pub(crate) use slice::SliceExecutor;

use crate::CudaKernelEvents;

/// Trait for customizing kernel launch behavior.
///
/// Implementations can add tracing, async callbacks, or other behavior
/// around kernel launches.
pub trait LaunchStrategy: Debug + Send + Sync + 'static {
    /// Returns the event flags to use for this launch.
    fn event_flags(&self) -> CUevent_flags;

    /// Called after the kernel launch completes with the recorded events.
    fn on_complete(&self, events: &CudaKernelEvents, len: usize) -> VortexResult<()>;
}

/// Extension trait for executing a function which may generate CUDA operations, bracketing them
/// with CUDA events created using the launch strategy system.
pub trait LaunchStrategyExt: LaunchStrategy {
    fn with_strategy<F>(&self, stream: &CudaStream, len: usize, func: F) -> VortexResult<()>
    where
        F: FnMut() -> VortexResult<()>;
}

impl<S: ?Sized + LaunchStrategy> LaunchStrategyExt for S {
    fn with_strategy<F>(&self, stream: &CudaStream, len: usize, mut func: F) -> VortexResult<()>
    where
        F: FnMut() -> VortexResult<()>,
    {
        let flags = self.event_flags();

        let before = stream
            .record_event(Some(flags))
            .map_err(|e| vortex_err!("record_event: {e}"))?;

        func()?;

        let after = stream
            .record_event(Some(flags))
            .map_err(|e| vortex_err!("record_event: {e}"))?;

        self.on_complete(
            &CudaKernelEvents {
                before_launch: before,
                after_launch: after,
            },
            len,
        )?;

        Ok(())
    }
}

/// Default launch strategy with no tracing overhead.
#[derive(Debug)]
pub struct DefaultLaunchStrategy;

impl LaunchStrategy for DefaultLaunchStrategy {
    fn event_flags(&self) -> CUevent_flags {
        CUevent_flags::CU_EVENT_DISABLE_TIMING
    }

    fn on_complete(&self, _events: &CudaKernelEvents, _len: usize) -> VortexResult<()> {
        Ok(())
    }
}

/// Launch strategy that records timing and emits trace events.
#[derive(Debug)]
pub struct TracingLaunchStrategy;

impl LaunchStrategy for TracingLaunchStrategy {
    fn event_flags(&self) -> CUevent_flags {
        CUevent_flags::CU_EVENT_DEFAULT
    }

    fn on_complete(&self, events: &CudaKernelEvents, len: usize) -> VortexResult<()> {
        let duration = events.duration()?;
        trace!(
            execution_nanos = duration.as_nanos(),
            len, "execution completed"
        );
        Ok(())
    }
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
pub(crate) fn launch_cuda_kernel_impl(
    launch_builder: &mut LaunchArgs,
    event_flags: CUevent_flags,
    array_len: usize,
) -> VortexResult<CudaKernelEvents> {
    // Kernel launch configuration constants.
    // Must match ELEMENTS_PER_THREAD in CUDA kernels (kernels/*.cu).
    const THREADS_PER_BLOCK: u32 = 64; // 2 warps
    const ELEMENTS_PER_THREAD: u32 = 32;
    const ELEMENTS_PER_BLOCK: usize = (THREADS_PER_BLOCK * ELEMENTS_PER_THREAD) as usize; // 2048

    let num_blocks = u32::try_from(array_len.div_ceil(ELEMENTS_PER_BLOCK))?;

    let config = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    launch_cuda_kernel_with_config(launch_builder, config, event_flags)
}

/// Launches a CUDA kernel with the passed launch builder and config.
///
/// # Arguments
///
/// * `launch_builder` - Configured launch builder
/// * `config` - Launch config to use
///
/// # Returns
///
/// A pair of CUDA events submitted before and after the kernel.
/// Depending on `CUevent_flags` these events can contain timestamps. Use
/// `CU_EVENT_DISABLE_TIMING` for minimal overhead and `CU_EVENT_DEFAULT` to
/// enable timestamps.
pub(crate) fn launch_cuda_kernel_with_config(
    launch_builder: &mut LaunchArgs,
    config: LaunchConfig,
    event_flags: CUevent_flags,
) -> VortexResult<CudaKernelEvents> {
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

/// Loader for CUDA kernels with module caching.
///
/// Handles loading CUDA module files and functions.
#[derive(Debug)]
pub(crate) struct KernelLoader {
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

    /// Loads CUDA function by module name and type suffixes.
    ///
    /// This is a lower-level version of `load_function` that accepts string suffixes
    /// directly, useful for types that don't have a `PType` (e.g., i128, i256).
    ///
    /// # Arguments
    ///
    /// * `module_name` - Name of the module (`kernels/{module_name}.fatbin`)
    /// * `type_suffixes` - List of type suffix strings for the kernel name (`kernel_i128`)
    /// * `cuda_context` - CUDA context for loading the module
    pub fn load_function(
        &self,
        module_name: &str,
        type_suffixes: &[&str],
        cuda_context: &Arc<CudaContext>,
    ) -> VortexResult<CudaFunction> {
        // Kernel name pattern: `<module>_<type_1>_..<type_n>`.
        let kernel_name = if type_suffixes.is_empty() {
            module_name.to_string()
        } else {
            format!("{}_{}", module_name, type_suffixes.join("_"))
        };

        // Check if module is already cached
        let module = if let Some(entry) = self.modules.get(module_name) {
            Arc::clone(entry.value())
        } else {
            let module_path = Self::path_for_module(module_name);

            // Load the CUDA module.
            let module = cuda_context
                .load_module(Ptx::from_file(&module_path))
                .map_err(|e| {
                    vortex_err!(
                        "Failed to load CUDA module {}, module path {}: {}",
                        module_name,
                        module_path.display(),
                        e
                    )
                })?;

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

    /// Returns the CUDA module file path for a given module name.
    ///
    /// Checks for `VORTEX_CUDA_KERNELS_DIR` environment variable at runtime first,
    /// falling back to the path baked in at compile time by build.rs.
    ///
    /// # Arguments
    ///
    /// * `module_name` - Name of the module
    ///
    /// # Returns
    ///
    /// The full path to the CUDA module file
    fn path_for_module(module_name: &str) -> PathBuf {
        let kernels_dir = std::env::var("VORTEX_CUDA_KERNELS_DIR")
            .unwrap_or_else(|_| env!("VORTEX_CUDA_KERNELS_DIR").to_string());
        Path::new(&kernels_dir).join(format!("{}.fatbin", module_name))
    }
}

#[cfg(test)]
mod tests {

    use cudarc::driver::CudaContext;
    use cudarc::driver::PushKernelArg;
    use vortex::error::VortexExpect;

    use super::KernelLoader;

    /// Test that verifies Rust launch config constants match CUDA kernel constants.
    ///
    /// This test launches a special config_check kernel that reports the kernel-side
    /// constants, then verifies they match the Rust-side constants used in
    /// `launch_cuda_kernel_impl`.
    #[crate::test]
    fn test_kernel_config_matches_rust_config() {
        // These must match the constants in launch_cuda_kernel_impl
        const THREADS_PER_BLOCK: u32 = 64;
        const ELEMENTS_PER_THREAD: u32 = 32;
        const ELEMENTS_PER_BLOCK: u32 = THREADS_PER_BLOCK * ELEMENTS_PER_THREAD;

        let ctx = CudaContext::new(0).expect("failed to create CUDA context");
        let stream = ctx.new_stream().expect("failed to create CUDA stream");

        // Allocate output buffer for 3 u32 values
        // SAFETY: Allocating uninitialized memory that will be written by kernel
        let output = unsafe {
            stream
                .alloc::<u32>(3)
                .expect("failed to allocate output buffer")
        };

        // Load and launch the config_check kernel
        let kernel_loader = KernelLoader::new();
        let function = kernel_loader
            .load_function("config_check", &[], &ctx)
            .vortex_expect("failed to load config_check kernel");

        let config = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut launch_args = stream.launch_builder(&function);
        launch_args.arg(&output);

        // SAFETY: kernel only writes to output buffer
        unsafe {
            launch_args
                .launch(config)
                .expect("failed to launch config_check kernel");
        }

        // Copy results back to host
        let host_output = stream
            .clone_dtoh(&output)
            .expect("failed to copy results to host");

        let kernel_elements_per_thread = host_output[0];
        let kernel_block_dim_x = host_output[1];
        let kernel_elements_per_block = host_output[2];

        assert_eq!(
            kernel_elements_per_thread, ELEMENTS_PER_THREAD,
            "ELEMENTS_PER_THREAD mismatch: kernel has {}, Rust has {}",
            kernel_elements_per_thread, ELEMENTS_PER_THREAD
        );

        assert_eq!(
            kernel_block_dim_x, THREADS_PER_BLOCK,
            "block_dim.x mismatch: kernel received {}, Rust sent {}",
            kernel_block_dim_x, THREADS_PER_BLOCK
        );

        assert_eq!(
            kernel_elements_per_block, ELEMENTS_PER_BLOCK,
            "ELEMENTS_PER_BLOCK mismatch: kernel computed {}, Rust expects {}",
            kernel_elements_per_block, ELEMENTS_PER_BLOCK
        );
    }
}
