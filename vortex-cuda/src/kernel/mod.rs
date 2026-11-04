// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA kernel loading and management.

use std::fmt::Debug;
use std::sync::Arc;

use tracing::trace;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::utils::aliases::dash_map::DashMap;
use vortex_cuda_driver::CudaContext;
use vortex_cuda_driver::CudaFunction;
use vortex_cuda_driver::CudaModule;
use vortex_cuda_driver::CudaStream;
use vortex_cuda_driver::LaunchArgs;
use vortex_cuda_driver::LaunchConfig;
use vortex_cuda_driver::sys::CUevent_flags;

mod arrays;
// Fatbins let the CUDA driver select among the configured real and virtual architectures.
mod embedded_kernels {
    include!(concat!(env!("OUT_DIR"), "/embedded_kernels.rs"));
}
mod encodings;
mod filter;
mod patches;
mod slice;

pub(crate) use arrays::ConstantNumericExecutor;
pub(crate) use arrays::DictExecutor;
pub(crate) use arrays::ListExecutor;
pub(crate) use arrays::MaskedExecutor;
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

/// Loader for embedded CUDA kernels with module caching.
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
    /// * `module_name` - Kernel source name without the `.cu` extension
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
            let module = Self::load_module(module_name, cuda_context)?;
            self.modules
                .insert(module_name.to_string(), Arc::clone(&module));
            module
        };

        // Load the CUDA function from the compiled module.
        module
            .load_function(&kernel_name)
            .map_err(|e| vortex_err!("Failed to load kernel function '{}': {}", kernel_name, e))
    }

    /// Loads a CUDA module from a fatbin embedded in the binary.
    fn load_module(
        module_name: &str,
        cuda_context: &Arc<CudaContext>,
    ) -> VortexResult<Arc<CudaModule>> {
        let fatbin = embedded_kernels::embedded_kernel(module_name).ok_or_else(|| {
            vortex_err!("CUDA module {module_name} was not embedded at build time")
        })?;

        // SAFETY: The build embeds complete fatbins produced by nvcc.
        unsafe { cuda_context.load_module(fatbin) }
            .map_err(|e| vortex_err!("Failed to load embedded CUDA module {module_name}: {e}"))
    }
}
