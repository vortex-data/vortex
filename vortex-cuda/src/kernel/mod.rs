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

mod decimal_cast;
pub(crate) use decimal_cast::widen_decimal_values;

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
    /// Return `Some(flags)` to record events around the launch, or `None` to skip both
    /// the events and the [`Self::on_complete`] callback.
    fn event_flags(&self) -> Option<CUevent_flags>;

    /// Called after both events have been enqueued, not necessarily after the GPU has
    /// finished the kernel. [`CudaKernelEvents::duration`] waits for both events.
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
        let Some(flags) = self.event_flags() else {
            return func();
        };

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
    fn event_flags(&self) -> Option<CUevent_flags> {
        None
    }

    fn on_complete(&self, _events: &CudaKernelEvents, _len: usize) -> VortexResult<()> {
        Ok(())
    }
}

/// Launch strategy that records timing and emits trace events.
#[derive(Debug)]
pub struct TracingLaunchStrategy;

impl LaunchStrategy for TracingLaunchStrategy {
    fn event_flags(&self) -> Option<CUevent_flags> {
        Some(CUevent_flags::CU_EVENT_DEFAULT)
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

/// Builds the default launch configuration for `array_len` elements. Each block covers
/// 2,048 elements (64 threads × 32 elements), with no dynamic shared memory. Kernels
/// with a different work distribution must supply their own [`LaunchConfig`].
pub(crate) fn default_launch_config(array_len: usize) -> VortexResult<LaunchConfig> {
    // Kernel launch configuration constants.
    // Must match ELEMENTS_PER_THREAD in CUDA kernels (kernels/*.cu).
    const THREADS_PER_BLOCK: u32 = 64; // 2 warps
    const ELEMENTS_PER_THREAD: u32 = 32;
    const ELEMENTS_PER_BLOCK: usize = (THREADS_PER_BLOCK * ELEMENTS_PER_THREAD) as usize; // 2048

    let num_blocks = u32::try_from(array_len.div_ceil(ELEMENTS_PER_BLOCK))?;

    Ok(LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Launches a CUDA kernel with the passed launch builder and config.
///
/// # Arguments
///
/// * `launch_builder` - Builder with the kernel function and its arguments.
/// * `config` - Grid dimensions, block dimensions, and dynamic shared memory for the launch.
/// * `event_flags` - `None` skips launch events. `Some(CU_EVENT_DISABLE_TIMING)` records
///   events that can be waited on but not timed. Use `Some(CU_EVENT_DEFAULT)` for timestamps.
///
/// # Returns
///
/// The events recorded before and after the kernel, or `None` if no pair was requested.
/// This choice does not affect the driver's tracking of buffer accesses across streams.
pub(crate) fn launch_cuda_kernel_with_config(
    launch_builder: &mut LaunchArgs,
    config: LaunchConfig,
    event_flags: Option<CUevent_flags>,
) -> VortexResult<Option<CudaKernelEvents>> {
    if let Some(flags) = event_flags {
        launch_builder.record_kernel_launch(flags);
    }
    // SAFETY: Each kernel call site supplies the arguments and configuration for its function.
    let events = unsafe { launch_builder.launch(config) }
        .map_err(|e| vortex_err!("Failed to launch kernel: {e}"))?;
    if event_flags.is_none() {
        return Ok(None);
    }
    let (before_launch, after_launch) =
        events.ok_or_else(|| vortex_err!("CUDA events not recorded"))?;
    Ok(Some(CudaKernelEvents {
        before_launch,
        after_launch,
    }))
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
