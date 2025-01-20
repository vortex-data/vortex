use std::sync::Arc;

use crate::exec::inline::InlineDriver;
#[cfg(feature = "tokio")]
use crate::exec::tokio::TokioDriver;
use crate::exec::ExecDriver;

/// The [`ExecutionMode`] describes how the CPU-bound layout evaluation tasks are executed.
/// Typically, there is one task per file split (row-group).
pub enum ExecutionMode {
    /// Executes the tasks inline as part of polling the returned
    /// [`vortex_array::stream::ArrayStream`]. In other words, uses the same runtime.
    Inline,
    /// Spawns the tasks onto a provided Rayon thread pool.
    #[cfg(feature = "rayon")]
    RayonThreadPool(Arc<rayon::ThreadPool>),
    /// Spawns the tasks onto a provided Tokio runtime.
    #[cfg(feature = "tokio")]
    TokioRuntime(tokio::runtime::Handle),
}

#[allow(clippy::derivable_impls)]
impl Default for ExecutionMode {
    fn default() -> Self {
        // Default to tokio-specific behavior if its enabled and there's a runtime running.
        #[cfg(feature = "tokio")]
        if let Ok(h) = tokio::runtime::Handle::try_current() {
            return ExecutionMode::TokioRuntime(h);
        }

        ExecutionMode::Inline
    }
}

impl ExecutionMode {
    pub fn into_driver(self, concurrency: usize) -> Arc<dyn ExecDriver> {
        match self {
            ExecutionMode::Inline => Arc::new(InlineDriver::with_concurrency(concurrency)),
            #[cfg(feature = "rayon")]
            ExecutionMode::RayonThreadPool(_) => {
                todo!()
            }
            #[cfg(feature = "tokio")]
            ExecutionMode::TokioRuntime(handle) => Arc::new(TokioDriver {
                handle,
                concurrency,
            }),
        }
    }
}
