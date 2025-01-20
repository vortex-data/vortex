use std::sync::Arc;

#[cfg(feature = "tokio")]
use tokio::runtime::Handle;

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
    TokioRuntime(Handle),
}

impl ExecutionMode {
    pub fn into_driver(self) -> Arc<dyn ExecDriver> {
        match self {
            ExecutionMode::Inline => {
                // Default to tokio-specific behavior if its enabled and there's a runtime running.
                #[cfg(feature = "tokio")]
                match Handle::try_current() {
                    Ok(h) => Arc::new(TokioDriver(h)),
                    Err(_) => Arc::new(InlineDriver),
                }

                #[cfg(not(feature = "tokio"))]
                Arc::new(InlineDriver)
            }
            #[cfg(feature = "rayon")]
            ExecutionMode::RayonThreadPool(_) => {
                todo!()
            }
            #[cfg(feature = "tokio")]
            ExecutionMode::TokioRuntime(handle) => Arc::new(TokioDriver(handle)),
        }
    }
}
