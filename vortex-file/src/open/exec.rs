use std::sync::Arc;

use crate::exec::inline::InlineDriver;
use crate::exec::tokio::TokioDriver;
use crate::exec::ExecDriver;

/// The [`ExecutionMode`] describes how the CPU-bound layout evaluation tasks are executed.
/// Typically, there is one task per file split (row-group).
pub enum ExecutionMode {
    /// Executes the tasks inline as part of polling the returned
    /// [`vortex_array::stream::ArrayStream`]. In other words, uses the same runtime.
    Inline,
    /// Spawns the tasks onto a provided Rayon thread pool.
    // TODO(ngates): feature-flag this dependency.
    RayonThreadPool(Arc<rayon::ThreadPool>),
    /// Spawns the tasks onto a provided Tokio runtime.
    // TODO(ngates): feature-flag this dependency.
    TokioRuntime(tokio::runtime::Handle),
}

impl ExecutionMode {
    pub fn into_driver(self) -> Arc<dyn ExecDriver> {
        match self {
            ExecutionMode::Inline => Arc::new(InlineDriver),
            ExecutionMode::RayonThreadPool(_) => {
                todo!()
            }
            ExecutionMode::TokioRuntime(handle) => Arc::new(TokioDriver(handle)),
        }
    }
}
