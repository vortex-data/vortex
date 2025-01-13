use std::sync::Arc;

use crate::v2::driver::{BlockOnDriver, ExecDriver};

/// The [`ExecutionMode`] describes how the CPU-bound layout evaluation tasks are executed.
/// Typically, there is one task per file split (row-group).
pub enum ExecutionMode {
    /// Executes the tasks inline as part of polling the returned
    /// [`vortex_array::stream::ArrayStream`].
    BlockOn,
    /// Spawns the tasks onto a provided Rayon thread pool.
    // TODO(ngates): feature-flag this dependency.
    RayonThreadPool(Arc<rayon::ThreadPool>),
    /// Spawns the tasks onto a provided Tokio runtime.
    // TODO(ngates): feature-flag this dependency.
    TokioRuntime(Arc<tokio::runtime::Handle>),
}

impl ExecutionMode {
    pub fn into_driver(self) -> Arc<dyn ExecDriver> {
        match self {
            ExecutionMode::BlockOn => Arc::new(BlockOnDriver),
            ExecutionMode::RayonThreadPool(_) => {
                todo!()
            }
            ExecutionMode::TokioRuntime(_) => {
                todo!()
            }
        }
    }
}
