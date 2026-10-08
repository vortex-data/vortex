// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Runs Vortex IO on a separate Tokio executor for overlap experiments.

use futures::future::BoxFuture;
use vortex::io::runtime::AbortHandleRef;
use vortex::io::runtime::Executor;

/// Routes Vortex IO and compute tasks to their respective Tokio runtimes.
pub struct SplitExecutor {
    /// Executor for general and CPU tasks.
    pub compute: tokio::runtime::Handle,
    /// Executor for asynchronous and blocking IO tasks.
    pub io: tokio::runtime::Handle,
}

impl Executor for SplitExecutor {
    fn spawn(&self, future: BoxFuture<'static, ()>) -> AbortHandleRef {
        Executor::spawn(&self.compute, future)
    }

    fn spawn_io(&self, future: BoxFuture<'static, ()>) -> AbortHandleRef {
        Executor::spawn(&self.io, future)
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        Executor::spawn_cpu(&self.compute, task)
    }

    fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        Executor::spawn_blocking_io(&self.io, task)
    }
}
