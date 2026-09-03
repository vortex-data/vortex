// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use futures::future::BoxFuture;

use crate::runtime::AbortHandleRef;
use crate::runtime::Executor;
use crate::runtime::abort::TaskAbortHandle;
use crate::runtime::blocking_pool::BlockingPool;

/// The async executor and instance-owned blocking pool backing a current-thread runtime.
pub(crate) struct SmolExecutor {
    executor: smol::Executor<'static>,
    blocking_pool: BlockingPool,
}

impl Default for SmolExecutor {
    fn default() -> Self {
        Self {
            executor: smol::Executor::new(),
            blocking_pool: BlockingPool::default(),
        }
    }
}

impl SmolExecutor {
    pub(crate) fn async_executor(&self) -> &smol::Executor<'static> {
        &self.executor
    }
}

impl Deref for SmolExecutor {
    type Target = smol::Executor<'static>;

    fn deref(&self) -> &Self::Target {
        &self.executor
    }
}

impl Executor for SmolExecutor {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        TaskAbortHandle::new_handle(self.executor.spawn(fut))
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        // For now, we spawn CPU work back onto the same execution.
        TaskAbortHandle::new_handle(self.executor.spawn(async move { task() }))
    }

    fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        self.blocking_pool.spawn(task)
    }
}
