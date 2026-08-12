// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use futures::future::BoxFuture;

use crate::runtime::AbortHandle;
use crate::runtime::AbortHandleRef;
use crate::runtime::Executor;
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
        SmolAbortHandle::new_handle(self.executor.spawn(fut))
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        // For now, we spawn CPU work back onto the same execution.
        SmolAbortHandle::new_handle(self.executor.spawn(async move { task() }))
    }

    fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        self.blocking_pool.spawn(task)
    }
}

/// An abort handle for a `smol::Task`.
pub(crate) struct SmolAbortHandle<T> {
    task: Option<smol::Task<T>>,
}

impl<T: 'static + Send> SmolAbortHandle<T> {
    pub(crate) fn new_handle(task: smol::Task<T>) -> AbortHandleRef {
        Box::new(Self { task: Some(task) })
    }
}

impl<T: Send> AbortHandle for SmolAbortHandle<T> {
    fn abort(mut self: Box<Self>) {
        // Aborting a smol::Task is done by dropping it.
        drop(self.task.take());
    }
}

impl<T> Drop for SmolAbortHandle<T> {
    fn drop(&mut self) {
        // We prevent the task from being canceled by detaching it.
        if let Some(task) = self.task.take() {
            task.detach()
        }
    }
}
