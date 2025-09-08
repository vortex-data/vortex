// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::BoxFuture;
use smol::Executor;

use crate::runtime::{AbortHandle, AbortHandleRef, IoTask, Runtime};

impl Runtime for Executor<'static> {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        SmolAbortHandle::new_handle(self.spawn(fut))
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        // For now, we spawn CPU work back onto the same executor.
        SmolAbortHandle::new_handle(self.spawn(async move { task() }))
    }

    fn spawn_io(&self, task: IoTask) {
        self.spawn(task.source.drive_send(task.stream)).detach()
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
