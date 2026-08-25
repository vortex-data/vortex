// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::AbortHandle;
use crate::runtime::AbortHandleRef;

/// An abort handle for an [`async_executor::Task`].
///
/// This inverts the task's own drop semantics: dropping an `async_executor::Task` cancels it,
/// whereas dropping this handle detaches the task so it runs to completion.
pub(crate) struct TaskAbortHandle<T> {
    task: Option<async_executor::Task<T>>,
}

impl<T: 'static + Send> TaskAbortHandle<T> {
    pub(crate) fn new_handle(task: async_executor::Task<T>) -> AbortHandleRef {
        Box::new(Self { task: Some(task) })
    }
}

impl<T: Send> AbortHandle for TaskAbortHandle<T> {
    fn abort(mut self: Box<Self>) {
        // Aborting a task is done by dropping it.
        drop(self.task.take());
    }
}

impl<T> Drop for TaskAbortHandle<T> {
    fn drop(&mut self) {
        // We prevent the task from being canceled by detaching it.
        if let Some(task) = self.task.take() {
            task.detach()
        }
    }
}
