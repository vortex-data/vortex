// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-core runtime helpers and dispatching executor.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::future::BoxFuture;
use vortex_error::vortex_panic;

use crate::runtime::AbortHandleRef;
use crate::runtime::Executor;
use crate::runtime::Handle;
use crate::runtime::IoTask;
use crate::runtime::LocalExecutor;
use crate::runtime::LocalSpawn;

/// An executor that dispatches work across a fixed set of underlying executors.
///
/// Tasks are assigned round-robin; there is no work stealing. This is intended to pair with
/// per-core runtimes where each executor owns a single thread/reactor.
pub(crate) struct HandleSetExecutor {
    executors: Arc<[Arc<dyn Executor>]>,
    picker: AtomicUsize,
}

impl HandleSetExecutor {
    pub(crate) fn new(executors: Vec<Arc<dyn Executor>>) -> Self {
        assert!(!executors.is_empty());
        Self {
            executors: executors.into(),
            picker: AtomicUsize::new(0),
        }
    }

    fn pick(&self) -> &Arc<dyn Executor> {
        let idx = self.picker.fetch_add(1, Ordering::Relaxed);
        // Relaxed is sufficient: we only need uniqueness, not ordering guarantees.
        &self.executors[idx % self.executors.len()]
    }
}

/// A thin wrapper around a set of executors that produces a dispatching [`Handle`].
///
/// This is intended to be backed by per-core runtimes (e.g., io_uring reactors), but it can be
/// constructed from any set of executors for now.
pub(crate) struct HandleSet {
    executors: Arc<[Arc<dyn Executor>]>,
    dispatcher: Arc<HandleSetExecutor>,
}

impl HandleSet {
    pub(crate) fn new(executors: Vec<Arc<dyn Executor>>) -> Self {
        let executors: Arc<[Arc<dyn Executor>]> = executors.into();
        let dispatcher = Arc::new(HandleSetExecutor::new(
            executors.iter().cloned().collect(),
        ));
        Self {
            executors,
            dispatcher,
        }
    }

    /// Returns a handle that round-robins spawned work across the underlying executors.
    pub(crate) fn dispatching_handle(&self) -> Handle {
        let exec: Arc<dyn Executor> = self.dispatcher.clone();
        Handle::new(Arc::downgrade(&exec))
    }

    /// Access to the underlying executors, useful for building per-core pools later.
    pub(crate) fn executors(&self) -> &[Arc<dyn Executor>] {
        &self.executors
    }
}

impl Executor for HandleSetExecutor {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        self.pick().spawn(fut)
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        self.pick().spawn_cpu(task)
    }

    fn spawn_blocking(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        self.pick().spawn_blocking(task)
    }

    fn spawn_io(&self, task: IoTask) {
        self.pick().spawn_io(task)
    }

    fn as_local_executor(&self) -> Option<Arc<dyn LocalExecutor>> {
        self.pick().as_local_executor()
    }
}

impl LocalExecutor for HandleSetExecutor {
    fn spawn_local(&self, f: LocalSpawn) -> AbortHandleRef {
        match self.pick().as_local_executor() {
            Some(exec) => exec.spawn_local(f),
            None => vortex_panic!("LocalExecutor requested but not supported by any underlying executor"),
        }
    }
}
