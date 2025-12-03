// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-core runtime helpers and dispatching executor.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::future::BoxFuture;
use vortex_error::vortex_panic;

use crate::runtime::blocking::BlockingRuntime;
use crate::runtime::AbortHandleRef;
use crate::runtime::current::CurrentThreadRuntime;
use crate::runtime::current::CurrentThreadWorkerPool;
use crate::runtime::Executor;
use crate::runtime::Handle;
use crate::runtime::IoTask;
use crate::runtime::LocalExecutor;
use crate::runtime::LocalSpawn;

#[allow(dead_code)]
/// An executor that dispatches work across a fixed set of underlying executors.
///
/// Tasks are assigned round-robin; there is no work stealing. This is intended to pair with
/// per-core runtimes where each executor owns a single thread/reactor.
pub(crate) struct HandleSetExecutor {
    executors: Arc<[Arc<dyn Executor>]>,
    picker: AtomicUsize,
}

#[allow(dead_code)]
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

#[allow(dead_code)]
/// A thin wrapper around a set of executors that produces a dispatching [`Handle`].
///
/// This is intended to be backed by per-core runtimes (e.g., io_uring reactors), but it can be
/// constructed from any set of executors for now.
pub(crate) struct HandleSet {
    executors: Arc<[Arc<dyn Executor>]>,
    dispatcher: Arc<HandleSetExecutor>,
}

#[allow(dead_code)]
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

/// Create a [`Handle`] that dispatches work round-robin across the provided handles.
///
/// This is useful for thread-per-core runtimes where each handle is tied to a single reactor.
pub fn dispatching_handle(handles: &[Handle]) -> Handle {
    let executors = handles
        .iter()
        .map(|h| h.runtime())
        .collect::<Vec<_>>();
    let set = HandleSet::new(executors);
    set.dispatching_handle()
}

/// A lightweight per-core pool using current-thread runtimes and background workers.
///
/// This is a stopgap until a true io_uring-backed runtime is wired in. Each core owns its own
/// executor driven by a single worker thread, and the exposed handle dispatches round-robin
/// across them.
#[allow(dead_code)]
pub struct PerCoreRuntimePool {
    cores: Vec<CurrentThreadCore>,
    handle: Handle,
}

#[allow(dead_code)]
impl PerCoreRuntimePool {
    /// Build a pool with `cores` runtimes (defaults to available_parallelism if None).
    pub fn new(cores: Option<usize>) -> Self {
        let core_count = cores
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1);

        let cores: Vec<_> = (0..core_count).map(|_| CurrentThreadCore::new()).collect();
        let handles: Vec<_> = cores.iter().map(|c| c.handle()).collect();
        let handle = dispatching_handle(&handles);

        Self { cores, handle }
    }

    /// A handle that spreads work across the per-core runtimes.
    pub fn handle(&self) -> Handle {
        self.handle.clone()
    }
}

struct CurrentThreadCore {
    runtime: CurrentThreadRuntime,
    _pool: CurrentThreadWorkerPool,
}

impl CurrentThreadCore {
    fn new() -> Self {
        let runtime = CurrentThreadRuntime::new();
        let pool = runtime.new_pool();
        pool.set_workers(1);
        Self {
            runtime,
            _pool: pool,
        }
    }

    fn handle(&self) -> Handle {
        self.runtime.handle()
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
