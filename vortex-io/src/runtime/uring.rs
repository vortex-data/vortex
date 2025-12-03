// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-core runtime helpers and dispatching executor for io_uring-based runtimes.
//!
//! This module is only compiled on Linux with the `uring` feature enabled.

#![cfg(all(target_os = "linux", feature = "uring"))]

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::thread;

use futures::future::BoxFuture;
use kanal::Receiver;
use kanal::Sender;
use monoio::IoUringDriver;
use monoio::RuntimeBuilder;
use monoio::blocking::DefaultThreadPool;
use vortex_error::vortex_panic;

use crate::runtime::AbortHandle;
use crate::runtime::AbortHandleRef;
use crate::runtime::Executor;
use crate::runtime::Handle;
use crate::runtime::IoTask;
use crate::runtime::LocalExecutor;
use crate::runtime::LocalSpawn;

/// An executor that dispatches work across a fixed set of underlying executors.
///
/// Tasks are assigned round-robin; there is no work stealing. This pairs with per-core runtimes
/// where each executor owns a single thread/reactor.
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
        &self.executors[idx % self.executors.len()]
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
            None => vortex_panic!(
                "LocalExecutor requested but not supported by any underlying executor"
            ),
        }
    }
}

/// A thin wrapper around a set of executors that produces a dispatching [`Handle`].
pub(crate) struct HandleSet {
    executors: Arc<[Arc<dyn Executor>]>,
    dispatcher: Arc<HandleSetExecutor>,
}

#[allow(dead_code)]
impl HandleSet {
    pub(crate) fn new(executors: Vec<Arc<dyn Executor>>) -> Self {
        let executors: Arc<[Arc<dyn Executor>]> = executors.into();
        let dispatcher = Arc::new(HandleSetExecutor::new(executors.iter().cloned().collect()));
        Self {
            executors,
            dispatcher,
        }
    }

    /// Returns a handle that round-robins spawned work across the underlying executors.
    pub(crate) fn handle(&self) -> Handle {
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
/// The underlying dispatcher is leaked to satisfy the `Handle` lifetime; use [`HandleSet`] directly
/// when you need ownership.
pub fn dispatching_handle(handles: &[Handle]) -> Handle {
    let executors = handles.iter().map(|h| h.runtime()).collect::<Vec<_>>();
    let set = Box::leak(Box::new(HandleSet::new(executors)));
    set.handle()
}

/// Messages sent to a per-core runtime thread.
enum Command {
    Spawn(BoxFuture<'static, ()>),
    SpawnLocal(LocalSpawn),
    SpawnCpu(Box<dyn FnOnce() + Send + 'static>),
    SpawnBlocking(Box<dyn FnOnce() + Send + 'static>),
    SpawnIo(IoTask),
}

/// A single-threaded io_uring runtime driven by a background thread.
#[derive(Clone)]
pub struct UringRuntime {
    sender: Sender<Command>,
}

impl UringRuntime {
    pub fn new() -> Self {
        let (sender, receiver) = kanal::unbounded::<Command>();

        thread::Builder::new()
            .name("vortex-uring-runtime".to_string())
            .spawn(move || run_runtime(receiver))
            .expect("failed to spawn uring runtime thread");

        Self { sender }
    }
}

impl Executor for UringRuntime {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        let _ = self.sender.send(Command::Spawn(fut));
        Box::new(NoopAbortHandle)
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        let _ = self.sender.send(Command::SpawnCpu(task));
        Box::new(NoopAbortHandle)
    }

    fn spawn_blocking(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        let _ = self.sender.send(Command::SpawnBlocking(task));
        Box::new(NoopAbortHandle)
    }

    fn spawn_io(&self, task: IoTask) {
        let _ = self.sender.send(Command::SpawnIo(task));
    }

    fn as_local_executor(&self) -> Option<Arc<dyn LocalExecutor>> {
        Some(Arc::new(self.clone()))
    }
}

impl LocalExecutor for UringRuntime {
    fn spawn_local(&self, f: LocalSpawn) -> AbortHandleRef {
        let _ = self.sender.send(Command::SpawnLocal(f));
        Box::new(NoopAbortHandle)
    }
}

fn run_runtime(receiver: Receiver<Command>) {
    // Use the IoUring driver explicitly to avoid ambiguity with feature combinations.
    let mut rt = RuntimeBuilder::<IoUringDriver>::new()
        .enable_timer()
        .attach_thread_pool(Box::new(DefaultThreadPool::new(8)))
        .build()
        .expect("failed to build uring runtime");

    rt.block_on(async move {
        let recv = receiver.as_async();
        futures::pin_mut!(recv);

        while let Ok(cmd) = recv.recv().await {
            match cmd {
                Command::Spawn(fut) => {
                    monoio::spawn(async move {
                        fut.await;
                    });
                }
                Command::SpawnLocal(f) => {
                    monoio::spawn(async move {
                        (f)().await;
                    });
                }
                Command::SpawnCpu(task) | Command::SpawnBlocking(task) => {
                    monoio::spawn_blocking(task);
                }
                Command::SpawnIo(task) => {
                    monoio::spawn(task.source.drive_send(task.stream));
                }
            }
        }
    });
}

struct NoopAbortHandle;

impl AbortHandle for NoopAbortHandle {
    fn abort(self: Box<Self>) {}
}

/// A per-core pool of uring runtimes with a dispatching handle.
#[allow(dead_code)]
pub struct PerCoreUringPool {
    _runtimes: Vec<Arc<UringRuntime>>,
    handle_set: HandleSet,
}

#[allow(dead_code)]
impl PerCoreUringPool {
    pub fn new(cores: Option<usize>) -> Self {
        let core_count = cores
            .or_else(|| thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1);

        let runtimes: Vec<_> = (0..core_count)
            .map(|_| Arc::new(UringRuntime::new()))
            .collect();
        let executors: Vec<Arc<dyn Executor>> = runtimes
            .iter()
            .cloned()
            .map(|rt| rt as Arc<dyn Executor>)
            .collect();
        let handle_set = HandleSet::new(executors);

        Self {
            _runtimes: runtimes,
            handle_set,
        }
    }

    pub fn handle(&self) -> Handle {
        self.handle_set.handle()
    }
}
