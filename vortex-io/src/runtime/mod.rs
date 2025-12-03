// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A Vortex runtime provides an abstract way of scheduling mixed I/O and CPU workloads onto the
//! various threading models supported by Vortex.
//!
//! In the future, it may also include a buffer manager or other shared resources.
//!
//! The threading models we currently support are:
//! * Single-threaded: all work is driven on the current thread.
//! * Multi-threaded: work is driven on a pool of threads managed by Vortex.
//! * Worker Pool: work is driven on a pool of threads provided by the caller.
//! * Tokio: work is driven on a Tokio runtime provided by the caller.
//!

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::stream::BoxStream;

use crate::file::IoRequest;
use crate::file::ReadSourceRef;

mod blocking;
pub use blocking::*;
mod handle;
pub use handle::*;

#[cfg(not(target_arch = "wasm32"))]
pub mod current;
#[cfg(not(target_arch = "wasm32"))]
mod pool;
#[cfg(not(target_arch = "wasm32"))]
pub mod single;
#[cfg(not(target_arch = "wasm32"))]
mod smol;
#[cfg(feature = "tokio")]
pub mod tokio;
#[cfg(not(target_arch = "wasm32"))]
pub mod uring;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(test)]
mod tests;

/// Trait used to abstract over different async runtimes.
pub trait Executor: Send + Sync {
    /// Spawns a future to be executed on the runtime.
    ///
    /// The future should continue to be polled in the background by the runtime.
    /// The returned `AbortHandle` may be used to optimistically cancel the future.
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef;

    /// Spawns a CPU-bound task for execution on the runtime.
    ///
    /// The returned `AbortHandle` may be used to optimistically cancel the task if it has not
    /// yet started executing.
    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef;

    /// Spawns a blocking I/O task for execution on the runtime.
    ///
    /// The returned `AbortHandle` may be used to optimistically cancel the task if it has not
    /// yet started executing.
    fn spawn_blocking(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef;

    /// Spawns an I/O task for execution on the runtime.
    /// The runtime can choose to invoke the task's `Send` or `!Send` versions.
    ///
    /// Cancellation is implied by termination of the request stream.
    fn spawn_io(&self, task: IoTask);

    /// Returns a [`LocalExecutor`] view if the runtime supports spawning `!Send` futures.
    ///
    /// Default implementation returns `None` for runtimes that only support `Send` futures.
    fn as_local_executor(&self) -> Option<Arc<dyn LocalExecutor>> {
        None
    }
}

/// Extension trait for runtimes that can build and drive `!Send` futures on a single thread.
///
/// The factory is `Send` so it may be sent to the runtime's thread; the produced future can be
/// `!Send` because it never leaves that thread after creation.
pub trait LocalExecutor: Executor {
    fn spawn_local(&self, f: LocalSpawn) -> AbortHandleRef;
}

/// A boxed future that may be `!Send`.
pub(crate) type LocalBoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// A boxed factory for building a local future on the target runtime thread.
pub(crate) type LocalSpawn = Box<dyn FnOnce() -> LocalBoxFuture<'static, ()> + Send + 'static>;

/// A handle that may be used to optimistically abort a spawned task.
///
/// If dropped, the task should continue to completion.
/// If explicitly aborted, the task should be cancelled if it has not yet started executing.
pub trait AbortHandle: Send + Sync {
    fn abort(self: Box<Self>);
}

pub type AbortHandleRef = Box<dyn AbortHandle>;

/// A task for driving I/O requests against a source.
///
/// Instead of just spawning a future to process requests, we allow each runtime to decide how
/// spawn the driver for the request stream. This allows runtimes to shared, parallelize, further
/// spawn, or otherwise manage the I/O task as they see fit.
///
// NOTE(ngates): We could in theory make IoSource support as_any if we wanted each runtime to implement the
// actual read logic themselves? Not sure yet...
pub struct IoTask {
    pub(crate) source: ReadSourceRef,
    pub(crate) stream: BoxStream<'static, IoRequest>,
}

impl IoTask {
    pub(crate) fn new(source: ReadSourceRef, stream: BoxStream<'static, IoRequest>) -> Self {
        IoTask { source, stream }
    }
}
