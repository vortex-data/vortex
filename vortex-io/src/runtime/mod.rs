// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::BoxFuture;
use futures::stream::BoxStream;

use crate::file::{IoRequest, IoSourceRef};

mod handle;
pub use handle::*;
#[cfg(not(target_arch = "wasm32"))]
pub mod current;
#[cfg(not(target_arch = "wasm32"))]
pub mod single;
#[cfg(not(target_arch = "wasm32"))]
mod smol;
// TODO(ngates): feature-flag this by Tokio once we add I/O support for runtimes.
#[cfg(not(target_arch = "wasm32"))]
pub mod tokio;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(test)]
mod tests;

/// A Vortex runtime provides an abstract way of scheduling mixed I/O and CPU workloads onto the
/// various threading models supported by Vortex.
///
/// In the future, it may also include a buffer manager or other shared resources.
///
/// The threading models we currently support are:
/// * Single-threaded: all work is driven on the current thread.
/// * Multi-threaded: work is driven on a pool of threads managed by Vortex.
/// * Worker Pool: work is driven on a pool of threads provided by the caller.
/// * Tokio: work is driven on a Tokio runtime provided by the caller.
///
/// Note: users interact with the [`Handle`] API rather than the [`Runtime`] trait.
pub(crate) trait Runtime: Send + Sync {
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

    /// Spawns an I/O task for execution on the runtime.
    /// The runtime can choose to invoke the task's `Send` or `!Send` versions.
    ///
    /// Cancellation is implied by termination of the request stream.
    fn spawn_io(&self, task: IoTask);
}

/// A handle that may be used to optimistically abort a spawned task.
///
/// If dropped, the task should continue to completion.
/// If explicitly aborted, the task should be cancelled if it has not yet started executing.
pub(crate) trait AbortHandle: Send + Sync {
    fn abort(self: Box<Self>);
}

pub(crate) type AbortHandleRef = Box<dyn AbortHandle>;

/// A task for driving I/O requests against a source.
///
/// Instead of just spawning a future to process requests, we allow each runtime to decide how
/// spawn the driver for the request stream. This allows runtimes to shared, parallelize, further
/// spawn, or otherwise manage the I/O task as they see fit.
///
// NOTE(ngates): We could in theory make IoSource support as_any if we wanted each runtime to implement the
// actual read logic themselves? Not sure yet...
pub(crate) struct IoTask {
    pub(crate) source: IoSourceRef,
    pub(crate) stream: BoxStream<'static, IoRequest>,
}

impl IoTask {
    pub(crate) fn new(source: IoSourceRef, stream: BoxStream<'static, IoRequest>) -> Self {
        IoTask { source, stream }
    }
}
