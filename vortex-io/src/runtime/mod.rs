// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod handle;

use std::sync::Arc;

pub use handle::*;
#[cfg(feature = "smol")]
pub mod current;
#[cfg(feature = "smol")]
pub mod multi;
#[cfg(feature = "smol")]
pub mod single;
#[cfg(feature = "smol")]
mod smol;
// TODO(ngates): feature-flag this by Tokio once we add I/O support for runtimes.
#[cfg(not(target_arch = "wasm32"))]
pub mod tokio;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

use futures::future::{BoxFuture, LocalBoxFuture};
use futures::stream::BoxStream;

use crate::file::{IoRequest, IoSource};

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
///
// TODO(ngates): I/O is coming in a future change.
pub(crate) trait Runtime<'rt>: Send + Sync {
    /// Spawns a future to be executed on the runtime.
    ///
    /// The future should continue to be polled in the background by the runtime.
    /// The returned `AbortHandle` may be used to optimistically cancel the future.
    fn spawn(&self, fut: BoxFuture<'rt, ()>) -> AbortHandleRef<'rt>;

    /// Spawns a CPU-bound task for execution on the runtime.
    ///
    /// The returned `AbortHandle` may be used to optimistically cancel the task if it has not
    /// yet started executing.
    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef<'rt>;

    /// Spawns an I/O task for execution on the runtime.
    /// The runtime can choose to invoke the task's `Send` or `!Send` versions.
    fn spawn_io(&self, task: IoTask<'rt>);
}

/// A handle that may be used to optimistically abort a spawned task.
///
/// If dropped, the task should continue to completion.
/// If explicitly aborted, the task should be cancelled if it has not yet started executing.
pub(crate) trait AbortHandle<'rt>: Send + Sync {
    fn abort(self: Box<Self>);
}

pub(crate) type AbortHandleRef<'rt> = Box<dyn AbortHandle<'rt> + 'rt>;

/// A task for driving I/O requests against a source.
pub(crate) struct IoTask<'rt> {
    source: Arc<dyn IoSource>,
    stream: BoxStream<'rt, IoRequest>,
    handle: Handle<'rt>,
}

impl<'rt> IoTask<'rt> {
    pub(crate) fn new(
        source: Arc<dyn IoSource>,
        stream: BoxStream<'rt, IoRequest>,
        handle: Handle<'rt>,
    ) -> Self {
        IoTask {
            source,
            stream,
            handle,
        }
    }

    /// Create a new I/O task for the given source and request stream.
    pub(crate) fn drive_send(self) -> BoxFuture<'rt, ()> {
        self.source.drive_send(self.stream, self.handle.clone())
    }

    /// Create a new I/O task for the given source and request stream that runs on the local thread.
    #[allow(dead_code)] // Used only with smol currently.
    pub(crate) fn drive_local(self) -> LocalBoxFuture<'rt, ()> {
        self.source.drive_local(self.stream, self.handle.clone())
    }
}
