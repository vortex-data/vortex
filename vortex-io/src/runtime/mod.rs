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

use futures::future::BoxFuture;

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
// target_os = "unknown" matches wasm32-unknown-unknown (browser), excluding WASI targets
// where wasm-bindgen's JS interop is not available.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
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

    /// Spawns a future doing IO to be executed on the runtime.
    /// This allows `Executor` implementation to split work between multiple async runtime.
    /// By default, it just calls `Executor::spawn`.
    fn spawn_io(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        self.spawn(fut)
    }

    /// Spawns a CPU-bound task for execution on the runtime.
    ///
    /// The returned `AbortHandle` may be used to optimistically cancel the task if it has not
    /// yet started executing.
    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef;

    /// Spawns a blocking I/O task for execution on the runtime.
    ///
    /// The returned `AbortHandle` may be used to optimistically cancel the task if it has not
    /// yet started executing.
    fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef;
}

/// A handle that may be used to optimistically abort a spawned task.
///
/// If dropped, the task should continue to completion.
/// If explicitly aborted, the task should be cancelled if it has not yet started executing.
pub trait AbortHandle: Send + Sync {
    fn abort(self: Box<Self>);
}

pub type AbortHandleRef = Box<dyn AbortHandle>;
