// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A Vortex runtime provides an abstract way of scheduling mixed I/O and CPU workloads onto the
//! various threading models supported by Vortex.
//!
//! In the future, it may also include a buffer manager or other shared resources.
//!
//! The threading models we currently support are:
//! * Single-threaded: all work is driven on the current thread. This is also the model to use on
//!   WebAssembly targets that have no JavaScript event loop.
//! * Multi-threaded: work is driven on a pool of threads managed by Vortex.
//! * Worker Pool: work is driven on a pool of threads provided by the caller.
//! * Tokio: work is driven on a Tokio runtime provided by the caller.
//! * WebAssembly: work is driven by `wasm_bindgen_futures`.
//!
//! Callers may also implement [`Executor`] themselves and install it with
//! `RuntimeSessionExt::with_handle`.

use futures::future::BoxFuture;

mod abort;
mod blocking;
pub use blocking::*;
mod handle;
pub use handle::*;
mod platform;
pub mod single;

// Runtimes that need threads, and so are unavailable on WebAssembly.
#[cfg(not(target_arch = "wasm32"))]
mod blocking_pool;
#[cfg(not(target_arch = "wasm32"))]
pub mod current;
#[cfg(not(target_arch = "wasm32"))]
mod pool;
#[cfg(not(target_arch = "wasm32"))]
mod smol;

#[cfg(feature = "tokio")]
pub mod tokio;
// target_os = "unknown" matches browser WebAssembly, excluding WASI targets that do not use this
// browser-specific runtime. Without `wasm-bindgen` there is no JavaScript event loop to schedule
// onto, and callers should drive a `single::SingleThreadRuntime` instead.
#[cfg(all(
    target_arch = "wasm32",
    target_os = "unknown",
    feature = "wasm-bindgen"
))]
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
