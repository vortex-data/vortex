//! `SpawnRuntime` exposes two primitives for offloading async work
//! out of an operator's `poll_*` call: `spawn` and `spawn_io`. Each
//! returns a `WorkHandle<T>` the operator holds in its local state
//! and polls on subsequent ticks.
//!
//! Both primitives route to the engine's `DriverIo` smol executor —
//! the same pool that drives Vortex's async I/O. CPU-bound work
//! that operators want to perform should run inline in their
//! `poll_*` body; the pipeline driver itself runs on a dedicated
//! worker thread (see [`crate::physical_plan::pool::Runtime`]) so
//! synchronous CPU bursts don't block any cooperative executor.
//!
//! No cancellation. Dropping a `WorkHandle` abandons the result;
//! the spawned work still runs to completion.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::channel::oneshot;

use crate::EngineError;
use crate::EngineResult;
use crate::physical_plan::driver_io::DriverIo;

/// Priority hint attached to spawned work. v0 ignores it; future
/// schedulers use it to rank work alongside cost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low,
    #[default]
    Normal,
    High,
}

/// Cost hint for `spawn_io`. The runtime may use this to decide
/// when to admit the work — useful for back-pressuring I/O against
/// a memory or bandwidth budget. v0 records but doesn't use it.
#[derive(Clone, Copy, Debug, Default)]
pub struct IoCost {
    pub estimated_bytes: u64,
    pub priority: Priority,
}

impl IoCost {
    pub const fn bytes(estimated_bytes: u64) -> Self {
        Self {
            estimated_bytes,
            priority: Priority::Normal,
        }
    }
}

/// Handle to a piece of spawned work. The owning operator polls
/// the handle on each tick until it returns `Ready`. Dropping the
/// handle without polling is fine — the work continues to
/// completion, the result is discarded.
pub struct WorkHandle<T> {
    rx: oneshot::Receiver<EngineResult<T>>,
}

impl<T> WorkHandle<T> {
    pub fn poll(&mut self, cx: &mut Context<'_>) -> Poll<EngineResult<T>> {
        match self.rx.poll_unpin(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(_)) => Poll::Ready(Err(EngineError::message(
                "spawned work cancelled (sender dropped)",
            ))),
            Poll::Pending => Poll::Pending,
        }
    }

    pub fn try_take(&mut self) -> Option<EngineResult<T>> {
        match self.rx.try_recv() {
            Ok(Some(result)) => Some(result),
            Ok(None) => None,
            Err(_) => Some(Err(EngineError::message(
                "spawned work cancelled (sender dropped)",
            ))),
        }
    }
}

/// Spawn primitives. Cheap to clone (one Arc bump). The runtime
/// constructs one per plan execution and threads `&SpawnRuntime`
/// through each operator's ctx.
///
/// Backed by the DriverIo smol executor. Spawned tasks must be
/// `Send`; futures may migrate across DriverIo worker threads.
#[derive(Clone)]
pub struct SpawnRuntime {
    io: Arc<DriverIo>,
}

impl SpawnRuntime {
    pub fn new(io: Arc<DriverIo>) -> Self {
        Self { io }
    }

    /// Returns the attached `DriverIo`. Operators that need a
    /// `vortex_io::Handle` for async I/O obtain it from here.
    pub fn io(&self) -> &Arc<DriverIo> {
        &self.io
    }

    /// Spawn an arbitrary future. Use when the work is async but
    /// not specifically I/O-bound — e.g. waiting on a barrier, a
    /// channel, or a custom Future.
    pub fn spawn<F, T>(&self, future: F) -> WorkHandle<T>
    where
        F: Future<Output = EngineResult<T>> + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        self.io
            .executor()
            .spawn(async move {
                let result = future.await;
                drop(tx.send(result));
            })
            .detach();
        WorkHandle { rx }
    }

    /// Spawn I/O work. v0: routed through the same executor as
    /// `spawn`. Future: routed to an I/O substrate (io_uring or
    /// equivalent) with admission control sized by `_cost`.
    pub fn spawn_io<F, T>(&self, future: F, _cost: IoCost) -> WorkHandle<T>
    where
        F: Future<Output = EngineResult<T>> + Send + 'static,
        T: Send + 'static,
    {
        self.spawn(future)
    }
}

// Helper for the runtime to drive a future to completion via a
// `Pin`-able boxed form.
pub(crate) type LocalBoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;
