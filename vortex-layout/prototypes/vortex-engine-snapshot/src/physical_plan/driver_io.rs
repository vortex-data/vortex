//! `DriverIo`: a smol-backed async I/O substrate for the v2 runtime.
//!
//! Owns a multi-thread `smol::Executor` plus a small pool of worker
//! threads that drive it. Operators that need to perform async I/O
//! (scan sources, in particular) obtain a `vortex_io::runtime::Handle`
//! pointing at this executor and use it to spawn I/O futures.
//! Concretely, the scan source builds Vortex's
//! `ScanBuilder::into_array_stream` against a session whose `Handle`
//! is the one from this DriverIo — so Vortex's per-split decode/I/O
//! tasks run on these threads, while the engine's lane stays on its
//! own `LocalExecutor` and polls a bounded mpsc fed by a forwarding
//! task running here.
//!
//! Shape and lifetime:
//! - One `DriverIo` is created per `run_plan_blocking` call.
//! - The workers run a long-lived `executor.run(<shutdown future>)`
//!   loop. On `Drop`, the shutdown event is fired and workers join.
//! - The struct is cheaply cloneable via `Arc<DriverIo>` (the
//!   primitive type itself is not Clone — clone the Arc).
//!
//! Tuning:
//! - `DriverIo::new(n_workers)` creates `n_workers` threads. Defaults
//!   are chosen by `run_plan_blocking`. With small workloads, 2-4 is
//!   typically enough; larger CPU-bound paths can take more.

use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;

use event_listener::Event;
use smol::Executor as SmolExecutor;
use vortex_io::runtime::Executor as VortexExecutor;
use vortex_io::runtime::Handle as VortexHandle;

/// Multi-thread smol executor + worker pool. See module docs.
pub struct DriverIo {
    /// The smol executor; tasks spawned here run on the worker threads.
    /// Held as a `dyn Executor` so `vortex_io::Handle` can take a
    /// `Weak<dyn Executor>` view of it.
    executor_dyn: Arc<dyn VortexExecutor>,
    /// Same executor, typed, for spawning concrete smol futures from
    /// engine code without going through the `dyn Executor` API.
    executor: Arc<SmolExecutor<'static>>,
    /// Notified on Drop; workers' `executor.run(...)` futures resolve
    /// when this listener fires and the worker exits.
    shutdown: Arc<Event>,
    workers: Vec<JoinHandle<()>>,
}

impl DriverIo {
    /// Build a DriverIo with `n_workers` background threads. Each
    /// worker calls `smol::block_on(executor.run(<shutdown listener>))`,
    /// which means it processes tasks until the shutdown event fires.
    pub fn new(n_workers: usize) -> Arc<Self> {
        let executor: Arc<SmolExecutor<'static>> = Arc::new(SmolExecutor::new());
        let executor_dyn: Arc<dyn VortexExecutor> = executor.clone();
        let shutdown = Arc::new(Event::new());

        let mut workers = Vec::with_capacity(n_workers);
        for i in 0..n_workers {
            let ex = Arc::clone(&executor);
            let sd = Arc::clone(&shutdown);
            let handle = thread::Builder::new()
                .name(format!("driver-io-{i}"))
                .spawn(move || {
                    // `executor.run(future)` polls the executor (running
                    // spawned tasks) while waiting for `future`. When the
                    // shutdown event fires, the listener future resolves
                    // and the worker exits.
                    smol::block_on(ex.run(async {
                        sd.listen().await;
                    }));
                })
                .expect("spawn driver-io worker");
            workers.push(handle);
        }

        Arc::new(Self {
            executor_dyn,
            executor,
            shutdown,
            workers,
        })
    }

    /// Typed smol executor handle. Use when you need to spawn futures
    /// directly from the engine without going through Vortex's
    /// `Handle` indirection.
    pub fn executor(&self) -> &Arc<SmolExecutor<'static>> {
        &self.executor
    }

    /// A `vortex_io::runtime::Handle` pointing at this DriverIo's
    /// executor. Cheaply clonable; safe to attach to any number of
    /// Vortex sessions.
    pub fn vortex_handle(&self) -> VortexHandle {
        VortexHandle::new(Arc::downgrade(&self.executor_dyn))
    }
}

impl Drop for DriverIo {
    fn drop(&mut self) {
        // Wake all worker listeners so their `executor.run(...)`
        // futures resolve and the threads exit.
        self.shutdown.notify(usize::MAX);
        for w in self.workers.drain(..) {
            drop(w.join());
        }
    }
}
