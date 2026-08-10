// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use log::LevelFilter;
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_log::Caching;
use pyo3_log::Logger;

pub(crate) mod arrays;
pub mod arrow;
pub(crate) mod classes;
#[cfg(feature = "tui")]
mod cli;
mod compress;
mod dataset;
pub(crate) mod dtype;
mod error;
mod expr;
mod file;
mod hf_store;
mod io;
mod iter;
mod object_store;
mod opendal_store;
mod python_repr;
mod registry;
mod runtime;
pub mod scalar;
mod scan;
mod serde;
mod session;
mod store;

use parking_lot::RwLock;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::current::CurrentThreadWorkerPool;
use vortex::utils::parallelism::get_available_parallelism;

use crate::session::reset_session_handle;

/// The current-thread runtime and its background worker pool, tagged with the process that built
/// them.
///
/// `fork(2)` copies only the calling thread, so a forked child inherits a runtime whose worker
/// threads no longer exist: the executor's sleeper list and the pool's handle list both describe
/// phantom threads, and `CurrentThreadWorkerPool::set_workers` therefore believes it already has
/// enough workers and spawns none. Any Vortex operation in the child then blocks forever. Rather
/// than try to repair that state, we tag it with the owning pid and build a completely fresh
/// runtime the first time it is used from a different process. See [`with_state`].
struct RuntimeState {
    /// The process that built this runtime.
    pid: u32,
    runtime: CurrentThreadRuntime,
    pool: CurrentThreadWorkerPool,
}

impl RuntimeState {
    fn new(pid: u32) -> Self {
        let runtime = CurrentThreadRuntime::new();
        let pool = runtime.new_pool();
        pool.set_workers(requested_workers());
        Self { pid, runtime, pool }
    }
}

static RUNTIME_STATE: RwLock<Option<RuntimeState>> = RwLock::new(None);

/// The worker count to give a newly built pool, or [`WORKERS_UNSET`] to derive one.
///
/// Held outside [`RUNTIME_STATE`] so that a forked child inherits the parent's configuration even
/// though it discards the parent's runtime.
static REQUESTED_WORKERS: AtomicUsize = AtomicUsize::new(WORKERS_UNSET);

const WORKERS_UNSET: usize = usize::MAX;

/// The worker count for a new pool: whatever [`runtime::set_worker_threads`] last requested,
/// else `VORTEX_MAX_THREADS` if it is set to a non-negative integer, else
/// `available_parallelism() - 1`.
fn requested_workers() -> usize {
    match REQUESTED_WORKERS.load(Ordering::Relaxed) {
        WORKERS_UNSET => {}
        workers => return workers,
    }
    if let Some(n) = std::env::var("VORTEX_MAX_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        return n;
    }
    get_available_parallelism()
        .map(|n| n.saturating_sub(1).max(1))
        .unwrap_or(1)
}

/// Record the worker count requested by the user, so that a forked child inherits it.
pub(crate) fn set_requested_workers(workers: usize) {
    REQUESTED_WORKERS.store(workers, Ordering::Relaxed);
}

/// Runs `f` with this process's runtime state, building it if this is the first use or if the
/// process has forked since it was built.
///
/// `f` runs under a read lock, so it must not itself call back into this module.
fn with_state<T>(f: impl FnOnce(&RuntimeState) -> T) -> T {
    let pid = std::process::id();
    loop {
        {
            let guard = RUNTIME_STATE.read();
            if let Some(state) = guard.as_ref()
                && state.pid == pid
            {
                return f(state);
            }
        }
        build_state(pid);
    }
}

fn build_state(pid: u32) {
    let handle = {
        let mut guard = RUNTIME_STATE.write();
        if guard.as_ref().is_some_and(|state| state.pid == pid) {
            // Another thread got here first.
            return;
        }
        let state = RuntimeState::new(pid);
        let handle = state.runtime.handle();
        // Never run destructors on state inherited from a fork: dropping the pool takes its own
        // mutex, and dropping the executor takes its task-slab mutex and then runs the destructors
        // of every task it still holds. Any of those may have been held by a thread that did not
        // survive the fork. Leaking is the lesser evil, and only ever happens once per fork.
        if let Some(stale) = guard.replace(state) {
            Box::leak(Box::new(stale));
        }
        handle
    };

    // Repoint the shared session at the new executor. This must happen with the lock released,
    // because building the session reads the runtime.
    reset_session_handle(handle);
}

/// The current-thread runtime backing Python Vortex operations.
pub(crate) fn current_runtime() -> CurrentThreadRuntime {
    with_state(|state| state.runtime.clone())
}

/// Runs `f` with the worker pool that drives [`current_runtime`]'s executor in the background.
///
/// The pool is deliberately not handed out by value: `CurrentThreadWorkerPool` shares its state
/// between clones but shuts every worker down on `Drop`, so dropping a temporary clone would stop
/// the whole pool.
pub(crate) fn with_pool<T>(f: impl FnOnce(&CurrentThreadWorkerPool) -> T) -> T {
    with_state(|state| f(&state.pool))
}

/// Rebuild the runtime in a freshly forked child process.
///
/// Installed as an `os.register_at_fork(after_in_child=...)` handler. The pid check in
/// [`with_state`] alone would be enough, but running the rebuild from the fork handler keeps it
/// on the child's single-threaded startup path instead of racing whichever thread touches Vortex
/// first, and it also warms the global blocking-IO pool (see [`warm_blocking_pool`]).
fn reset_after_fork() {
    // Only rebuild if this process's parent had actually built a runtime; otherwise leave it lazy
    // so that children which never touch Vortex do not pay for worker threads.
    if RUNTIME_STATE.read().is_none() {
        return;
    }
    warm_blocking_pool(&current_runtime());
}

/// Force the process-global blocking-IO thread pool to spawn at least one live thread.
///
/// Vortex reads route through `Handle::spawn_blocking`, which is backed by the `blocking` crate's
/// process-global pool. That pool only grows while `queue.len() > idle_count * 5`, and a forked
/// child inherits an `idle_count` counting threads that no longer exist — so the child's first reads
/// are queued and never run. Submitting no-op tasks pushes the queue past the inherited threshold,
/// which spawns a real thread; that thread then drains the queue and serves subsequent work.
///
/// Probes go out in small batches with a short wait in between: batching keeps the number of probes
/// needed (`5 * inherited_idle_count`) from costing one wait each, while a small batch size keeps the
/// pool from overshooting into dozens of threads before the first one starts draining the queue.
///
/// Completion is signalled through an atomic polled with `thread::sleep`, not a channel. This runs
/// inside an `atfork` child handler, where `std::sync::mpsc` must not be used: its blocking receive
/// parks on `std::thread::park`, which on macOS is backed by a libdispatch semaphore, and libdispatch
/// traps (`SIGTRAP`) when a semaphore created before the fork is waited on in the child.
fn warm_blocking_pool(runtime: &CurrentThreadRuntime) {
    /// Bounds the warmup in case the pool is unrecoverable, e.g. its mutex was held across the fork.
    const MAX_PROBES: usize = 8192;
    /// Probes per completion check.
    const BATCH: usize = 8;
    /// Wait between completion checks. A healthy pool answers the first batch within this window, so
    /// the common case costs one short sleep.
    const POLL: Duration = Duration::from_millis(2);
    /// Extra checks after the last batch, for a pool that is slow rather than broken.
    const GRACE_POLLS: usize = 250;

    let handle = runtime.handle();
    let completed = Arc::new(AtomicUsize::new(0));

    let mut submitted = 0;
    while submitted < MAX_PROBES {
        for _ in 0..BATCH {
            let completed = Arc::clone(&completed);
            handle
                .spawn_blocking(move || {
                    completed.fetch_add(1, Ordering::Release);
                })
                .detach();
        }
        submitted += BATCH;
        std::thread::sleep(POLL);
        if completed.load(Ordering::Acquire) > 0 {
            return;
        }
    }

    for _ in 0..GRACE_POLLS {
        std::thread::sleep(POLL);
        if completed.load(Ordering::Acquire) > 0 {
            return;
        }
    }

    log::warn!(
        "Vortex could not restart its blocking IO thread pool after fork; reads may not complete"
    );
}

/// Install the `after_in_child` fork handler that rebuilds the runtime.
fn register_at_fork(py: Python) -> PyResult<()> {
    let handler = wrap_pyfunction!(_reset_after_fork, py)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "after_in_child"), handler)?;
    py.import("os")?
        .call_method(intern!(py, "register_at_fork"), (), Some(&kwargs))?;
    Ok(())
}

/// Python entry point for [`reset_after_fork`].
#[pyfunction]
fn _reset_after_fork(py: Python) {
    py.detach(reset_after_fork);
}

/// Vortex is an Apache Arrow-compatible toolkit for working with compressed array data.
#[cfg(feature = "extension-module")]
#[pymodule]
fn _lib(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    Python::attach(|py| -> PyResult<()> {
        Logger::new(py, Caching::LoggersAndLevels)?
            .filter(LevelFilter::Info)
            .install()
            .map(|_| ())
            .map_err(|err| PyRuntimeError::new_err(format!("could not initialize logger {err}")))
    })?;

    // `fork(2)` leaves the inherited Vortex runtime unusable, so rebuild it in the child.
    register_at_fork(py)?;

    // Initialize our submodules, living under vortex._lib
    arrays::init(py, m)?;
    #[cfg(feature = "tui")]
    cli::init(py, m)?;
    compress::init(py, m)?;
    dataset::init(py, m)?;
    dtype::init(py, m)?;
    expr::init(py, m)?;
    file::init(py, m)?;
    hf_store::init(py, m)?;
    io::init(py, m)?;
    iter::init(py, m)?;
    opendal_store::init(py, m)?;
    runtime::init(py, m)?;
    store::init(py, m)?;
    registry::init(py, m)?;
    scalar::init(py, m)?;
    serde::init(py, m)?;
    scan::init(py, m)?;

    Ok(())
}

/// Initialize a module and add it to `sys.modules`.
///
/// Without this, it's not possible to use native submodules as "packages". For example:
///
/// ```pycon
/// >>> from vortex._lib.dtype import bool_  # This fails
/// ModuleNotFoundError: No module named 'vortex._lib.dtype'; 'vortex._lib' is not a package
/// ```
///
/// After this, we can import submodules both as modules:
///
/// ```pycon
/// >>> from vortex._lib import dtype
/// ```
///
/// And have direct import access to functions and classes in the submodule:
///
/// ```pycon
/// >>> from vortex._lib.dtype import bool_
/// ```
///
/// See <https://github.com/PyO3/pyo3/issues/759#issuecomment-1811992321>.
pub fn install_module(name: &str, module: &Bound<PyModule>) -> PyResult<()> {
    module
        .py()
        .import("sys")?
        .getattr(intern!(module.py(), "modules"))?
        .set_item(name, module)?;
    // needs to be set *after* `add_submodule()`
    module.setattr(intern!(module.py(), "__name__"), name)?;
    Ok(())
}

/// An adapter struct used to localize trait impls to this crate.
pub struct PyVortex<T>(pub T);

impl<T> From<T> for PyVortex<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T> PyVortex<T> {
    pub fn into_inner(self) -> T {
        self.0
    }

    pub fn inner(&self) -> &T {
        &self.0
    }
}

impl<T> Deref for PyVortex<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
