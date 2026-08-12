// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;
#[cfg(unix)]
use std::ptr;
#[cfg(not(unix))]
use std::sync::LazyLock;
#[cfg(unix)]
use std::sync::OnceLock;
#[cfg(unix)]
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use log::LevelFilter;
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
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

use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::current::CurrentThreadWorkerPool;
use vortex::utils::parallelism::get_available_parallelism;

#[cfg(unix)]
use crate::session::reset_session_handle;

/// The current-thread runtime and its background worker pool.
struct RuntimeState {
    runtime: CurrentThreadRuntime,
    pool: CurrentThreadWorkerPool,
}

impl RuntimeState {
    fn new() -> Self {
        let runtime = CurrentThreadRuntime::new();
        let pool = runtime.new_pool();
        pool.set_workers(requested_workers());
        Self { runtime, pool }
    }
}

#[cfg(not(unix))]
static RUNTIME_STATE: LazyLock<RuntimeState> = LazyLock::new(RuntimeState::new);

/// A process-tagged initialization cell.
///
/// The pid is checked before touching `state`, so a child never waits on a `OnceLock` whose
/// initializer was running on another thread at the instant of `fork(2)`.
#[cfg(unix)]
struct RuntimeSlot {
    pid: u32,
    state: OnceLock<RuntimeState>,
}

#[cfg(unix)]
static RUNTIME_SLOT: AtomicPtr<RuntimeSlot> = AtomicPtr::new(ptr::null_mut());

/// The worker count to give a newly built pool, or [`WORKERS_UNSET`] to derive one.
///
/// Held outside the process-local runtime state so that a forked child inherits the parent's
/// configuration even though it discards the parent's runtime.
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

#[cfg(unix)]
fn runtime_slot() -> &'static RuntimeSlot {
    let pid = std::process::id();
    loop {
        let current = RUNTIME_SLOT.load(Ordering::Acquire);
        if !current.is_null() {
            // SAFETY: Published slots are never freed or replaced in-place. A forked child may
            // publish a new slot, but deliberately leaks the inherited allocation.
            let slot = unsafe { &*current };
            if slot.pid == pid {
                return slot;
            }
        }

        let fresh = Box::into_raw(Box::new(RuntimeSlot {
            pid,
            state: OnceLock::new(),
        }));
        match RUNTIME_SLOT.compare_exchange(current, fresh, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                // SAFETY: `fresh` was just published and published slots are never freed.
                return unsafe { &*fresh };
            }
            Err(_) => {
                // SAFETY: The failed compare-exchange proves `fresh` was never published.
                drop(unsafe { Box::from_raw(fresh) });
            }
        }
    }
}

#[cfg(unix)]
fn runtime_state() -> &'static RuntimeState {
    runtime_slot().state.get_or_init(|| {
        let state = RuntimeState::new();
        // Existing objects hold clones of the shared session, so repoint it before publishing the
        // new runtime state to callers in this process.
        reset_session_handle(state.runtime.handle());
        state
    })
}

#[cfg(not(unix))]
fn runtime_state() -> &'static RuntimeState {
    &RUNTIME_STATE
}

/// The current-thread runtime backing Python Vortex operations.
pub(crate) fn current_runtime() -> CurrentThreadRuntime {
    runtime_state().runtime.clone()
}

/// Runs `f` with the worker pool that drives [`current_runtime`]'s executor in the background.
///
/// The pool is deliberately not handed out by value: `CurrentThreadWorkerPool` shares its state
/// between clones but shuts every worker down on `Drop`, so dropping a temporary clone would stop
/// the whole pool.
pub(crate) fn with_pool<T>(f: impl FnOnce(&CurrentThreadWorkerPool) -> T) -> T {
    f(&runtime_state().pool)
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
