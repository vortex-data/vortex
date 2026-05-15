// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;
use std::sync::LazyLock;

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
mod io;
mod iter;
mod object_store;
mod python_repr;
mod registry;
mod runtime;
pub mod scalar;
mod scan;
mod serde;
mod session;
mod store;

use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::runtime::current::CurrentThreadWorkerPool;

/// Shared current-thread runtime backing Python Vortex operations.
pub(crate) static RUNTIME: LazyLock<CurrentThreadRuntime> =
    LazyLock::new(CurrentThreadRuntime::new);

/// Shared worker pool that drives [`RUNTIME`]'s executor in the background.
///
/// On first access, the pool is sized to `VORTEX_MAX_THREADS` (if set to a
/// non-negative integer) or otherwise to `available_parallelism() - 1`.
pub(crate) static POOL: LazyLock<CurrentThreadWorkerPool> = LazyLock::new(|| {
    let pool = RUNTIME.new_pool();
    match std::env::var("VORTEX_MAX_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        Some(n) => pool.set_workers(n),
        None => pool.set_workers_to_available_parallelism(),
    }
    pool
});

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
    io::init(py, m)?;
    iter::init(py, m)?;
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
