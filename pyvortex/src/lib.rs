#![allow(unsafe_op_in_unsafe_fn)]

use std::ops::Deref;
use std::sync::LazyLock;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub(crate) mod arrays;
mod compress;
mod dataset;
mod dtype;
mod expr;
mod io;
mod object_store_urls;
mod python_repr;
mod record_batch_reader;
mod registry;
pub(crate) mod scalar;
mod serde;

use log::LevelFilter;
use pyo3_log::{Caching, Logger};
use tokio::runtime::Runtime;
use vortex::error::{VortexError, VortexExpect as _};

pub static TOKIO_RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    Runtime::new()
        .map_err(VortexError::IOError)
        .vortex_expect("tokio runtime must not fail to start")
});

/// Vortex is an Apache Arrow-compatible toolkit for working with compressed array data.
#[pymodule]
fn _lib(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    Python::with_gil(|py| -> PyResult<()> {
        Logger::new(py, Caching::LoggersAndLevels)?
            .filter(LevelFilter::Info)
            .filter_target("my_module::verbose_submodule".to_owned(), LevelFilter::Warn)
            .install()
            .map(|_| ())
            .map_err(|err| PyRuntimeError::new_err(format!("could not initialize logger {}", err)))
    })?;

    // Initialize our submodules, living under vortex._lib
    arrays::init(py, m)?;
    compress::init(py, m)?;
    dataset::init(py, m)?;
    dtype::init(py, m)?;
    expr::init(py, m)?;
    io::init(py, m)?;
    registry::init(py, m)?;
    scalar::init(py, m)?;
    serde::init(py, m)?;

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
        .getattr("modules")?
        .set_item(name, module)?;
    // needs to be set *after* `add_submodule()`
    module.setattr("__name__", name)?;
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
