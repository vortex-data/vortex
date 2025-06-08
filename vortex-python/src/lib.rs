#![allow(unsafe_op_in_unsafe_fn)]

use std::ops::Deref;
use std::sync::LazyLock;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub(crate) mod arrays;
mod compress;
mod dataset;
pub(crate) mod dtype;
mod expr;
mod file;
mod io;
mod iter;
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

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

static TOKIO_RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    Runtime::new()
        .map_err(VortexError::IOError)
        .vortex_expect("tokio runtime must not fail to start")
});

use pyo3_stub_gen::define_stub_info_gatherer;

/// Vortex is an Apache Arrow-compatible toolkit for working with compressed array data.
#[pymodule(name = "vortex")]
fn entry_point(root_module: &Bound<PyModule>) -> PyResult<()> {
    Python::with_gil(|py| -> PyResult<()> {
        Logger::new(py, Caching::LoggersAndLevels)?
            .filter(LevelFilter::Info)
            .filter_target("my_module::verbose_submodule".to_owned(), LevelFilter::Warn)
            .install()
            .map(|_| ())
            .map_err(|err| PyRuntimeError::new_err(format!("could not initialize logger {err}")))
    })?;

    // Initialize our submodules
    arrays::init(root_module)?;
    compress::init(root_module)?;
    dataset::init(root_module)?;
    dtype::init(root_module)?;
    expr::init(root_module)?;
    file::init(root_module)?;
    io::init(root_module)?;
    iter::init(root_module)?;
    registry::init(root_module)?;
    scalar::init(root_module)?;
    serde::init(root_module)?;

    Ok(())
}

define_stub_info_gatherer!(stub_info);

/// Initialize a module and add it to `sys.modules`.
///
/// Without this, it's not possible to use native submodules as "packages". For example:
///
/// ```pycon
/// >>> from vortex.dtype import bool_  # This fails
/// ModuleNotFoundError: No module named 'vortex.dtype'; 'vortex._lib' is not a package
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
/// >>> from vortex.dtype import bool_
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
