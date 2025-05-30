pub(crate) mod context;
pub(crate) mod parts;

use pyo3::prelude::*;
use pyo3::{Bound, Python};

use crate::install_module;
use crate::serde::context::PyArrayContext;
use crate::serde::parts::PyArrayParts;

/// Register serde functions and classes.
pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "serde")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.serde", &m)?;

    m.add_class::<PyArrayParts>()?;
    m.add_class::<PyArrayContext>()?;

    Ok(())
}
