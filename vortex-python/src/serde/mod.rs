pub(crate) mod context;
pub(crate) mod parts;

use pyo3::Bound;
use pyo3::prelude::*;

use crate::install_module;
use crate::serde::context::PyArrayContext;
use crate::serde::parts::PyArrayParts;

/// Register serde functions and classes.
pub(crate) fn init(parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "serde")?;
    parent.add_submodule(&m)?;
    install_module("vortex.serde", &m)?;

    m.add_class::<PyArrayParts>()?;
    m.add_class::<PyArrayContext>()?;

    Ok(())
}
