// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::Bound;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::prelude::*;
use vortex::array::session::ArraySessionExt;

use crate::SESSION;
use crate::arrays::py::PythonVTable;
use crate::arrays::py::id_from_obj;
use crate::install_module;

/// Register serde functions and classes.
pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "registry")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.registry", &m)?;

    m.add_function(wrap_pyfunction!(register, &m)?)?;

    Ok(())
}

/// Register an array encoding implemented by subclassing `PyArray`.
///
/// It's not currently possible to register a layout encoding from Python.
#[pyfunction]
pub(crate) fn register(cls: &Bound<PyAny>) -> PyResult<()> {
    let id = id_from_obj(cls)?;
    SESSION.arrays().register(id, PythonVTable);
    Ok(())
}
