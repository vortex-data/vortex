// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::PyResult;
use pyo3::Python;
use vortex::vtable::ArrayVTableExt;
use vortex_array::session::ArraySessionExt;

use crate::arrays::py::PythonVTable;
use crate::install_module;
use crate::SESSION;

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
pub(crate) fn register(cls: PythonVTable) -> PyResult<()> {
    let vtable = ArrayVTableExt::into_vtable(cls);
    SESSION.arrays().register(vtable);
    Ok(())
}
