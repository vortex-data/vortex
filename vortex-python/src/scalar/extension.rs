// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::IntoPyObject;
use pyo3::Py;
use pyo3::PyAny;
use pyo3::PyRef;
use pyo3::PyResult;
use pyo3::pyclass;
use pyo3::pymethods;
use vortex::scalar::ExtScalar;

use crate::PyVortex;
use crate::scalar::AsScalarRef;
use crate::scalar::PyScalar;
use crate::scalar::ScalarSubclass;

/// Concrete class for extension scalars.
#[pyclass(name = "ExtensionScalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyExtensionScalar;

impl ScalarSubclass for PyExtensionScalar {
    type Scalar<'a> = ExtScalar<'a>;
}

#[pymethods]
impl PyExtensionScalar {
    /// Return the underlying storage scalar.
    pub fn storage(self_: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let scalar = self_.as_scalar_ref();
        let storage = scalar.storage();
        PyVortex(&storage)
            .into_pyobject(self_.py())
            .map(|v| v.into())
    }
}
