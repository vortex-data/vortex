// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::IntoPyObject;
use pyo3::Py;
use pyo3::PyAny;
use pyo3::PyRef;
use pyo3::PyResult;
use pyo3::exceptions::PyIndexError;
use pyo3::pyclass;
use pyo3::pymethods;
use vortex::scalar::ListScalar;

use crate::PyVortex;
use crate::scalar::AsScalarRef;
use crate::scalar::PyScalar;
use crate::scalar::ScalarSubclass;

/// Concrete class for list scalars.
#[pyclass(name = "ListScalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyListScalar;

impl ScalarSubclass for PyListScalar {
    type Scalar<'a> = ListScalar<'a>;
}

#[pymethods]
impl PyListScalar {
    /// Return the child scalar at the given index.
    pub fn element(self_: PyRef<'_, Self>, idx: usize) -> PyResult<Py<PyAny>> {
        let scalar = self_.as_scalar_ref();
        let child = scalar
            .element(idx)
            .ok_or_else(|| PyIndexError::new_err(format!("Index out of bounds {idx}")))?;
        PyVortex(&child).into_pyobject(self_.py()).map(|v| v.into())
    }
}
