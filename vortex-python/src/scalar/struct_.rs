// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::IntoPyObject;
use pyo3::Py;
use pyo3::PyAny;
use pyo3::PyRef;
use pyo3::pyclass;
use pyo3::pymethods;
use vortex::error::vortex_err;
use vortex::scalar::StructScalar;

use crate::PyVortex;
use crate::error::PyVortexError;
use crate::scalar::AsScalarRef;
use crate::scalar::PyScalar;
use crate::scalar::ScalarSubclass;

/// Concrete class for struct scalars.
#[pyclass(name = "StructScalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyStructScalar;

impl ScalarSubclass for PyStructScalar {
    type Scalar<'a> = StructScalar<'a>;
}

#[pymethods]
impl PyStructScalar {
    /// Return the child scalar with the given field name.
    pub fn field(self_: PyRef<'_, Self>, name: &str) -> Result<Py<PyAny>, PyVortexError> {
        let scalar = self_.as_scalar_ref();
        let child = scalar
            .field(name)
            .ok_or_else(|| vortex_err!("No field {name}"))?;
        Ok(PyVortex(&child)
            .into_pyobject(self_.py())
            .map(|v| v.into())?)
    }
}
