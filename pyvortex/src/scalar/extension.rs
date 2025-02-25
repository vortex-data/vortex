use pyo3::{IntoPyObject, PyObject, PyRef, PyResult, pyclass, pymethods};
use vortex::scalar::ExtScalar;

use crate::PyVortex;
use crate::scalar::{AsScalarRef, PyScalar, ScalarSubclass};

/// Concrete class for extension scalars.
#[pyclass(name = "ExtensionScalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyExtensionScalar;

impl ScalarSubclass for PyExtensionScalar {
    type Scalar<'a> = ExtScalar<'a>;
}

#[pymethods]
impl PyExtensionScalar {
    /// Return the underlying storage scalar.
    pub fn storage(self_: PyRef<'_, Self>) -> PyResult<PyObject> {
        let scalar = self_.as_scalar_ref();
        let storage = scalar.storage();
        PyVortex(&storage)
            .into_pyobject(self_.py())
            .map(|v| v.into())
    }
}
