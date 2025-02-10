use pyo3::{pyclass, pymethods, IntoPyObject, PyObject, PyRef, PyResult};
use vortex::scalar::ExtScalar;

use crate::scalar::{AsScalarRef, PyScalar, ScalarSubclass};
use crate::PyVortex;

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
