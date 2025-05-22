use pyo3::exceptions::PyIndexError;
use pyo3::{IntoPyObject, PyObject, PyRef, PyResult, pyclass, pymethods};
use vortex::scalar::ListScalar;

use crate::PyVortex;
use crate::scalar::{AsScalarRef, PyScalar, ScalarSubclass};

/// Concrete class for list scalars.
#[pyclass(name = "ListScalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyListScalar;

impl ScalarSubclass for PyListScalar {
    type Scalar<'a> = ListScalar<'a>;
}

#[pymethods]
impl PyListScalar {
    /// Return the child scalar at the given index.
    pub fn element(self_: PyRef<'_, Self>, idx: usize) -> PyResult<PyObject> {
        let scalar = self_.as_scalar_ref();
        let child = scalar
            .element(idx)
            .ok_or_else(|| PyIndexError::new_err(format!("Index out of bounds {idx}")))?;
        PyVortex(&child).into_pyobject(self_.py()).map(|v| v.into())
    }
}
