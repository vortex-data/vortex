use pyo3::exceptions::PyIndexError;
use pyo3::{pyclass, pymethods, IntoPy, PyObject, PyRef, PyResult};
use vortex::scalar::ListScalar;

use crate::scalar::{AsScalarRef, PyScalar, ScalarSubclass};
use crate::PyVortex;

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
            .ok_or_else(|| PyIndexError::new_err(format!("Index out of bounds {}", idx)))?;
        Ok(PyVortex(&child).into_py(self_.py()))
    }
}
