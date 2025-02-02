use pyo3::{pyclass, pymethods, PyRef};
use vortex::scalar::BoolScalar;

use crate::scalar::{AsScalarRef, PyScalar, ScalarSubclass};

#[pyclass(name = "BoolScalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyBoolScalar;

impl ScalarSubclass for PyBoolScalar {
    type Scalar<'a> = BoolScalar<'a>;
}

#[pymethods]
impl PyBoolScalar {
    /// Return this value as a Python bool.
    pub fn as_py(self_: PyRef<'_, Self>) -> Option<bool> {
        self_.as_scalar_ref().value()
    }
}
