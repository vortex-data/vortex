use pyo3::{pyclass, pymethods, Bound, PyRef, PyResult};
use vortex::array::ConstantEncoding;

use crate::arrays::{ArraySubclass, AsArrayRef, PyArray};
use crate::scalar::PyScalar;

/// Concrete class for arrays with `vortex.constant` encoding.
#[pyclass(name = "ConstantEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyConstantEncoding;

impl ArraySubclass for PyConstantEncoding {
    type Encoding = ConstantEncoding;
}

#[pymethods]
impl PyConstantEncoding {
    /// Return the scalar value of the constant array.
    pub fn scalar(self_: PyRef<'_, Self>) -> PyResult<Bound<PyScalar>> {
        PyScalar::init(self_.py(), self_.as_array_ref().scalar())
    }
}
