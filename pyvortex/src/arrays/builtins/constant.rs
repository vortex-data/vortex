use pyo3::{pyclass, pymethods, Bound, PyRef, PyResult};
use vortex::array::ConstantEncoding;

use crate::arrays::{ArraySubclass, AsArrayRef, PyArray};
use crate::scalar::PyScalar;

/// Concrete class for arrays with `vortex.constant` encoding.
#[pyclass(name = "ConstantArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyConstantArray;

impl ArraySubclass for PyConstantArray {
    type Encoding = ConstantEncoding;
}

#[pymethods]
impl PyConstantArray {
    /// Return the scalar value of the constant array.
    pub fn scalar(self_: PyRef<'_, Self>) -> PyResult<Bound<PyScalar>> {
        PyScalar::init(self_.py(), self_.as_array_ref().scalar())
    }
}
