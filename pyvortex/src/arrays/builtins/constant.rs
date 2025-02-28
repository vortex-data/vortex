use pyo3::{Bound, PyRef, PyResult, pyclass, pymethods};
use vortex::arrays::ConstantEncoding;

use crate::arrays::{AsArrayRef, EncodingSubclass, PyArray};
use crate::scalar::PyScalar;

/// Concrete class for arrays with `vortex.constant` encoding.
#[pyclass(name = "ConstantArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyConstantArray;

impl EncodingSubclass for PyConstantArray {
    type Encoding = ConstantEncoding;
}

#[pymethods]
impl PyConstantArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &ConstantEncoding, PyConstantArray)
    }

    /// Return the scalar value of the constant array.
    pub fn scalar(self_: PyRef<'_, Self>) -> PyResult<Bound<PyScalar>> {
        PyScalar::init(self_.py(), self_.as_array_ref().scalar().clone())
    }
}
