use pyo3::{Bound, PyRef, PyResult, pyclass, pymethods};
use vortex::arrays::ConstantEncoding;

use crate::arrays::{AsArrayRef, EncodingSubclass, PyArray};
use crate::scalar::PyScalar;

/// Concrete class for arrays with `vortex.constant` encoding.
#[pyclass(name = "ConstantEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyConstantEncoding;

impl EncodingSubclass for PyConstantEncoding {
    type Encoding = ConstantEncoding;
}

#[pymethods]
impl PyConstantEncoding {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, PyConstantEncoding)
    }

    /// Return the scalar value of the constant array.
    pub fn scalar(self_: PyRef<'_, Self>) -> PyResult<Bound<PyScalar>> {
        PyScalar::init(self_.py(), self_.as_array_ref().scalar().clone())
    }
}
