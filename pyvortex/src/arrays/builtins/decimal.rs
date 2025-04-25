use pyo3::{PyRef, pyclass, pymethods};
use vortex::arrays::DecimalEncoding;

use crate::arrays::native::{AsArrayRef, EncodingSubclass, PyNativeArray};

/// Concrete class for arrays with `vortex.decimal` encoding.
#[pyclass(name = "DecimalArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyDecimalArray;

impl EncodingSubclass for PyDecimalArray {
    type Encoding = DecimalEncoding;
}

#[pymethods]
impl PyDecimalArray {
    #[getter]
    fn precision(slf: PyRef<Self>) -> u8 {
        slf.as_array_ref().precision()
    }

    #[getter]
    fn scale(slf: PyRef<Self>) -> i8 {
        slf.as_array_ref().scale()
    }
}
