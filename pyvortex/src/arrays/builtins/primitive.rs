use pyo3::{PyRef, pyclass, pymethods};
use vortex::arrays::PrimitiveEncoding;
use vortex::variants::PrimitiveArrayTrait;

use crate::arrays::native::{AsArrayRef, EncodingSubclass, PyNativeArray};
use crate::dtype::PyPType;

/// Concrete class for arrays with `vortex.primitive` encoding.
#[pyclass(name = "PrimitiveArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyPrimitiveArray;

impl EncodingSubclass for PyPrimitiveArray {
    type Encoding = PrimitiveEncoding;
}

#[pymethods]
impl PyPrimitiveArray {
    #[getter]
    fn ptype(slf: PyRef<Self>) -> PyPType {
        slf.as_array_ref().ptype().into()
    }
}
