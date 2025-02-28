use pyo3::{Bound, PyRef, PyResult, pyclass, pymethods};
use vortex::arrays::StructEncoding;
use vortex::variants::StructArrayTrait;

use crate::arrays::{AsArrayRef, EncodingSubclass, PyArray};

/// Concrete class for arrays with `vortex.struct` encoding.
#[pyclass(name = "StructArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyStructArray;

impl EncodingSubclass for PyStructArray {
    type Encoding = StructEncoding;
}

#[pymethods]
impl PyStructArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &StructEncoding, PyStructArray)
    }

    /// Returns the given field of the struct array.
    pub fn field<'py>(self_: PyRef<'py, Self>, name: &str) -> PyResult<Bound<'py, PyArray>> {
        let field = self_.as_array_ref().maybe_null_field_by_name(name)?;
        PyArray::init(self_.py(), field)
    }
}
