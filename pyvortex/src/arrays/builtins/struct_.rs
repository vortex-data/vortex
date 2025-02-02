use pyo3::exceptions::PyKeyError;
use pyo3::{pyclass, pymethods, Bound, PyRef, PyResult};
use vortex::array::StructEncoding;
use vortex::variants::StructArrayTrait;

use crate::arrays::{ArraySubclass, AsArrayRef, PyArray};

/// Concrete class for arrays with `vortex.struct` encoding.
#[pyclass(name = "StructArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyStructArray;

impl ArraySubclass for PyStructArray {
    type Encoding = StructEncoding;
}

#[pymethods]
impl PyStructArray {
    /// Returns the given field of the struct array.
    pub fn field<'py>(self_: PyRef<'py, Self>, name: &str) -> PyResult<Bound<'py, PyArray>> {
        let field = self_
            .as_array_ref()
            .maybe_null_field_by_name(name)
            .ok_or_else(|| PyKeyError::new_err(format!("Field name not found: {}", name)))?;
        PyArray::init(self_.py(), field)
    }
}
