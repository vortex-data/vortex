use pyo3::{PyRef, PyResult, pyclass, pymethods};
use vortex::arrays::StructEncoding;
use vortex::variants::StructArrayTrait;

use crate::arrays::PyArrayRef;
use crate::arrays::native::{AsArrayRef, EncodingSubclass, PyNativeArray};

/// Concrete class for arrays with `vortex.struct` encoding.
#[pyclass(name = "StructArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyStructArray;

impl EncodingSubclass for PyStructArray {
    type Encoding = StructEncoding;
}

#[pymethods]
impl PyStructArray {
    /// Returns the given field of the struct array.
    pub fn field(self_: PyRef<'_, Self>, name: &str) -> PyResult<PyArrayRef> {
        let field = self_.as_array_ref().maybe_null_field_by_name(name)?;
        Ok(PyArrayRef::from(field))
    }
}
