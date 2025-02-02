use std::ops::Deref;

use pyo3::prelude::*;
use vortex::dtype::DType;
use vortex::error::vortex_panic;

use crate::dtype::PyDType;

/// Concrete class for struct dtypes.
#[pyclass(name = "StructDType", module = "vortex", extends=PyDType, frozen)]
pub(crate) struct PyStructDType;

#[pymethods]
impl PyStructDType {
    pub fn fields(self_: PyRef<'_, Self>) -> PyResult<Vec<Bound<PyDType>>> {
        let DType::Struct(dtype, _) = self_.as_ref().deref() else {
            vortex_panic!("Not a struct DType");
        };

        let mut fields = Vec::with_capacity(dtype.names().len());
        for dtype in dtype.dtypes() {
            fields.push(PyDType::init(self_.py(), dtype)?);
        }
        Ok(fields)
    }
}
