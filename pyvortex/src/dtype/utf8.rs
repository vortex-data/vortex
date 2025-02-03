use pyo3::prelude::*;

use crate::dtype::PyDType;

/// Concrete class for utf8 dtypes.
#[pyclass(name = "Utf8DType", module = "vortex", extends=PyDType, frozen)]
pub(crate) struct PyUtf8DType;
