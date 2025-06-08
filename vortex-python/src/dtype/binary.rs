use pyo3::prelude::*;

use crate::dtype::PyDType;

/// Concrete class for utf8 dtypes.
#[pyclass(name = "BinaryDType", module = "vortex.dtype", extends=PyDType, frozen)]
pub(crate) struct PyBinaryDType;
