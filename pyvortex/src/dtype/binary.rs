use pyo3::prelude::*;

use crate::dtype::PyDType;

/// Concrete class for utf8 dtypes.
#[pyclass(name = "BinaryDType", module = "vortex", extends=PyDType, frozen)]
pub(crate) struct PyBinaryDType;
