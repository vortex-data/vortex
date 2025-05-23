use pyo3::prelude::*;

use crate::dtype::PyDType;

/// Concrete class for boolean dtypes.
#[pyclass(name = "BoolDType", module = "vortex", extends=PyDType, frozen)]
pub(crate) struct PyBoolDType;
