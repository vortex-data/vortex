use pyo3::prelude::*;

use crate::dtype::PyDType;

/// Concrete class for null dtypes.
#[pyclass(name = "NullDType", module = "vortex", extends=PyDType, frozen)]
pub(crate) struct PyNullDType;
