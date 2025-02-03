use pyo3::prelude::*;

use crate::dtype::PyDType;

/// Concrete class for primitive dtypes.
#[pyclass(name = "PrimitiveDType", module = "vortex", extends=PyDType, frozen)]
pub(crate) struct PyPrimitiveDType;
