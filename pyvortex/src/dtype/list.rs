use pyo3::prelude::*;

use crate::dtype::PyDType;

/// Concrete class for list dtypes.
#[pyclass(name = "ListDType", module = "vortex", extends=PyDType, frozen)]
pub(crate) struct PyListDType;
