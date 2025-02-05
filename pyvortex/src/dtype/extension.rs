use pyo3::prelude::*;

use crate::dtype::PyDType;

/// Concrete class for extension dtypes.
#[pyclass(name = "ExtensionDType", module = "vortex", extends=PyDType, frozen)]
pub(crate) struct PyExtensionDType;
