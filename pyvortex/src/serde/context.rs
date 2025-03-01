use std::ops::Deref;

use pyo3::{pyclass, pymethods};
use vortex::ArrayContext;

/// An ArrayContext captures an ordered set of encodings.
///
/// In a serialized array, encodings are identified by a positional index into such an
/// :class:`~vortex.ArrayContext`.
#[pyclass(name = "ArrayContext", module = "vortex", frozen)]
pub(crate) struct PyArrayContext(ArrayContext);

impl From<ArrayContext> for PyArrayContext {
    fn from(context: ArrayContext) -> Self {
        Self(context)
    }
}

impl Deref for PyArrayContext {
    type Target = ArrayContext;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PyArrayContext {
    fn __len__(&self) -> usize {
        self.encodings().len()
    }
}
