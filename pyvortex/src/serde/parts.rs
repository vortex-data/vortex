use std::ops::Deref;

use pyo3::{Bound, PyResult, Python, pyclass, pymethods};
use vortex::serde::ArrayParts;

use crate::arrays::PyArray;
use crate::dtype::PyDType;
use crate::serde::context::PyArrayContext;

/// ArrayParts is a parsed representation of a serialized array.
///
/// It can be decoded into a full array using the `decode` method.
#[pyclass(name = "ArrayParts", module = "vortex", frozen)]
pub(crate) struct PyArrayParts(ArrayParts);

impl Deref for PyArrayParts {
    type Target = ArrayParts;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<ArrayParts> for PyArrayParts {
    fn from(parts: ArrayParts) -> Self {
        Self(parts)
    }
}

#[pymethods]
impl PyArrayParts {
    /// Decode the array parts into a full array.
    ///
    /// # Returns
    ///
    /// The decoded array.
    fn decode<'py>(
        &self,
        py: Python<'py>,
        ctx: &PyArrayContext,
        dtype: PyDType,
        len: usize,
    ) -> PyResult<Bound<'py, PyArray>> {
        PyArray::init(py, self.0.decode(ctx, dtype.into_inner(), len)?)
    }

    /// Fetch the serialized metadata of the array.
    #[getter]
    fn metadata(&self) -> Option<&[u8]> {
        self.0.metadata()
    }

    /// The number of buffers the array has.
    #[getter]
    fn nbuffers(&self) -> usize {
        self.0.nbuffers()
    }

    /// The number of child arrays the array has.
    #[getter]
    fn nchildren(&self) -> usize {
        self.0.nchildren()
    }
}
