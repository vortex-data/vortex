use std::ops::Deref;

use pyo3::prelude::PyAnyMethods;
use pyo3::{Bound, PyAny, PyRef, PyResult, Python, pyclass, pymethods};
use vortex::buffer::ByteBuffer;
use vortex::serde::ArrayParts;

use crate::arrays::PyArrayRef;
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
    /// Parse a serialized array into its parts.
    #[staticmethod]
    fn parse(data: &[u8]) -> PyResult<PyArrayParts> {
        // TODO(ngates): create a buffer from a slice of bytes?
        let buffer = ByteBuffer::copy_from(data);
        Ok(PyArrayParts(ArrayParts::try_from(buffer)?))
    }

    /// Decode the array parts into a full array.
    ///
    /// # Returns
    ///
    /// The decoded array.
    fn decode(&self, ctx: &PyArrayContext, dtype: PyDType, len: usize) -> PyResult<PyArrayRef> {
        Ok(PyArrayRef::from(self.0.decode(
            ctx,
            dtype.into_inner(),
            len,
        )?))
    }

    /// Fetch the serialized metadata of the array.
    #[getter]
    fn metadata(&self) -> &[u8] {
        self.0.metadata()
    }

    /// The number of buffers the array has.
    #[getter]
    fn nbuffers(&self) -> usize {
        self.0.nbuffers()
    }

    /// Return the buffers of the array, currently as :class:`pyarrow.Buffer`.
    // TODO(ngates): ideally we'd use the buffer protocol, but that requires the 3.11 ABI.
    #[getter]
    fn buffers<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        if slf.nbuffers() == 0 {
            return Ok(Vec::new());
        }

        let pyarrow = py.import("pyarrow")?;

        let mut buffers = Vec::with_capacity(slf.nbuffers());
        for buffer in (0..slf.nbuffers()).map(|i| slf.buffer(i)) {
            let buffer: ByteBuffer = buffer?;

            let addr = buffer.as_ptr() as usize;
            let size = buffer.len();
            let base = &slf;
            let pa_buffer = pyarrow.call_method("foreign_buffer", (addr, size, base), None)?;
            buffers.push(pa_buffer);
        }

        Ok(buffers)
    }

    /// The number of child arrays the array has.
    #[getter]
    fn nchildren(&self) -> usize {
        self.0.nchildren()
    }

    /// Return the child :class:`~vortex.ArrayParts` of the array.
    #[getter]
    fn children(&self) -> Vec<PyArrayParts> {
        (0..self.0.nchildren())
            .map(|idx| self.0.child(idx))
            .map(PyArrayParts)
            .collect()
    }
}
