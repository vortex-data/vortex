// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use pyo3::Bound;
use pyo3::PyAny;
use pyo3::PyRef;
use pyo3::Python;
use pyo3::intern;
use pyo3::prelude::PyAnyMethods;
use pyo3::pyclass;
use pyo3::pymethods;
use vortex::array::serde::ArrayParts;
use vortex::buffer::ByteBuffer;

use crate::SESSION;
use crate::arrays::PyArrayRef;
use crate::dtype::PyDType;
use crate::error::PyVortexResult;
use crate::serde::context::PyReadContext;

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
    fn parse(data: &[u8]) -> PyVortexResult<PyArrayParts> {
        // TODO(ngates): create a buffer from a slice of bytes?
        let buffer = ByteBuffer::copy_from(data);
        Ok(PyArrayParts(ArrayParts::try_from(buffer)?))
    }

    /// Decode the array parts into a full array.
    ///
    /// # Returns
    ///
    /// The decoded array.
    fn decode(
        &self,
        ctx: &PyReadContext,
        dtype: PyDType,
        len: usize,
    ) -> PyVortexResult<PyArrayRef> {
        Ok(PyArrayRef::from(self.0.decode(
            dtype.inner(),
            len,
            ctx,
            &SESSION,
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
    fn buffers<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyVortexResult<Vec<Bound<'py, PyAny>>> {
        if slf.nbuffers() == 0 {
            return Ok(Vec::new());
        }

        let pyarrow = py.import("pyarrow")?;

        let mut buffers = Vec::with_capacity(slf.nbuffers());
        for buffer in (0..slf.nbuffers()).map(|i| slf.buffer(i)) {
            let buffer: ByteBuffer = buffer.and_then(|b| b.try_to_host_sync())?;

            let addr = buffer.as_ptr() as usize;
            let size = buffer.len();
            let base = &slf;
            let pa_buffer =
                pyarrow.call_method(intern!(py, "foreign_buffer"), (addr, size, base), None)?;
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
