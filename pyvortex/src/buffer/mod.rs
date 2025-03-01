use std::ffi::{c_int, c_void};
use std::ops::Deref;

use pyo3::exceptions::PyBufferError;
use pyo3::{PyRef, PyResult, ffi, pyclass, pymethods};
use vortex::buffer::ByteBuffer;
use vortex::error::VortexExpect;

#[pyclass(name = "ByteBuffer", module = "vortex", frozen)]
pub struct PyByteBuffer(ByteBuffer);

impl From<ByteBuffer> for PyByteBuffer {
    fn from(buffer: ByteBuffer) -> Self {
        Self(buffer)
    }
}

impl Deref for PyByteBuffer {
    type Target = ByteBuffer;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PyByteBuffer {
    /// Buffer protocol
    unsafe fn __getbuffer__(
        slf: PyRef<Self>,
        buffer: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let buffer = buffer
            .as_mut()
            .vortex_expect("Python passed us a null buffer pointer");

        // If a writable buffer is requested, we error.
        if flags & ffi::PyBUF_WRITABLE > 0 {
            return Err(PyBufferError::new_err("ByteBuffer is read-only"));
        }

        // Basic information about the buffer
        buffer.obj = slf.as_ptr() as *mut ffi::PyObject;
        buffer.len = slf
            .0
            .len()
            .try_into()
            .vortex_expect("Buffer length is too large");
        buffer.buf = slf.0.as_ptr() as *mut c_void;
        buffer.readonly = 1;
        buffer.itemsize = 1; // 1 byte per item
        buffer.format = "B\0".as_ptr() as *mut _; // unsigned char

        // Shape and strides
        buffer.ndim = 1;

        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _buffer: *mut ffi::Py_buffer) -> () {
        // Nothing to do, since the lifetime of ByteBuffer is managed by the PyByteBuffer object.
    }

    /// The number of bytes in the buffer.
    fn __len__(&self) -> usize {
        self.0.len()
    }

    /// The alignment of the buffer.
    fn alignment(&self) -> usize {
        *self.0.alignment()
    }
}
