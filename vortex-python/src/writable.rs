// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bridge a caller-owned Python binary stream to the Vortex file writer.

use std::io;
use std::io::Write;

use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

pub(crate) struct PyWritable {
    write: Py<PyAny>,
    flush: Py<PyAny>,
}

impl PyWritable {
    pub(crate) fn try_new(output: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = output.py();
        let (Some(write), Some(flush)) = (
            output.getattr_opt(intern!(py, "write"))?,
            output.getattr_opt(intern!(py, "flush"))?,
        ) else {
            return Err(not_writable());
        };
        if !write.is_callable() || !flush.is_callable() {
            return Err(not_writable());
        }
        Ok(Self {
            write: write.unbind(),
            flush: flush.unbind(),
        })
    }
}

fn not_writable() -> PyErr {
    PyTypeError::new_err("expected a binary output stream with write() and flush() methods")
}

impl Write for PyWritable {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        Python::attach(|py| {
            let written = self
                .write
                .call1(py, (PyBytes::new(py, buffer),))?
                .extract::<usize>(py)?;
            if written > buffer.len() {
                return Err(PyValueError::new_err(
                    "write() returned more bytes than were passed to it",
                ));
            }
            Ok(written)
        })
        .map_err(io::Error::other)
    }

    fn flush(&mut self) -> io::Result<()> {
        Python::attach(|py| self.flush.call0(py).map(|_| ())).map_err(io::Error::other)
    }
}
