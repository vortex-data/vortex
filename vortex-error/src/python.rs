//! Python bindings for Vortex errors.

use std::backtrace::Backtrace;

use pyo3::PyErr;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;

use crate::VortexError;

impl From<VortexError> for PyErr {
    fn from(value: VortexError) -> Self {
        PyValueError::new_err(value.to_string())
    }
}

impl From<PyErr> for VortexError {
    fn from(value: PyErr) -> Self {
        VortexError::InvalidArgument(value.to_string().into(), Backtrace::capture())
    }
}
