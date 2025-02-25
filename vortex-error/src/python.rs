//! Python bindings for Vortex errors.

use pyo3::PyErr;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;

use crate::VortexError;

impl From<VortexError> for PyErr {
    fn from(value: VortexError) -> Self {
        PyValueError::new_err(value.to_string())
    }
}
