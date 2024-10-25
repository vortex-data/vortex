use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use vortex_rs::error::VortexError;

pub struct PyVortexError(VortexError);

impl PyVortexError {
    pub fn new(error: VortexError) -> Self {
        Self(error)
    }

    pub fn map_err(error: VortexError) -> PyErr {
        Self::new(error).into()
    }
}

impl From<VortexError> for PyVortexError {
    fn from(val: VortexError) -> Self {
        PyVortexError::new(val)
    }
}

impl From<PyVortexError> for PyErr {
    fn from(value: PyVortexError) -> Self {
        PyValueError::new_err(value.0.to_string())
    }
}
