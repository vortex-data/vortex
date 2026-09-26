// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::CastError;
use pyo3::PyErr;
use pyo3::exceptions::PyAssertionError;
use pyo3::exceptions::PyIndexError;
use pyo3::exceptions::PyKeyError;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::exceptions::PyOSError;
use pyo3::exceptions::PyOverflowError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use vortex::error::VortexError;
use vortex::error::VortexErrorKind;

/// Error type to merge [`VortexError`] and [`PyErr`].
pub enum PyVortexError {
    Py(PyErr),
    Vortex(VortexError),
}

/// A [`Result`] alias where the error is [`PyVortexError`].
pub type PyVortexResult<T> = Result<T, PyVortexError>;

impl From<PyErr> for PyVortexError {
    fn from(value: PyErr) -> Self {
        Self::Py(value)
    }
}

impl From<VortexError> for PyVortexError {
    fn from(value: VortexError) -> Self {
        Self::Vortex(value)
    }
}

impl<'py, 'a> From<CastError<'py, 'a>> for PyVortexError {
    fn from(value: CastError<'py, 'a>) -> Self {
        Self::Py(value.into())
    }
}

impl From<PyVortexError> for PyErr {
    fn from(value: PyVortexError) -> Self {
        match value {
            PyVortexError::Py(py) => py,
            PyVortexError::Vortex(vx) => vortex_to_py_err(vx),
        }
    }
}

/// Raises a [`VortexError`] as the Python exception its kind is the analogue of.
fn vortex_to_py_err(error: VortexError) -> PyErr {
    let message = error.to_string();
    match error.kind() {
        VortexErrorKind::OutOfBounds => PyIndexError::new_err(message),
        VortexErrorKind::NotFound => PyKeyError::new_err(message),
        VortexErrorKind::Overflow => PyOverflowError::new_err(message),
        VortexErrorKind::InvalidArgument | VortexErrorKind::Serde => PyValueError::new_err(message),
        VortexErrorKind::NotImplemented => PyNotImplementedError::new_err(message),
        VortexErrorKind::MismatchedTypes => PyTypeError::new_err(message),
        VortexErrorKind::AssertionFailed => PyAssertionError::new_err(message),
        VortexErrorKind::Io => PyOSError::new_err(message),
        // `Other`, and any kind added after this binding was written.
        _ => PyRuntimeError::new_err(message),
    }
}
