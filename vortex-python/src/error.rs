// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::error::Error;

use pyo3::CastError;
use pyo3::PyErr;
use pyo3::Python;
use pyo3::exceptions::PyIndexError;
use pyo3::exceptions::PyRuntimeError;
use vortex::error::VortexError;

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
            PyVortexError::Vortex(vx) => {
                if let Some(py_err) = vx.source().and_then(|err| err.downcast_ref::<PyErr>()) {
                    return Python::attach(|py| {
                        if py_err.is_instance_of::<PyIndexError>(py) {
                            py_err.clone_ref(py)
                        } else {
                            PyRuntimeError::new_err(vx.to_string())
                        }
                    });
                }
                PyRuntimeError::new_err(vx.to_string())
            }
        }
    }
}
