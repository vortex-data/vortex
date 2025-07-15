// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

impl From<PyErr> for VortexError {
    fn from(value: PyErr) -> Self {
        VortexError::InvalidArgument {
            reason: value.to_string().into(),
        }
    }
}
