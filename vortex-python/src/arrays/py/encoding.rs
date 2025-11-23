// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::{FromPyObject, Py, PyAny};
use vortex::EncodingId;

/// Wrapper struct encapsulating a Python encoding.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct PythonEncoding {
    pub(super) id: EncodingId,
    pub(super) cls: Arc<Py<PyType>>,
}

/// Convert a Python class into a [`PythonEncoding`].
impl<'py> FromPyObject<'_, 'py> for PythonEncoding {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let cls = ob.cast::<PyType>()?;

        let id = EncodingId::new_arc(
            cls.getattr("id")
                .map_err(|_| {
                    PyValueError::new_err(format!(
                        "PyEncoding subclass {cls:?} must have an 'id' attribute"
                    ))
                })?
                .extract::<String>()
                .map_err(|_| PyValueError::new_err("'id' attribute must be a string"))?
                .into(),
        );

        Ok(PythonEncoding {
            id,
            cls: Arc::new(cls.to_owned().unbind()),
        })
    }
}
