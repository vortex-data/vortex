use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::{Bound, FromPyObject, Py, PyAny, PyResult};
use vortex::EncodingId;

/// Wrapper struct encapsulating a Python encoding.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct PythonEncoding {
    pub(super) id: EncodingId,
    pub(super) cls: Arc<Py<PyType>>,
}

/// Convert a Python class into a [`PythonEncoding`].
impl<'py> FromPyObject<'py> for PythonEncoding {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let cls = ob.downcast::<PyType>()?;

        let id = EncodingId::new_arc(
            cls.getattr("id")
                .map_err(|_| {
                    PyValueError::new_err(format!(
                        "PyEncoding subclass {} must have an 'id' attribute",
                        ob
                    ))
                })?
                .extract::<String>()
                .map_err(|_| PyValueError::new_err("'id' attribute must be a string"))?
                .into(),
        );

        Ok(PythonEncoding {
            id,
            cls: Arc::new(cls.clone().unbind()),
        })
    }
}
