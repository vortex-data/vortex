#![allow(unused_variables)]
#![allow(dead_code)]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::{Bound, FromPyObject, Py, PyAny, PyResult};
use vortex::vtable::{ComputeVTable, EncodingVTable, SerdeVTable, StatisticsVTable};
use vortex::{Array, EmptyMetadata, Encoding, EncodingId};

use crate::encoding::py::array::PyEncodingInstance;

/// Wrapper struct encapsulating a Python encoding.
pub struct PyEncodingClass {
    id: EncodingId,
    cls: Py<PyType>,
}

/// Convert a Python class into a [`PyEncodingClass`].
impl<'py> FromPyObject<'py> for PyEncodingClass {
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

        Ok(PyEncodingClass {
            id,
            cls: cls.clone().unbind(),
        })
    }
}

impl Encoding for PyEncodingClass {
    type Array = PyEncodingInstance;
    type Metadata = EmptyMetadata;
}

impl EncodingVTable for PyEncodingClass {
    fn id(&self) -> EncodingId {
        self.id.clone()
    }
}

impl ComputeVTable for PyEncodingClass {}

impl SerdeVTable<&dyn Array> for PyEncodingClass {}

impl StatisticsVTable<&dyn Array> for PyEncodingClass {}
