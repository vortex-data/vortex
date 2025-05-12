use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::{Bound, FromPyObject, Py, PyAny, PyResult};
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::serde::ArrayParts;
use vortex::vtable::SerdeVTable;
use vortex::{ArrayContext, DeserializeMetadata, EmptyMetadata, EncodingId};

use crate::arrays::py::PythonVTable;
use crate::arrays::py::array::PythonArray;

/// Wrapper struct encapsulating a Python encoding.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct PythonEncoding {
    id: EncodingId,
    cls: Arc<Py<PyType>>,
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

impl SerdeVTable<PythonVTable> for PythonVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &PythonArray) -> Option<Self::Metadata> {
        todo!()
    }

    fn decode(
        encoding: &PythonEncoding,
        _dtype: DType,
        _len: usize,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        _children: &[ArrayParts],
        _ctx: &ArrayContext,
    ) -> VortexResult<PythonArray> {
        Python::with_gil(|py| {
            let _cls = encoding.cls.bind(py);
            todo!()
        })
    }
}
