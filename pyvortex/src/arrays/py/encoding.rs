use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::{Bound, FromPyObject, Py, PyAny, PyResult};
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::serde::ArrayParts;
use vortex::vtable::{ComputeVTable, EncodingVTable, SerdeVTable, StatisticsVTable};
use vortex::{Array, ArrayContext, ArrayRef, EmptyMetadata, Encoding, EncodingId};

use crate::arrays::py::array::PyArrayInstance;

/// Wrapper struct encapsulating a Python encoding.
#[allow(dead_code)]
#[derive(Debug)]
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
    type Array = PyArrayInstance;
    type Metadata = EmptyMetadata;
}

impl EncodingVTable for PyEncodingClass {
    fn id(&self) -> EncodingId {
        self.id.clone()
    }
}

impl SerdeVTable<&dyn Array> for PyEncodingClass {
    fn decode(
        &self,
        _parts: &ArrayParts,
        _ctx: &ArrayContext,
        _dtype: DType,
        _len: usize,
    ) -> VortexResult<ArrayRef> {
        todo!()
    }
}

impl ComputeVTable for PyEncodingClass {}

impl StatisticsVTable<&dyn Array> for PyEncodingClass {}
