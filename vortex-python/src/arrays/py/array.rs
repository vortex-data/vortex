use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::{Bound, FromPyObject, Py, PyAny, PyResult};
use vortex::EncodingRef;
use vortex::dtype::DType;
use vortex::error::VortexError;
use vortex::stats::ArrayStats;

use crate::arrays::py::PyPythonArray;

/// Wrapper struct encapsulating a Vortex array implemented using a Python object.
///
/// The user-code object is expected to subclass the abstract base class `vx.PyArray` which
/// will ensure the object implements the necessary methods.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PythonArray {
    pub(super) object: Arc<Py<PyAny>>,
    pub(super) encoding: EncodingRef,
    pub(super) len: usize,
    pub(super) dtype: DType,
    pub(super) stats: ArrayStats,
}

impl<'py> FromPyObject<'py> for PythonArray {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let python_array = ob.downcast::<PyPythonArray>()?.get();
        Ok(Self {
            object: Arc::new(ob.clone().unbind()),
            encoding: python_array.encoding.clone(),
            len: python_array.len,
            dtype: python_array.dtype.clone(),
            stats: python_array.stats.clone(),
        })
    }
}

impl<'py> IntoPyObject<'py> for PythonArray {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = VortexError;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.object.bind(py).to_owned())
    }
}
