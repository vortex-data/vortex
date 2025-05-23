use pyo3::conversion::FromPyObjectBound;
use pyo3::prelude::*;
use pyo3::types::PyType;
use vortex::EncodingRef;
use vortex::dtype::DType;
use vortex::stats::ArrayStats;

use crate::arrays::PyArray;
use crate::arrays::py::PythonEncoding;
use crate::dtype::PyDType;

/// Base class for implementing a Vortex encoding in Python.
///
// This class can hold everything _except_ a reference to its own object self. So when we
// downcast and extract a [`crate::arrays::PythonArray`] from this Python object, we just have
// to wrap it up with the object instance.
#[pyclass(name = "PythonArray", module = "vortex", extends=PyArray, sequence, subclass, frozen)]
pub struct PyPythonArray {
    pub(crate) encoding: EncodingRef,
    pub(crate) len: usize,
    pub(crate) dtype: DType,
    pub(crate) stats: ArrayStats,
}

#[pymethods]
impl PyPythonArray {
    #[new]
    fn new(
        cls: &Bound<'_, PyType>,
        len: usize,
        dtype: PyDType,
    ) -> PyResult<PyClassInitializer<Self>> {
        let encoding =
            PythonEncoding::from_py_object_bound(cls.as_any().as_borrowed())?.to_encoding();
        Ok(PyClassInitializer::from(PyArray).add_subclass(Self {
            encoding,
            len,
            dtype: dtype.into_inner(),
            stats: Default::default(),
        }))
    }
}
