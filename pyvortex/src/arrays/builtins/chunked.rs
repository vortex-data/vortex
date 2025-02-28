use itertools::Itertools;
use pyo3::{Bound, PyRef, PyResult, pyclass, pymethods};
use vortex::arrays::ChunkedEncoding;

use crate::arrays::{AsArrayRef, EncodingSubclass, PyArray};

/// Concrete class for arrays with `vortex.chunked` encoding.
#[pyclass(name = "ChunkedArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyChunkedArray;

impl EncodingSubclass for PyChunkedArray {
    type Encoding = ChunkedEncoding;
}

#[pymethods]
impl PyChunkedArray {
    #[new]
    fn new(array: Bound<PyArray>) -> PyResult<Bound<Self>> {
        PyArray::init_encoding(array, &ChunkedEncoding, PyChunkedArray)
    }

    pub fn chunks(self_: PyRef<'_, Self>) -> PyResult<Vec<Bound<'_, PyArray>>> {
        self_
            .as_array_ref()
            .chunks()
            .iter()
            .map(|chunk| PyArray::init(self_.py(), chunk.clone()))
            .try_collect()
    }
}
