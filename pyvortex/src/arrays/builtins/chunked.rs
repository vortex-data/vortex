use itertools::Itertools;
use pyo3::{pyclass, pymethods, Bound, PyRef, PyResult};
use vortex::array::ChunkedEncoding;

use crate::arrays::{ArraySubclass, AsArrayRef, PyArray};

/// Concrete class for arrays with `vortex.chunked` encoding.
#[pyclass(name = "ChunkedArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyChunkedArray;

impl ArraySubclass for PyChunkedArray {
    type Encoding = ChunkedEncoding;
}

#[pymethods]
impl PyChunkedArray {
    pub fn chunks(self_: PyRef<'_, Self>) -> PyResult<Vec<Bound<'_, PyArray>>> {
        self_
            .as_array_ref()
            .chunks()
            .map(|chunk| PyArray::init(self_.py(), chunk))
            .try_collect()
    }
}
