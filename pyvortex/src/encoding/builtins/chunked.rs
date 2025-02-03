use itertools::Itertools;
use pyo3::{pyclass, pymethods, Bound, PyRef, PyResult};
use vortex::array::ChunkedEncoding;

use crate::arrays::{ArraySubclass, AsArrayRef, PyArray};

/// Concrete class for arrays with `vortex.chunked` encoding.
#[pyclass(name = "ChunkedEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyChunkedEncoding;

impl ArraySubclass for PyChunkedEncoding {
    type Encoding = ChunkedEncoding;
}

#[pymethods]
impl PyChunkedEncoding {
    pub fn chunks(self_: PyRef<'_, Self>) -> PyResult<Vec<Bound<'_, PyArray>>> {
        self_
            .as_array_ref()
            .chunks()
            .map(|chunk| PyArray::init(self_.py(), chunk))
            .try_collect()
    }
}
