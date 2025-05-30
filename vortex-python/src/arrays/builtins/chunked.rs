use pyo3::{PyRef, pyclass, pymethods};
use vortex::arrays::ChunkedVTable;

use crate::arrays::PyArrayRef;
use crate::arrays::native::{AsArrayRef, EncodingSubclass, PyNativeArray};

/// Concrete class for arrays with `vortex.chunked` encoding.
#[pyclass(name = "ChunkedArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyChunkedArray;

impl EncodingSubclass for PyChunkedArray {
    type VTable = ChunkedVTable;
}

#[pymethods]
impl PyChunkedArray {
    pub fn chunks(self_: PyRef<'_, Self>) -> Vec<PyArrayRef> {
        self_
            .as_array_ref()
            .chunks()
            .iter()
            .map(|chunk| PyArrayRef::from(chunk.clone()))
            .collect()
    }
}
