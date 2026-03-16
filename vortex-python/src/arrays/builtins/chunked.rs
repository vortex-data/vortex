// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::PyRef;
use pyo3::pyclass;
use pyo3::pymethods;
use vortex::array::arrays::Chunked;

use crate::arrays::PyArrayRef;
use crate::arrays::native::AsArrayRef;
use crate::arrays::native::EncodingSubclass;
use crate::arrays::native::PyNativeArray;

/// Concrete class for arrays with `vortex.chunked` encoding.
#[pyclass(name = "ChunkedArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyChunkedArray;

impl EncodingSubclass for PyChunkedArray {
    type VTable = Chunked;
}

#[pymethods]
impl PyChunkedArray {
    pub fn chunks(self_: PyRef<'_, Self>) -> Vec<PyArrayRef> {
        self_
            .as_array_ref()
            .iter_chunks()
            .map(|chunk| PyArrayRef::from(chunk.clone()))
            .collect()
    }
}
