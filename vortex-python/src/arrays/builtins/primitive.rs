// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::PyRef;
use pyo3::pyclass;
use pyo3::pymethods;
use vortex::array::arrays::PrimitiveVTable;

use crate::arrays::native::AsArrayRef;
use crate::arrays::native::EncodingSubclass;
use crate::arrays::native::PyNativeArray;
use crate::dtype::PyPType;

/// Concrete class for arrays with `vortex.primitive` encoding.
#[pyclass(name = "PrimitiveArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyPrimitiveArray;

impl EncodingSubclass for PyPrimitiveArray {
    type VTable = PrimitiveVTable;
}

#[pymethods]
impl PyPrimitiveArray {
    #[getter]
    fn ptype(slf: PyRef<Self>) -> PyPType {
        slf.as_array_ref().ptype().into()
    }
}
