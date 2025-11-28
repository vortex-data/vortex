// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::PyRef;
use pyo3::pyclass;
use pyo3::pymethods;
use vortex::array::arrays::DecimalVTable;

use crate::arrays::native::AsArrayRef;
use crate::arrays::native::EncodingSubclass;
use crate::arrays::native::PyNativeArray;

/// Concrete class for arrays with `vortex.decimal` encoding.
#[pyclass(name = "DecimalArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyDecimalArray;

impl EncodingSubclass for PyDecimalArray {
    type VTable = DecimalVTable;
}

#[pymethods]
impl PyDecimalArray {
    #[getter]
    fn precision(slf: PyRef<Self>) -> u8 {
        slf.as_array_ref().precision()
    }

    #[getter]
    fn scale(slf: PyRef<Self>) -> i8 {
        slf.as_array_ref().scale()
    }
}
