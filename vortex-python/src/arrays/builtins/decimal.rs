// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::PyRef;
use pyo3::pyclass;
use pyo3::pymethods;
use vortex::array::arrays::Decimal;
use vortex::array::arrays::decimal::DecimalArrayExt;
use vortex::error::VortexExpect;

use crate::arrays::native::EncodingSubclass;
use crate::arrays::native::PyNativeArray;

/// Concrete class for arrays with `vortex.decimal` encoding.
#[pyclass(name = "DecimalArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyDecimalArray;

impl EncodingSubclass for PyDecimalArray {
    type VTable = Decimal;
}

#[pymethods]
impl PyDecimalArray {
    #[getter]
    fn precision(slf: PyRef<Self>) -> u8 {
        slf.as_super()
            .inner()
            .as_opt::<Decimal>()
            .vortex_expect("PyDecimalArray wraps a Decimal array")
            .precision()
    }

    #[getter]
    fn scale(slf: PyRef<Self>) -> i8 {
        slf.as_super()
            .inner()
            .as_opt::<Decimal>()
            .vortex_expect("PyDecimalArray wraps a Decimal array")
            .scale()
    }
}
