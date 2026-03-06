// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::Bound;
use pyo3::PyRef;
use pyo3::PyResult;
use pyo3::pyclass;
use pyo3::pymethods;
use vortex::array::arrays::ConstantVTable;

use crate::arrays::native::AsArrayRef;
use crate::arrays::native::EncodingSubclass;
use crate::arrays::native::PyNativeArray;
use crate::scalar::PyScalar;

/// Concrete class for arrays with `vortex.constant` encoding.
#[pyclass(name = "ConstantArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyConstantArray;

impl EncodingSubclass for PyConstantArray {
    type VTable = ConstantVTable;
}

#[pymethods]
impl PyConstantArray {
    /// Return the scalar value of the constant array.
    pub fn scalar(self_: PyRef<'_, Self>) -> PyResult<Bound<'_, PyScalar>> {
        PyScalar::init(self_.py(), self_.as_array_ref().scalar().clone())
    }
}
