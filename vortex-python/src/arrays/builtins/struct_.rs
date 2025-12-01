// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use pyo3::PyRef;
use pyo3::PyResult;
use pyo3::pyclass;
use pyo3::pymethods;
use vortex::array::arrays::StructVTable;

use crate::arrays::PyArrayRef;
use crate::arrays::native::AsArrayRef;
use crate::arrays::native::EncodingSubclass;
use crate::arrays::native::PyNativeArray;

/// Concrete class for arrays with `vortex.struct` encoding.
#[pyclass(name = "StructArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyStructArray;

impl EncodingSubclass for PyStructArray {
    type VTable = StructVTable;
}

#[pymethods]
impl PyStructArray {
    /// Returns the given field of the struct array.
    pub fn field(self_: PyRef<'_, Self>, name: &str) -> PyResult<PyArrayRef> {
        let field = self_.as_array_ref().field_by_name(name)?.clone();
        Ok(PyArrayRef::from(field))
    }

    /// Get an ordered list of field names for the struct fields.
    pub fn names(self_: PyRef<'_, Self>) -> PyResult<Vec<String>> {
        Ok(self_
            .as_array_ref()
            .struct_fields()
            .names()
            .iter()
            .map(|f| f.to_string())
            .collect_vec())
    }
}
