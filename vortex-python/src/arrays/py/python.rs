// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::conversion::FromPyObject;
use pyo3::prelude::*;
use pyo3::types::PyType;
use vortex::array::stats::ArrayStats;
use vortex::array::vtable::ArrayVTable;
use vortex::array::vtable::ArrayVTableExt;
use vortex::dtype::DType;

use crate::arrays::PyArray;
use crate::arrays::py::PythonVTable;
use crate::dtype::PyDType;

/// Base class for implementing a Vortex encoding in Python.
///
// This class can hold everything _except_ a reference to its own object self. So when we
// downcast and extract a [`crate::arrays::PythonArray`] from this Python object, we just have
// to wrap it up with the object instance.
#[pyclass(name = "PythonArray", module = "vortex", extends=PyArray, sequence, subclass, frozen)]
pub struct PyPythonArray {
    pub(crate) vtable: ArrayVTable,
    pub(crate) len: usize,
    pub(crate) dtype: DType,
    pub(crate) stats: ArrayStats,
}

#[pymethods]
impl PyPythonArray {
    #[new]
    fn new(
        cls: &Bound<'_, PyType>,
        len: usize,
        dtype: PyDType,
    ) -> PyResult<PyClassInitializer<Self>> {
        let vtable = PythonVTable::extract(cls.as_any().as_borrowed())?.into_vtable();
        Ok(PyClassInitializer::from(PyArray).add_subclass(Self {
            vtable,
            len,
            dtype: dtype.into_inner(),
            stats: Default::default(),
        }))
    }
}
