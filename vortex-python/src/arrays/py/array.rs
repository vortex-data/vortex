// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use pyo3::Bound;
use pyo3::FromPyObject;
use pyo3::Py;
use pyo3::PyAny;
use pyo3::prelude::*;
use vortex::array::stats::ArrayStats;
use vortex::array::vtable::ArrayVTable;
use vortex::dtype::DType;
use vortex::error::VortexError;

use crate::arrays::py::PyPythonArray;

/// Wrapper struct encapsulating a Vortex array implemented using a Python object.
///
/// The user-code object is expected to subclass the abstract base class `vx.PyArray` which
/// will ensure the object implements the necessary methods.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PythonArray {
    pub(super) object: Arc<Py<PyAny>>,
    pub(super) vtable: ArrayVTable,
    pub(super) len: usize,
    pub(super) dtype: DType,
    pub(super) stats: ArrayStats,
}

impl<'py> FromPyObject<'_, 'py> for PythonArray {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let ob_cast = ob.cast::<PyPythonArray>()?;
        let python_array = ob_cast.get();
        Ok(Self {
            object: Arc::new(ob.to_owned().unbind()),
            vtable: python_array.vtable.clone(),
            len: python_array.len,
            dtype: python_array.dtype.clone(),
            stats: python_array.stats.clone(),
        })
    }
}

impl<'py> IntoPyObject<'py> for PythonArray {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = VortexError;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.object.bind(py).to_owned())
    }
}
