// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use pyo3::Bound;
use pyo3::FromPyObject;
use pyo3::Py;
use pyo3::PyAny;
use pyo3::prelude::*;
use vortex::array::Array;
use vortex::array::ArrayParts;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::stats::ArrayStats;
use vortex::array::stats::StatsSet;
use vortex::dtype::DType;

use crate::arrays::py::PyPythonArray;
use crate::arrays::py::PythonVTable;
use crate::error::PyVortexError;

/// Wrapper struct encapsulating a Vortex array implemented using a Python object.
///
/// The user-code object is expected to subclass the abstract base class `vx.PyArray` which
/// will ensure the object implements the necessary methods.
#[derive(Debug, Clone)]
pub struct PythonArray {
    pub(super) vtable: PythonVTable,
    pub(super) object: Arc<Py<PyAny>>,
    pub(super) len: usize,
    pub(super) dtype: DType,
    pub(super) stats: ArrayStats,
}

impl Display for PythonArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "len: {}, dtype: {}", self.len, self.dtype)
    }
}

impl<'py> FromPyObject<'_, 'py> for PythonArray {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let ob_cast = ob.cast::<PyPythonArray>()?;
        let python_array = ob_cast.get();
        Ok(Self {
            vtable: PythonVTable {
                id: python_array.id.clone(),
            },
            object: Arc::new(ob.to_owned().unbind()),
            len: python_array.len,
            dtype: python_array.dtype.clone(),
            stats: python_array.stats.clone(),
        })
    }
}

impl<'py> IntoPyObject<'py> for PythonArray {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyVortexError;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.object.bind(py).to_owned())
    }
}

impl IntoArray for PythonArray {
    fn into_array(self) -> ArrayRef {
        let vtable = self.vtable.clone();
        let dtype = self.dtype.clone();
        let len = self.len;
        let stats = StatsSet::from(self.stats.clone());
        match Array::try_from_parts(ArrayParts::new(vtable, dtype, len, self)) {
            Ok(array) => array.with_stats_set(stats).into_array(),
            Err(err) => unreachable!(
                "PythonArray metadata extracted from PyPythonArray must be valid: {err}"
            ),
        }
    }
}
