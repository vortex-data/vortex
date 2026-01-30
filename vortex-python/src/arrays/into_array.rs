// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatchReader as _;
use arrow_array::ffi_stream::ArrowArrayStreamReader;
use arrow_array::make_array;
use arrow_data::ArrayData;
use pyo3::Borrowed;
use pyo3::FromPyObject;
use pyo3::PyAny;
use pyo3::PyErr;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyTypeError;
use pyo3::types::PyAnyMethods;
use vortex::array::ArrayRef;
use vortex::array::arrow::FromArrowArray as _;
use vortex::array::iter::ArrayIteratorAdapter;
use vortex::array::iter::ArrayIteratorExt;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType as _;
use vortex::error::VortexError;

use crate::PyVortex;
use crate::arrays::PyArrayRef;
use crate::arrays::native::PyNativeArray;
use crate::arrays::py::PyPythonArray;
use crate::arrow::FromPyArrow;

/// Conversion type for converting Python objects into a [`vortex::Array`].
pub struct PyIntoArray(PyArrayRef);

impl PyIntoArray {
    pub fn inner(&self) -> &ArrayRef {
        self.0.inner()
    }

    #[allow(dead_code)]
    pub fn into_inner(self) -> ArrayRef {
        self.0.into_inner()
    }
}

impl<'py> FromPyObject<'_, 'py> for PyIntoArray {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        if ob.is_instance_of::<PyNativeArray>() || ob.is_instance_of::<PyPythonArray>() {
            return PyArrayRef::extract(ob).map(PyIntoArray);
        }

        let py = ob.py();
        let pa = py.import("pyarrow")?;

        if ob.is_instance(&pa.getattr("Array")?)? {
            let arrow_array_data = ArrayData::from_pyarrow(&ob.as_borrowed())?;
            let array = ArrayRef::from_arrow(make_array(arrow_array_data).as_ref(), false)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            return Ok(PyIntoArray(PyVortex(array)));
        }

        if ob.is_instance(&pa.getattr("Table")?)? {
            let arrow_stream = ArrowArrayStreamReader::from_pyarrow(&ob.as_borrowed())?;
            let dtype = DType::from_arrow(arrow_stream.schema());
            let vortex_iter = arrow_stream.into_iter().map(|batch_result| {
                batch_result
                    .map_err(VortexError::from)
                    .and_then(|b| ArrayRef::from_arrow(b, false))
            });
            let array = ArrayIteratorAdapter::new(dtype, vortex_iter)
                .read_all()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            return Ok(PyIntoArray(PyVortex(array)));
        }

        Err(PyTypeError::new_err(
            "Expected an object that can be converted to a Vortex ArrayRef (vortex.Array, pyarrow.Array, or pyarrow.Table)",
        ))
    }
}
