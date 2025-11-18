use crate::PyVortex;
use crate::arrays::py::PyPythonArray;
use crate::arrays::{PyArrayRef, native::PyNativeArray};
use crate::arrow::FromPyArrow;
use arrow_array::ffi_stream::ArrowArrayStreamReader;
use arrow_array::{RecordBatchReader as _, make_array};
use arrow_data::ArrayData;
use pyo3::{Bound, FromPyObject, PyAny, PyResult, exceptions::PyTypeError, types::PyAnyMethods};
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType as _;
use vortex::error::VortexResult;
use vortex::iter::{ArrayIteratorAdapter, ArrayIteratorExt};
use vortex::{ArrayRef, arrow::FromArrowArray as _};

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

impl<'py> FromPyObject<'py> for PyIntoArray {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if ob.is_instance_of::<PyNativeArray>() || ob.is_instance_of::<PyPythonArray>() {
            return PyArrayRef::extract_bound(ob).map(PyIntoArray);
        }

        let py = ob.py();
        let pa = py.import("pyarrow")?;

        if ob.is_instance(&pa.getattr("Array")?)? {
            let arrow_array_data = ArrayData::from_pyarrow_bound(ob)?;
            return Ok(PyIntoArray(PyVortex(ArrayRef::from_arrow(
                make_array(arrow_array_data).as_ref(),
                false,
            ))));
        }

        if ob.is_instance(&pa.getattr("Table")?)? {
            let arrow_stream = ArrowArrayStreamReader::from_pyarrow_bound(ob)?;
            let dtype = DType::from_arrow(arrow_stream.schema());
            let vortex_iter = arrow_stream
                .into_iter()
                .map(|batch_result| -> VortexResult<_> {
                    Ok(ArrayRef::from_arrow(batch_result?, false))
                });
            let array = ArrayIteratorAdapter::new(dtype, vortex_iter).read_all()?;
            return Ok(PyIntoArray(PyVortex(array)));
        }

        Err(PyTypeError::new_err(
            "Expected an object that can be converted to a Vortex ArrayRef (vortex.Array, pyarrow.Array, or pyarrow.Table)",
        ))
    }
}
