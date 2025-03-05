use arrow::array::{ArrayData as ArrowArrayData, make_array};
use arrow::datatypes::{DataType, Field};
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::FromPyArrow;
use arrow::record_batch::RecordBatchReader;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vortex::arrays::ChunkedArray;
use vortex::arrow::FromArrowArray;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::{VortexError, VortexResult};
use vortex::{Array, ArrayRef, TryIntoArray};

use crate::arrays::PyArrayRef;

/// Convert an Arrow object to a Vortex array.
pub(super) fn from_arrow(obj: &Bound<'_, PyAny>) -> PyResult<PyArrayRef> {
    let pa = obj.py().import("pyarrow")?;
    let pa_array = pa.getattr("Array")?;
    let chunked_array = pa.getattr("ChunkedArray")?;
    let table = pa.getattr("Table")?;

    if obj.is_instance(&pa_array)? {
        let arrow_array = ArrowArrayData::from_pyarrow_bound(obj).map(make_array)?;
        let is_nullable = arrow_array.is_nullable();
        let enc_array = ArrayRef::from_arrow(arrow_array, is_nullable);
        Ok(PyArrayRef::from(enc_array))
    } else if obj.is_instance(&chunked_array)? {
        let chunks: Vec<Bound<PyAny>> = obj.getattr("chunks")?.extract()?;
        let encoded_chunks = chunks
            .iter()
            .map(|a| {
                ArrowArrayData::from_pyarrow_bound(a)
                    .map(make_array)
                    .map(|a| ArrayRef::from_arrow(a, false))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let dtype: DType = obj
            .getattr("type")
            .and_then(|v| DataType::from_pyarrow_bound(&v))
            .map(|dt| DType::from_arrow(&Field::new("_", dt, false)))?;
        Ok(PyArrayRef::from(
            ChunkedArray::try_new(encoded_chunks, dtype)?.into_array(),
        ))
    } else if obj.is_instance(&table)? {
        let array_stream = ArrowArrayStreamReader::from_pyarrow_bound(obj)?;
        let dtype = DType::from_arrow(array_stream.schema());
        let chunks = array_stream
            .into_iter()
            .map(|b| b.map_err(VortexError::ArrowError))
            .map(|b| b.and_then(|b| b.try_into_array()))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(PyArrayRef::from(
            ChunkedArray::try_new(chunks, dtype)?.into_array(),
        ))
    } else {
        Err(PyValueError::new_err(
            "Cannot convert object to Vortex array",
        ))
    }
}
