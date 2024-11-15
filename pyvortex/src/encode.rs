use arrow::array::{make_array, ArrayData as ArrowArrayData};
use arrow::datatypes::{DataType, Field};
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::FromPyArrow;
use arrow::record_batch::RecordBatchReader;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vortex::array::ChunkedArray;
use vortex::arrow::{FromArrowArray, FromArrowType};
use vortex::dtype::DType;
use vortex::error::{VortexError, VortexResult};
use vortex::{ArrayData, IntoArrayData};

use crate::array::PyArray;

// Private, ergo not documented.
#[pyfunction]
pub fn _encode<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray>> {
    let pa = obj.py().import_bound("pyarrow")?;
    let pa_array = pa.getattr("Array")?;
    let chunked_array = pa.getattr("ChunkedArray")?;
    let table = pa.getattr("Table")?;

    if obj.is_instance(&pa_array)? {
        let arrow_array = ArrowArrayData::from_pyarrow_bound(obj).map(make_array)?;
        let is_nullable = arrow_array.is_nullable();
        let enc_array = ArrayData::from_arrow(arrow_array, is_nullable);
        Bound::new(obj.py(), PyArray::new(enc_array))
    } else if obj.is_instance(&chunked_array)? {
        let chunks: Vec<Bound<PyAny>> = obj.getattr("chunks")?.extract()?;
        let encoded_chunks = chunks
            .iter()
            .map(|a| {
                ArrowArrayData::from_pyarrow_bound(a)
                    .map(make_array)
                    .map(|a| ArrayData::from_arrow(a, false))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let dtype: DType = obj
            .getattr("type")
            .and_then(|v| DataType::from_pyarrow_bound(&v))
            .map(|dt| DType::from_arrow(&Field::new("_", dt, false)))?;
        Bound::new(
            obj.py(),
            PyArray::new(ChunkedArray::try_new(encoded_chunks, dtype)?.into_array()),
        )
    } else if obj.is_instance(&table)? {
        let array_stream = ArrowArrayStreamReader::from_pyarrow_bound(obj)?;
        let dtype = DType::from_arrow(array_stream.schema());
        let chunks = array_stream
            .into_iter()
            .map(|b| b.map_err(VortexError::ArrowError))
            .map(|b| b.and_then(ArrayData::try_from))
            .collect::<VortexResult<Vec<_>>>()?;
        Bound::new(
            obj.py(),
            PyArray::new(ChunkedArray::try_new(chunks, dtype)?.into_array()),
        )
    } else {
        Err(PyValueError::new_err(
            "Cannot convert object to Vortex array",
        ))
    }
}
