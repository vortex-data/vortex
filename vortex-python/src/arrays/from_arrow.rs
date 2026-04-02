// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatchReader;
use arrow_array::ffi_stream::ArrowArrayStreamReader;
use arrow_array::make_array;
use arrow_data::ArrayData as ArrowArrayData;
use arrow_schema::DataType;
use arrow_schema::Field;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrow::FromArrowArray;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::VortexError;
use vortex::error::VortexResult;

use crate::arrays::PyArrayRef;
use crate::arrow::FromPyArrow;
use crate::classes::array_class;
use crate::classes::chunked_array_class;
use crate::classes::table_class;
use crate::error::PyVortexError;
use crate::error::PyVortexResult;

/// Convert an Arrow object to a Vortex array.
pub(super) fn from_arrow(obj: &Borrowed<'_, '_, PyAny>) -> PyVortexResult<PyArrayRef> {
    let py = obj.py();
    let pa_array = array_class(py)?;
    let chunked_array = chunked_array_class(py)?;
    let table = table_class(py)?;

    if obj.is_instance(pa_array)? {
        let arrow_array = ArrowArrayData::from_pyarrow(&obj.as_borrowed()).map(make_array)?;
        let is_nullable = arrow_array.is_nullable();
        let enc_array = ArrayRef::from_arrow(arrow_array.as_ref(), is_nullable)?;
        Ok(PyArrayRef::from(enc_array))
    } else if obj.is_instance(chunked_array)? {
        let chunks: Vec<Bound<PyAny>> = obj.getattr(intern!(py, "chunks"))?.extract()?;
        let encoded_chunks = chunks
            .iter()
            .map(|a| {
                let arrow_array = ArrowArrayData::from_pyarrow(&a.as_borrowed()).map(make_array)?;
                ArrayRef::from_arrow(arrow_array.as_ref(), false).map_err(PyVortexError::from)
            })
            .collect::<PyVortexResult<Vec<_>>>()?;
        let dtype: DType = obj
            .getattr(intern!(py, "type"))
            .and_then(|v| DataType::from_pyarrow(&v.as_borrowed()))
            .map(|dt| DType::from_arrow(&Field::new("_", dt, false)))?;
        Ok(PyArrayRef::from(
            ChunkedArray::try_new(encoded_chunks, dtype)?.into_array(),
        ))
    } else if obj.is_instance(table)? {
        let array_stream = ArrowArrayStreamReader::from_pyarrow(&obj.as_borrowed())?;
        let dtype = DType::from_arrow(array_stream.schema());
        let chunks = array_stream
            .into_iter()
            .map(|b| {
                b.map_err(VortexError::from)
                    .and_then(|b| ArrayRef::from_arrow(b, false))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(PyArrayRef::from(
            ChunkedArray::try_new(chunks, dtype)?.into_array(),
        ))
    } else {
        Err(PyValueError::new_err("Cannot convert object to Vortex array").into())
    }
}
