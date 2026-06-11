// SPDX-FileCopyrightText: 2016-2025 Copyright The Apache Software Foundation
// SPDX-FileCopyrightText: 2025 Copyright the Vortex contributors
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileComment: Derived from upstream file arrow-pyarrow/src/main at commit 549709fb at https://github.com/apache/arrow-rs
// SPDX-FileNotice: https://github.com/apache/arrow-rs/blob/549709fbdf91cd1f6c263a7e4540c542b6fecf6b/NOTICE.txt
#![expect(clippy::same_name_method)]

use std::convert::From;
use std::convert::TryFrom;
use std::ffi::CStr;
use std::ptr::addr_of;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::RecordBatchIterator;
use arrow_array::RecordBatchOptions;
use arrow_array::RecordBatchReader;
use arrow_array::StructArray;
use arrow_array::ffi;
use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::ffi_stream::ArrowArrayStreamReader;
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_data::ArrayData;
use arrow_schema::ArrowError;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::Py_uintptr_t;
use pyo3::ffi::c_str;
use pyo3::import_exception;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3::types::PyTuple;
use vortex::array::arrow::normalize_array_data;

use crate::classes::array_class;
use crate::classes::data_type_class;
use crate::classes::field_class;
use crate::classes::record_batch_reader_class;
use crate::classes::schema_class;

const SCHEMA_NAME: &CStr = c_str!("arrow_schema");
const ARRAY_NAME: &CStr = c_str!("arrow_array");
const ARRAY_STREAM_NAME: &CStr = c_str!("arrow_array_stream");

import_exception!(pyarrow, ArrowException);
/// Represents an exception raised by PyArrow.
pub type PyArrowException = ArrowException;

fn to_py_err(err: ArrowError) -> PyErr {
    PyArrowException::new_err(err.to_string())
}

/// Trait for converting Python objects to arrow-rs types.
pub trait FromPyArrow<'a, 'py>: Sized {
    /// Convert a Python object to an arrow-rs type.
    ///
    /// Takes a GIL-bound value from Python and returns a result with the arrow-rs type.
    fn from_pyarrow(value: &Borrowed<'a, 'py, PyAny>) -> PyResult<Self>;
}

/// Create a new PyArrow object from a arrow-rs type.
pub trait ToPyArrow {
    /// Convert the implemented type into a Python object without consuming it.
    fn to_pyarrow(&self, py: Python) -> PyResult<Py<PyAny>>;
}

/// Convert an arrow-rs type into a PyArrow object.
pub trait IntoPyArrow {
    /// Convert the implemented type into a Python object while consuming it.
    fn into_pyarrow(self, py: Python) -> PyResult<Py<PyAny>>;
}

impl<'py> FromPyArrow<'_, 'py> for DataType {
    fn from_pyarrow(value: &Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let py = value.py();
        if !value.hasattr(intern!(py, "__arrow_c_schema__"))? {
            return Err(PyValueError::new_err(
                "Expected __arrow_c_schema__ attribute to be set.",
            ));
        }

        let capsule = value.getattr(intern!(py, "__arrow_c_schema__"))?.call0()?;
        let capsule = capsule.cast::<PyCapsule>()?;

        let schema_ptr = unsafe {
            capsule
                .pointer_checked(Some(SCHEMA_NAME))?
                .cast::<FFI_ArrowSchema>()
                .as_ref()
        };

        DataType::try_from(schema_ptr).map_err(to_py_err)
    }
}

impl ToPyArrow for DataType {
    fn to_pyarrow(&self, py: Python) -> PyResult<Py<PyAny>> {
        let c_schema = FFI_ArrowSchema::try_from(self).map_err(to_py_err)?;
        let dtype = data_type_class(py)?.call_method1(
            intern!(py, "_import_from_c"),
            (&raw const c_schema as Py_uintptr_t,),
        )?;
        Ok(dtype.into())
    }
}

impl<'py> FromPyArrow<'_, 'py> for Field {
    fn from_pyarrow(value: &Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let py = value.py();
        if !value.hasattr(intern!(py, "__arrow_c_schema__"))? {
            return Err(PyValueError::new_err(
                "Expected __arrow_c_schema__ attribute to be set.",
            ));
        }

        let capsule = value.getattr(intern!(py, "__arrow_c_schema__"))?.call0()?;
        let capsule = capsule.cast::<PyCapsule>()?;

        let schema_ptr = unsafe {
            capsule
                .pointer_checked(Some(SCHEMA_NAME))?
                .cast::<FFI_ArrowSchema>()
                .as_ref()
        };
        let field = Field::try_from(schema_ptr).map_err(to_py_err)?;
        Ok(field)
    }
}

impl ToPyArrow for Field {
    fn to_pyarrow(&self, py: Python) -> PyResult<Py<PyAny>> {
        let c_schema = FFI_ArrowSchema::try_from(self).map_err(to_py_err)?;
        let dtype = field_class(py)?.call_method1(
            intern!(py, "_import_from_c"),
            (&raw const c_schema as Py_uintptr_t,),
        )?;
        Ok(dtype.into())
    }
}

impl<'py> FromPyArrow<'_, 'py> for Schema {
    fn from_pyarrow(value: &Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let py = value.py();
        if !value.hasattr(intern!(py, "__arrow_c_schema__"))? {
            return Err(PyValueError::new_err(
                "Expected __arrow_c_schema__ attribute to be set.",
            ));
        }

        let capsule = value.getattr(intern!(py, "__arrow_c_schema__"))?.call0()?;
        let capsule = capsule.cast::<PyCapsule>()?;

        let schema_ptr = unsafe {
            capsule
                .pointer_checked(Some(SCHEMA_NAME))?
                .cast::<FFI_ArrowSchema>()
                .as_ref()
        };

        let schema = Schema::try_from(schema_ptr).map_err(to_py_err)?;
        Ok(schema)
    }
}

impl ToPyArrow for Schema {
    fn to_pyarrow(&self, py: Python) -> PyResult<Py<PyAny>> {
        let c_schema = FFI_ArrowSchema::try_from(self).map_err(to_py_err)?;
        let schema = schema_class(py)?.call_method1(
            intern!(py, "_import_from_c"),
            (&raw const c_schema as Py_uintptr_t,),
        )?;
        Ok(schema.into())
    }
}

impl<'py> FromPyArrow<'_, 'py> for ArrayData {
    fn from_pyarrow(value: &Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let py = value.py();
        if !value.hasattr(intern!(py, "__arrow_c_array__"))? {
            return Err(PyValueError::new_err(
                "Expected __arrow_c_array__ attribute to be set.",
            ));
        }

        let tuple = value.getattr(intern!(py, "__arrow_c_array__"))?.call0()?;

        if !tuple.is_instance_of::<PyTuple>() {
            return Err(PyTypeError::new_err(
                "Expected __arrow_c_array__ to return a tuple.",
            ));
        }

        let schema_capsule = tuple.get_item(0)?;
        let schema_capsule = schema_capsule.cast::<PyCapsule>()?;
        let array_capsule = tuple.get_item(1)?;
        let array_capsule = array_capsule.cast::<PyCapsule>()?;

        let schema_ptr = unsafe {
            schema_capsule
                .pointer_checked(Some(SCHEMA_NAME))?
                .cast::<FFI_ArrowSchema>()
                .as_ref()
        };
        let array_ptr = array_capsule
            .pointer_checked(Some(ARRAY_NAME))?
            .cast::<FFI_ArrowArray>()
            .as_ptr();

        let array = unsafe { FFI_ArrowArray::from_raw(array_ptr) };
        let data = unsafe { ffi::from_ffi(array, schema_ptr) }.map_err(to_py_err)?;
        // Rewrite sliced struct/fixed-size-list nodes that `make_array` would misinterpret.
        normalize_array_data(data).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

impl ToPyArrow for ArrayData {
    fn to_pyarrow(&self, py: Python) -> PyResult<Py<PyAny>> {
        let array = FFI_ArrowArray::new(self);
        let schema = FFI_ArrowSchema::try_from(self.data_type()).map_err(to_py_err)?;

        let array = array_class(py)?.call_method1(
            intern!(py, "_import_from_c"),
            (
                addr_of!(array) as Py_uintptr_t,
                addr_of!(schema) as Py_uintptr_t,
            ),
        )?;
        Ok(array.unbind())
    }
}

impl<'py> FromPyArrow<'_, 'py> for RecordBatch {
    fn from_pyarrow(value: &Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let py = value.py();
        if !value.hasattr(intern!(py, "__arrow_c_array__"))? {
            return Err(PyValueError::new_err(
                "Expected __arrow_c_array__ attribute to be set.",
            ));
        }

        let tuple = value.getattr(intern!(py, "__arrow_c_array__"))?.call0()?;

        if !tuple.is_instance_of::<PyTuple>() {
            return Err(PyTypeError::new_err(
                "Expected __arrow_c_array__ to return a tuple.",
            ));
        }

        let schema_capsule = tuple.get_item(0)?;
        let schema_capsule = schema_capsule.cast::<PyCapsule>()?;
        let array_capsule = tuple.get_item(1)?;
        let array_capsule = array_capsule.cast::<PyCapsule>()?;

        let schema_ptr = unsafe {
            schema_capsule
                .pointer_checked(Some(SCHEMA_NAME))?
                .cast::<FFI_ArrowSchema>()
                .as_ref()
        };
        let array_ptr = array_capsule
            .pointer_checked(Some(ARRAY_NAME))?
            .cast::<FFI_ArrowArray>()
            .as_ptr();

        let ffi_array = unsafe { FFI_ArrowArray::from_raw(array_ptr) };
        let mut array_data = unsafe { ffi::from_ffi(ffi_array, schema_ptr) }.map_err(to_py_err)?;
        if !matches!(array_data.data_type(), DataType::Struct(_)) {
            return Err(PyTypeError::new_err(
                "Expected Struct type from __arrow_c_array.",
            ));
        }
        let options = RecordBatchOptions::default().with_row_count(Some(array_data.len()));
        // Ensure data is aligned (by potentially copying the buffers).
        // This is needed because some python code (for example the
        // python flight client) produces unaligned buffers
        // See https://github.com/apache/arrow/issues/43552 for details
        array_data.align_buffers();
        let array = StructArray::from(array_data);
        // StructArray does not embed metadata from schema. We need to override
        // the output schema with the schema from the capsule.
        let schema = Arc::new(Schema::try_from(schema_ptr).map_err(to_py_err)?);
        let (_fields, columns, nulls) = array.into_parts();
        assert_eq!(
            nulls.map(|n| n.null_count()).unwrap_or_default(),
            0,
            "Cannot convert nullable StructArray to RecordBatch, see StructArray documentation"
        );

        RecordBatch::try_new_with_options(schema, columns, &options).map_err(to_py_err)
    }
}

impl ToPyArrow for RecordBatch {
    fn to_pyarrow(&self, py: Python) -> PyResult<Py<PyAny>> {
        // Workaround apache/arrow#37669 by returning RecordBatchIterator
        let reader = RecordBatchIterator::new(vec![Ok(self.clone())], self.schema());
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(reader);
        let py_reader = reader.into_pyarrow(py)?;
        py_reader.call_method0(py, intern!(py, "read_next_batch"))
    }
}

/// Import a `FFI_ArrowArrayStream` from a Python object exposing `__arrow_c_stream__`.
fn ffi_stream_from_pyarrow(value: &Borrowed<'_, '_, PyAny>) -> PyResult<FFI_ArrowArrayStream> {
    let py = value.py();
    if !value.hasattr(intern!(py, "__arrow_c_stream__"))? {
        return Err(PyValueError::new_err(
            "Expected __arrow_c_stream__ attribute to be set.",
        ));
    }

    let capsule = value.getattr(intern!(py, "__arrow_c_stream__"))?.call0()?;
    let capsule = capsule.cast::<PyCapsule>()?;

    let array_ptr = capsule
        .pointer_checked(Some(ARRAY_STREAM_NAME))?
        .cast::<FFI_ArrowArrayStream>()
        .as_ptr();

    Ok(unsafe { FFI_ArrowArrayStream::from_raw(array_ptr) })
}

/// Supports conversion from `pyarrow.RecordBatchReader` to [ArrowArrayStreamReader].
impl<'py> FromPyArrow<'_, 'py> for ArrowArrayStreamReader {
    fn from_pyarrow(value: &Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let stream = ffi_stream_from_pyarrow(value)?;
        ArrowArrayStreamReader::try_new(stream)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

/// A replacement for [`ArrowArrayStreamReader`] that normalizes each imported batch with
/// [`normalize_array_data`] before constructing the record batch.
///
/// arrow-rs (up to at least v59) panics when importing record batches that contain sliced
/// struct or fixed-size-list columns, because `ArrayData::slice` and `StructArray::from`
/// disagree on the meaning of a struct's offset. Prefer this reader for any stream coming
/// from pyarrow.
pub struct NormalizedArrayStreamReader {
    stream: FFI_ArrowArrayStream,
    schema: SchemaRef,
}

impl NormalizedArrayStreamReader {
    fn try_new(mut stream: FFI_ArrowArrayStream) -> Result<Self, ArrowError> {
        if stream.release.is_none() {
            return Err(ArrowError::CDataInterface(
                "input stream is already released".to_string(),
            ));
        }

        let mut ffi_schema = FFI_ArrowSchema::empty();
        let get_schema = stream.get_schema.ok_or_else(|| {
            ArrowError::CDataInterface("input stream has no get_schema function".to_string())
        })?;
        let ret_code = unsafe { get_schema(&raw mut stream, &raw mut ffi_schema) };
        if ret_code != 0 {
            return Err(ArrowError::CDataInterface(format!(
                "Cannot get schema from input stream. Error code: {ret_code:?}"
            )));
        }
        let schema = Arc::new(Schema::try_from(&ffi_schema)?);

        Ok(Self { stream, schema })
    }

    fn get_stream_last_error(&mut self) -> Option<String> {
        let get_last_error = self.stream.get_last_error?;

        let error_str = unsafe { get_last_error(&raw mut self.stream) };
        if error_str.is_null() {
            return None;
        }

        let error_str = unsafe { CStr::from_ptr(error_str) };
        Some(error_str.to_string_lossy().to_string())
    }

    /// The schema of the record batches produced by this reader.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl Iterator for NormalizedArrayStreamReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut array = FFI_ArrowArray::empty();
        let get_next = self.stream.get_next?;
        let ret_code = unsafe { get_next(&raw mut self.stream, &raw mut array) };

        if ret_code != 0 {
            let last_error = self
                .get_stream_last_error()
                .unwrap_or_else(|| format!("error code {ret_code}"));
            return Some(Err(ArrowError::CDataInterface(last_error)));
        }

        // The end of the stream has been reached.
        if array.is_released() {
            return None;
        }

        let result = unsafe {
            ffi::from_ffi_and_data_type(array, DataType::Struct(self.schema.fields().clone()))
        }
        .and_then(|data| {
            normalize_array_data(data).map_err(|e| ArrowError::CDataInterface(e.to_string()))
        })
        .and_then(|data| {
            let len = data.len();
            RecordBatch::try_new_with_options(
                self.schema(),
                StructArray::from(data).into_parts().1,
                &RecordBatchOptions::new().with_row_count(Some(len)),
            )
        });
        Some(result)
    }
}

impl RecordBatchReader for NormalizedArrayStreamReader {
    fn schema(&self) -> SchemaRef {
        NormalizedArrayStreamReader::schema(self)
    }
}

/// Supports conversion from `pyarrow.RecordBatchReader` (or any object exposing
/// `__arrow_c_stream__`) to [`NormalizedArrayStreamReader`].
impl<'py> FromPyArrow<'_, 'py> for NormalizedArrayStreamReader {
    fn from_pyarrow(value: &Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let stream = ffi_stream_from_pyarrow(value)?;
        NormalizedArrayStreamReader::try_new(stream)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

/// Convert a [`RecordBatchReader`] into a `pyarrow.RecordBatchReader`.
impl IntoPyArrow for Box<dyn RecordBatchReader + Send> {
    // We can't implement `ToPyArrow` for `T: RecordBatchReader + Send` because
    // there is already a blanket implementation for `T: ToPyArrow`.
    fn into_pyarrow(self, py: Python) -> PyResult<Py<PyAny>> {
        let mut stream = FFI_ArrowArrayStream::new(self);

        let args = PyTuple::new(py, [&raw mut stream as Py_uintptr_t])?;
        let reader =
            record_batch_reader_class(py)?.call_method1(intern!(py, "_import_from_c"), args)?;

        Ok(Py::from(reader))
    }
}
