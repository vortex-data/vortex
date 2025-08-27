// SPDX-FileCopyrightText: 2016-2025 Copyright The Apache Software Foundation
// SPDX-FileCopyrightText: 2025 Copyright the Vortex contributors
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileComment: Derived from upstream file arrow-pyarrow/src/lib.rs at commit 549709fb at https://github.com/apache/arrow-rs
// SPDX-FileNotice: https://github.com/apache/arrow-rs/blob/549709fbdf91cd1f6c263a7e4540c542b6fecf6b/NOTICE.txt
#![allow(clippy::same_name_method)]

use std::convert::{From, TryFrom};
use std::ptr::addr_of;
use std::sync::Arc;

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{
    RecordBatch, RecordBatchIterator, RecordBatchOptions, RecordBatchReader, StructArray, ffi,
};
use arrow_data::ArrayData;
use arrow_schema::{ArrowError, DataType, Field, Schema};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::ffi::Py_uintptr_t;
use pyo3::import_exception;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};

import_exception!(pyarrow, ArrowException);
/// Represents an exception raised by PyArrow.
pub type PyArrowException = ArrowException;

fn to_py_err(err: ArrowError) -> PyErr {
    PyArrowException::new_err(err.to_string())
}

/// Trait for converting Python objects to arrow-rs types.
pub trait FromPyArrow: Sized {
    /// Convert a Python object to an arrow-rs type.
    ///
    /// Takes a GIL-bound value from Python and returns a result with the arrow-rs type.
    fn from_pyarrow_bound(value: &Bound<PyAny>) -> PyResult<Self>;
}

/// Create a new PyArrow object from a arrow-rs type.
pub trait ToPyArrow {
    /// Convert the implemented type into a Python object without consuming it.
    fn to_pyarrow(&self, py: Python) -> PyResult<PyObject>;
}

/// Convert an arrow-rs type into a PyArrow object.
pub trait IntoPyArrow {
    /// Convert the implemented type into a Python object while consuming it.
    fn into_pyarrow(self, py: Python) -> PyResult<PyObject>;
}

fn validate_pycapsule(capsule: &Bound<PyCapsule>, name: &str) -> PyResult<()> {
    let Some(capsule_name) = capsule.name()?.map(|s| s.to_str()).transpose()? else {
        return Err(PyValueError::new_err(
            "Expected schema PyCapsule to have name set.",
        ));
    };

    if capsule_name != name {
        return Err(PyValueError::new_err(format!(
            "Expected name '{}' in PyCapsule, instead got '{}'",
            name, capsule_name
        )));
    }

    Ok(())
}

impl FromPyArrow for DataType {
    fn from_pyarrow_bound(value: &Bound<PyAny>) -> PyResult<Self> {
        if !value.hasattr("__arrow_c_schema__")? {
            return Err(PyValueError::new_err(
                "Expected __arrow_c_schema__ attribute to be set.",
            ));
        }

        let capsule = value.getattr("__arrow_c_schema__")?.call0()?;
        let capsule = capsule.downcast::<PyCapsule>()?;
        validate_pycapsule(capsule, "arrow_schema")?;

        let schema_ptr = unsafe { capsule.reference::<FFI_ArrowSchema>() };
        let dtype = DataType::try_from(schema_ptr).map_err(to_py_err)?;
        Ok(dtype)
    }
}

impl ToPyArrow for DataType {
    fn to_pyarrow(&self, py: Python) -> PyResult<PyObject> {
        let c_schema = FFI_ArrowSchema::try_from(self).map_err(to_py_err)?;
        let module = py.import("pyarrow")?;
        let class = module.getattr("DataType")?;
        let dtype = class.call_method1("_import_from_c", (&raw const c_schema as Py_uintptr_t,))?;
        Ok(dtype.into())
    }
}

impl FromPyArrow for Field {
    fn from_pyarrow_bound(value: &Bound<PyAny>) -> PyResult<Self> {
        if !value.hasattr("__arrow_c_schema__")? {
            return Err(PyValueError::new_err(
                "Expected __arrow_c_schema__ attribute to be set.",
            ));
        }

        let capsule = value.getattr("__arrow_c_schema__")?.call0()?;
        let capsule = capsule.downcast::<PyCapsule>()?;
        validate_pycapsule(capsule, "arrow_schema")?;

        let schema_ptr = unsafe { capsule.reference::<FFI_ArrowSchema>() };
        let field = Field::try_from(schema_ptr).map_err(to_py_err)?;
        Ok(field)
    }
}

impl ToPyArrow for Field {
    fn to_pyarrow(&self, py: Python) -> PyResult<PyObject> {
        let c_schema = FFI_ArrowSchema::try_from(self).map_err(to_py_err)?;
        let module = py.import("pyarrow")?;
        let class = module.getattr("Field")?;
        let dtype = class.call_method1("_import_from_c", (&raw const c_schema as Py_uintptr_t,))?;
        Ok(dtype.into())
    }
}

impl FromPyArrow for Schema {
    fn from_pyarrow_bound(value: &Bound<PyAny>) -> PyResult<Self> {
        if !value.hasattr("__arrow_c_schema__")? {
            return Err(PyValueError::new_err(
                "Expected __arrow_c_schema__ attribute to be set.",
            ));
        }

        let capsule = value.getattr("__arrow_c_schema__")?.call0()?;
        let capsule = capsule.downcast::<PyCapsule>()?;
        validate_pycapsule(capsule, "arrow_schema")?;

        let schema_ptr = unsafe { capsule.reference::<FFI_ArrowSchema>() };
        let schema = Schema::try_from(schema_ptr).map_err(to_py_err)?;
        Ok(schema)
    }
}

impl ToPyArrow for Schema {
    fn to_pyarrow(&self, py: Python) -> PyResult<PyObject> {
        let c_schema = FFI_ArrowSchema::try_from(self).map_err(to_py_err)?;
        let module = py.import("pyarrow")?;
        let class = module.getattr("Schema")?;
        let schema =
            class.call_method1("_import_from_c", (&raw const c_schema as Py_uintptr_t,))?;
        Ok(schema.into())
    }
}

impl FromPyArrow for ArrayData {
    fn from_pyarrow_bound(value: &Bound<PyAny>) -> PyResult<Self> {
        if !value.hasattr("__arrow_c_array__")? {
            return Err(PyValueError::new_err(
                "Expected __arrow_c_array__ attribute to be set.",
            ));
        }

        let tuple = value.getattr("__arrow_c_array__")?.call0()?;

        if !tuple.is_instance_of::<PyTuple>() {
            return Err(PyTypeError::new_err(
                "Expected __arrow_c_array__ to return a tuple.",
            ));
        }

        let schema_capsule = tuple.get_item(0)?;
        let schema_capsule = schema_capsule.downcast::<PyCapsule>()?;
        let array_capsule = tuple.get_item(1)?;
        let array_capsule = array_capsule.downcast::<PyCapsule>()?;

        validate_pycapsule(schema_capsule, "arrow_schema")?;
        validate_pycapsule(array_capsule, "arrow_array")?;

        let schema_ptr = unsafe { schema_capsule.reference::<FFI_ArrowSchema>() };
        let array = unsafe { FFI_ArrowArray::from_raw(array_capsule.pointer() as _) };
        unsafe { ffi::from_ffi(array, schema_ptr) }.map_err(to_py_err)
    }
}

impl ToPyArrow for ArrayData {
    fn to_pyarrow(&self, py: Python) -> PyResult<PyObject> {
        let array = FFI_ArrowArray::new(self);
        let schema = FFI_ArrowSchema::try_from(self.data_type()).map_err(to_py_err)?;

        let module = py.import("pyarrow")?;
        let class = module.getattr("Array")?;
        let array = class.call_method1(
            "_import_from_c",
            (
                addr_of!(array) as Py_uintptr_t,
                addr_of!(schema) as Py_uintptr_t,
            ),
        )?;
        Ok(array.unbind())
    }
}

impl FromPyArrow for RecordBatch {
    fn from_pyarrow_bound(value: &Bound<PyAny>) -> PyResult<Self> {
        if !value.hasattr("__arrow_c_array__")? {
            return Err(PyValueError::new_err(
                "Expected __arrow_c_array__ attribute to be set.",
            ));
        }

        let tuple = value.getattr("__arrow_c_array__")?.call0()?;

        if !tuple.is_instance_of::<PyTuple>() {
            return Err(PyTypeError::new_err(
                "Expected __arrow_c_array__ to return a tuple.",
            ));
        }

        let schema_capsule = tuple.get_item(0)?;
        let schema_capsule = schema_capsule.downcast::<PyCapsule>()?;
        let array_capsule = tuple.get_item(1)?;
        let array_capsule = array_capsule.downcast::<PyCapsule>()?;

        validate_pycapsule(schema_capsule, "arrow_schema")?;
        validate_pycapsule(array_capsule, "arrow_array")?;

        let schema_ptr = unsafe { schema_capsule.reference::<FFI_ArrowSchema>() };
        let ffi_array = unsafe { FFI_ArrowArray::from_raw(array_capsule.pointer().cast()) };
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
    fn to_pyarrow(&self, py: Python) -> PyResult<PyObject> {
        // Workaround apache/arrow#37669 by returning RecordBatchIterator
        let reader = RecordBatchIterator::new(vec![Ok(self.clone())], self.schema());
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(reader);
        let py_reader = reader.into_pyarrow(py)?;
        py_reader.call_method0(py, "read_next_batch")
    }
}

/// Supports conversion from `pyarrow.RecordBatchReader` to [ArrowArrayStreamReader].
impl FromPyArrow for ArrowArrayStreamReader {
    fn from_pyarrow_bound(value: &Bound<PyAny>) -> PyResult<Self> {
        if !value.hasattr("__arrow_c_stream__")? {
            return Err(PyValueError::new_err(
                "Expected __arrow_c_stream__ attribute to be set.",
            ));
        }

        let capsule = value.getattr("__arrow_c_stream__")?.call0()?;
        let capsule = capsule.downcast::<PyCapsule>()?;
        validate_pycapsule(capsule, "arrow_array_stream")?;

        let stream = unsafe { FFI_ArrowArrayStream::from_raw(capsule.pointer() as _) };

        let stream_reader = ArrowArrayStreamReader::try_new(stream)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        Ok(stream_reader)
    }
}

/// Convert a [`RecordBatchReader`] into a `pyarrow.RecordBatchReader`.
impl IntoPyArrow for Box<dyn RecordBatchReader + Send> {
    // We can't implement `ToPyArrow` for `T: RecordBatchReader + Send` because
    // there is already a blanket implementation for `T: ToPyArrow`.
    fn into_pyarrow(self, py: Python) -> PyResult<PyObject> {
        let mut stream = FFI_ArrowArrayStream::new(self);

        let module = py.import("pyarrow")?;
        let class = module.getattr("RecordBatchReader")?;
        let args = PyTuple::new(py, [&raw mut stream as Py_uintptr_t])?;
        let reader = class.call_method1("_import_from_c", args)?;

        Ok(PyObject::from(reader))
    }
}
