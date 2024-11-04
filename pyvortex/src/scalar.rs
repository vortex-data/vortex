//! Views into arrays of individual values.
//!
//! Vortex, like Arrow, avoids copying data. The classes in this package are returned by
//! :meth:`.Array.scalar_at`. They represent shared-memory views into individual values of a Vortex
//! array.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use vortex::buffer::{Buffer, BufferString};
use vortex::dtype::{DType, StructDType};
use vortex::error::vortex_panic;
use vortex::scalar::{PValue, Scalar, ScalarValue};

pub fn scalar_into_py(py: Python, x: Scalar, copy_into_python: bool) -> PyResult<PyObject> {
    let (dtype, value) = x.into_parts();
    scalar_value_into_py(py, value, &dtype, copy_into_python)
}

pub fn scalar_value_into_py(
    py: Python,
    x: ScalarValue,
    dtype: &DType,
    copy_into_python: bool,
) -> PyResult<PyObject> {
    match x {
        ScalarValue::Bool(x) => Ok(x.into_py(py)),
        ScalarValue::Primitive(PValue::U8(x)) => Ok(x.into_py(py)),
        ScalarValue::Primitive(PValue::U16(x)) => Ok(x.into_py(py)),
        ScalarValue::Primitive(PValue::U32(x)) => Ok(x.into_py(py)),
        ScalarValue::Primitive(PValue::U64(x)) => Ok(x.into_py(py)),
        ScalarValue::Primitive(PValue::I8(x)) => Ok(x.into_py(py)),
        ScalarValue::Primitive(PValue::I16(x)) => Ok(x.into_py(py)),
        ScalarValue::Primitive(PValue::I32(x)) => Ok(x.into_py(py)),
        ScalarValue::Primitive(PValue::I64(x)) => Ok(x.into_py(py)),
        ScalarValue::Primitive(PValue::F16(x)) => Ok(x.to_f32().into_py(py)),
        ScalarValue::Primitive(PValue::F32(x)) => Ok(x.into_py(py)),
        ScalarValue::Primitive(PValue::F64(x)) => Ok(x.into_py(py)),
        ScalarValue::Buffer(x) => {
            if copy_into_python {
                Ok(x.into_py(py))
            } else {
                PyBuffer::new_pyobject(py, x)
            }
        }
        ScalarValue::BufferString(x) => {
            if copy_into_python {
                Ok(x.into_py(py))
            } else {
                PyBufferString::new_pyobject(py, x)
            }
        }
        ScalarValue::List(x) => match dtype {
            DType::List(dtype, ..) => {
                if copy_into_python {
                    to_python_list(py, &x, dtype, true)
                } else {
                    PyVortexList::new_pyobject(py, x, dtype.clone())
                }
            }
            DType::Struct(dtype, ..) => {
                if copy_into_python {
                    to_python_dict(py, &x, dtype, true)
                } else {
                    PyVortexStruct::new_pyobject(py, x, dtype.clone())
                }
            }
            _ => vortex_panic!("impossible"),
        },
        ScalarValue::Null => Ok(py.None()),
    }
}

#[pyclass(name = "Buffer", module = "vortex", sequence, subclass)]
/// A view of binary data from a Vortex array.
pub struct PyBuffer {
    inner: Buffer,
}

impl PyBuffer {
    pub fn new(inner: Buffer) -> PyBuffer {
        PyBuffer { inner }
    }

    pub fn new_bound(py: Python, inner: Buffer) -> PyResult<Bound<PyBuffer>> {
        Bound::new(py, Self::new(inner))
    }

    pub fn new_pyobject(py: Python, inner: Buffer) -> PyResult<PyObject> {
        let bound = Bound::new(py, Self::new(inner))?;
        Ok(bound.into_py(py))
    }

    pub fn unwrap(&self) -> &Buffer {
        &self.inner
    }
}

#[pymethods]
impl PyBuffer {
    /// Copy this buffer from array memory into a Python bytes.
    #[pyo3(signature = (*, recursive = false))]
    #[allow(unused_variables)] // we want the same Python name across all methods
    pub fn into_python(self_: PyRef<Self>, recursive: bool) -> PyResult<PyObject> {
        Ok(self_.inner.into_py(self_.py()))
    }
}

#[pyclass(name = "BufferString", module = "vortex", sequence, subclass)]
/// A view of UTF-8 data from a Vortex array.
pub struct PyBufferString {
    inner: BufferString,
}

impl PyBufferString {
    pub fn new(inner: BufferString) -> PyBufferString {
        PyBufferString { inner }
    }

    pub fn new_bound(py: Python, inner: BufferString) -> PyResult<Bound<PyBufferString>> {
        Bound::new(py, Self::new(inner))
    }

    pub fn new_pyobject(py: Python, inner: BufferString) -> PyResult<PyObject> {
        let bound = Bound::new(py, Self::new(inner))?;
        Ok(bound.into_py(py))
    }

    pub fn unwrap(&self) -> &BufferString {
        &self.inner
    }
}

#[pymethods]
impl PyBufferString {
    /// Copy this buffer string from array memory into a :class:`str`.
    #[pyo3(signature = (*, recursive = false))]
    #[allow(unused_variables)] // we want the same Python name across all methods
    pub fn into_python(self_: PyRef<Self>, recursive: bool) -> PyResult<PyObject> {
        Ok(self_.inner.into_py(self_.py()))
    }
}

#[pyclass(name = "VortexList", module = "vortex", sequence, subclass)]
/// A view of a variable-length list of data from a Vortex array.
pub struct PyVortexList {
    inner: Arc<[ScalarValue]>,
    dtype: Arc<DType>,
}

impl PyVortexList {
    pub fn new(inner: Arc<[ScalarValue]>, dtype: Arc<DType>) -> PyVortexList {
        PyVortexList { inner, dtype }
    }

    pub fn new_bound(
        py: Python,
        inner: Arc<[ScalarValue]>,
        dtype: Arc<DType>,
    ) -> PyResult<Bound<PyVortexList>> {
        Bound::new(py, Self::new(inner, dtype))
    }

    pub fn new_pyobject(
        py: Python,
        inner: Arc<[ScalarValue]>,
        dtype: Arc<DType>,
    ) -> PyResult<PyObject> {
        let bound = Bound::new(py, Self::new(inner, dtype))?;
        Ok(bound.into_py(py))
    }

    pub fn unwrap(&self) -> &Arc<[ScalarValue]> {
        &self.inner
    }
}

#[pymethods]
impl PyVortexList {
    /// Copy the elements of this list from array memory into a :class:`list`.
    #[pyo3(signature = (*, recursive = false))]
    pub fn into_python(self_: PyRef<Self>, recursive: bool) -> PyResult<PyObject> {
        to_python_list(self_.py(), &self_.inner, &self_.dtype, recursive)
    }
}

fn to_python_list(
    py: Python,
    values: &[ScalarValue],
    dtype: &DType,
    recursive: bool,
) -> PyResult<PyObject> {
    Ok(values
        .iter()
        .cloned()
        .map(|x| scalar_value_into_py(py, x, dtype, recursive))
        .collect::<Result<Vec<_>, _>>()?
        .into_py(py))
}

#[pyclass(name = "VortexStruct", module = "vortex", sequence, subclass)]
/// A view of structured data from a Vortex array.
pub struct PyVortexStruct {
    inner: Arc<[ScalarValue]>,
    dtype: StructDType,
}

impl PyVortexStruct {
    pub fn new(inner: Arc<[ScalarValue]>, dtype: StructDType) -> PyVortexStruct {
        PyVortexStruct { inner, dtype }
    }

    pub fn new_bound(
        py: Python,
        inner: Arc<[ScalarValue]>,
        dtype: StructDType,
    ) -> PyResult<Bound<PyVortexStruct>> {
        Bound::new(py, Self::new(inner, dtype))
    }

    pub fn new_pyobject(
        py: Python,
        inner: Arc<[ScalarValue]>,
        dtype: StructDType,
    ) -> PyResult<PyObject> {
        let bound = Bound::new(py, Self::new(inner, dtype))?;
        Ok(bound.into_py(py))
    }

    pub fn unwrap(&self) -> &Arc<[ScalarValue]> {
        &self.inner
    }
}

#[pymethods]
impl PyVortexStruct {
    #[pyo3(signature = (*, recursive = false))]
    /// Copy the elements of this list from array memory into a :class:`dict`.
    pub fn into_python(self_: PyRef<Self>, recursive: bool) -> PyResult<PyObject> {
        to_python_dict(self_.py(), &self_.inner, &self_.dtype, recursive)
    }
}

fn to_python_dict(
    py: Python,
    values: &[ScalarValue],
    dtype: &StructDType,
    recursive: bool,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    for ((child, name), dtype) in values
        .iter()
        .cloned()
        .zip(dtype.names().iter())
        .zip(dtype.dtypes().iter())
    {
        dict.set_item(
            name.to_string(),
            scalar_value_into_py(py, child, dtype, recursive)?,
        )?
    }
    Ok(dict.into_py(py))
}
