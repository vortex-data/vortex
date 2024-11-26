//! Views into arrays of individual values.
//!
//! Vortex, like Arrow, avoids copying data. The classes in this package are returned by
//! :meth:`.Array.scalar_at`. They represent shared-memory views into individual values of a Vortex
//! array.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use vortex::buffer::{Buffer, BufferString};
use vortex::dtype::half::f16;
use vortex::dtype::{DType, PType};
use vortex::scalar::{ListScalar, Scalar, StructScalar};

pub fn scalar_into_py(py: Python, x: Scalar, copy_into_python: bool) -> PyResult<PyObject> {
    Ok(match x.dtype() {
        DType::Null => py.None(),
        DType::Bool(_) => x.as_bool().value().into_py(py),
        DType::Primitive(ptype, ..) => {
            let p = x.as_primitive();
            match ptype {
                PType::U8 => p.typed_value::<u8>().into_py(py),
                PType::U16 => p.typed_value::<u16>().into_py(py),
                PType::U32 => p.typed_value::<u32>().into_py(py),
                PType::U64 => p.typed_value::<u64>().into_py(py),
                PType::I8 => p.typed_value::<i8>().into_py(py),
                PType::I16 => p.typed_value::<i16>().into_py(py),
                PType::I32 => p.typed_value::<i32>().into_py(py),
                PType::I64 => p.typed_value::<i64>().into_py(py),
                PType::F16 => p.typed_value::<f16>().map(f16::to_f32).into_py(py),
                PType::F32 => p.typed_value::<f32>().into_py(py),
                PType::F64 => p.typed_value::<f64>().into_py(py),
            }
        }
        DType::Utf8(_) => {
            let x = x.as_utf8().value();
            match x {
                None => py.None(),
                Some(x) => {
                    if copy_into_python {
                        x.as_str().into_py(py)
                    } else {
                        return PyBufferString::new_pyobject(py, x);
                    }
                }
            }
        }
        DType::Binary(_) => {
            let x = x.as_binary().value();
            match x {
                None => py.None(),
                Some(x) => {
                    if copy_into_python {
                        x.as_slice().into_py(py)
                    } else {
                        PyBuffer::new_pyobject(py, x)?
                    }
                }
            }
        }
        DType::Struct(..) => {
            let struct_scalar = x.as_struct();
            if struct_scalar.is_null() {
                py.None()
            } else if copy_into_python {
                to_python_dict(py, struct_scalar, true)?
            } else {
                PyVortexStruct::new_pyobject(py, x)?
            }
        }
        DType::List(..) => {
            let list_scalar = x.as_list();
            if list_scalar.is_null() {
                py.None()
            } else if copy_into_python {
                to_python_list(py, list_scalar, true)?
            } else {
                PyVortexList::new_pyobject(py, x)?
            }
        }
        DType::Extension(_) => {
            todo!()
        }
    })
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
    inner: Scalar,
}

impl PyVortexList {
    pub fn new(inner: Scalar) -> PyVortexList {
        PyVortexList { inner }
    }

    pub fn new_bound(py: Python, inner: Scalar) -> PyResult<Bound<PyVortexList>> {
        Bound::new(py, Self::new(inner))
    }

    pub fn new_pyobject(py: Python, inner: Scalar) -> PyResult<PyObject> {
        let bound = Bound::new(py, Self::new(inner))?;
        Ok(bound.into_py(py))
    }

    pub fn unwrap(&self) -> &Scalar {
        &self.inner
    }
}

#[pymethods]
impl PyVortexList {
    /// Copy the elements of this list from array memory into a :class:`list`.
    #[pyo3(signature = (*, recursive = false))]
    pub fn into_python(self_: PyRef<Self>, recursive: bool) -> PyResult<PyObject> {
        to_python_list(self_.py(), self_.inner.as_list(), recursive)
    }
}

fn to_python_list(py: Python, scalar: ListScalar<'_>, recursive: bool) -> PyResult<PyObject> {
    Ok(scalar
        .elements()
        .map(|x| scalar_into_py(py, x, recursive))
        .collect::<PyResult<Vec<_>>>()?
        .into_py(py))
}

#[pyclass(name = "VortexStruct", module = "vortex", sequence, subclass)]
/// A view of structured data from a Vortex array.
pub struct PyVortexStruct {
    inner: Scalar,
}

impl PyVortexStruct {
    pub fn new(inner: Scalar) -> PyVortexStruct {
        PyVortexStruct { inner }
    }

    pub fn new_bound(py: Python, inner: Scalar) -> PyResult<Bound<PyVortexStruct>> {
        Bound::new(py, Self::new(inner))
    }

    pub fn new_pyobject(py: Python, inner: Scalar) -> PyResult<PyObject> {
        let bound = Bound::new(py, Self::new(inner))?;
        Ok(bound.into_py(py))
    }

    pub fn unwrap(&self) -> &Scalar {
        &self.inner
    }
}

#[pymethods]
impl PyVortexStruct {
    #[pyo3(signature = (*, recursive = false))]
    /// Copy the elements of this list from array memory into a :class:`dict`.
    pub fn into_python(self_: PyRef<Self>, recursive: bool) -> PyResult<PyObject> {
        to_python_dict(self_.py(), self_.inner.as_struct(), recursive)
    }
}

fn to_python_dict(
    py: Python,
    struct_scalar: StructScalar<'_>,
    recursive: bool,
) -> PyResult<PyObject> {
    let Some(fields) = struct_scalar.fields() else {
        return Ok(py.None());
    };

    let dtype = struct_scalar.struct_dtype();

    let dict = PyDict::new_bound(py);
    for (child, name) in fields.iter().zip(dtype.names().iter()) {
        dict.set_item(
            name.to_string(),
            scalar_into_py(py, child.clone(), recursive)?,
        )?
    }
    Ok(dict.into_py(py))
}
