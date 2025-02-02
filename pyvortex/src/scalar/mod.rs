//! Views into arrays of individual values.
//!
//! Vortex, like Arrow, avoids copying data. The classes in this package are returned by
//! :meth:`.Array.scalar_at`. They represent shared-memory views into individual values of a Vortex
//! array.

mod bool;
pub mod factory;
mod primitive;

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::PyClass;
use vortex::buffer::{BufferString, ByteBuffer};
use vortex::dtype::half::f16;
use vortex::dtype::{DType, PType};
use vortex::error::{VortexError, VortexExpect};
use vortex::scalar::{ListScalar, Scalar, StructScalar};

use crate::install_module;
use crate::scalar::bool::PyBoolScalar;
use crate::scalar::primitive::PyPrimitiveScalar;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(py, "scalar")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.scalar", &m)?;

    m.add_function(wrap_pyfunction!(factory::scalar, &m)?)?;

    m.add_class::<PyScalar>()?;
    m.add_class::<PyBoolScalar>()?;

    // TODO(ngates): rename these based on DType, e.g. Utf8Scalar, BinaryScalar
    m.add_class::<PyBuffer>()?;
    m.add_class::<PyBufferString>()?;
    m.add_class::<PyVortexList>()?;
    m.add_class::<PyVortexStruct>()?;

    Ok(())
}

/// Base class for Vortex scalar types.
#[pyclass(name = "Scalar", module = "vortex", subclass, frozen, eq, hash)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyScalar(Scalar);

/// A marker trait indicating a PyO3 class is a subclass of a Vortex `Scalar`.
pub trait ScalarSubclass: PyClass<BaseType = PyScalar> {
    type Scalar<'a>;
}

/// A trait for extracting a typed and borrowed scalar from a [`Scalar`].
///
/// This is functionally the same as `AsRef` trait, except that the result is an owned type
/// with a lifetime, instead of a reference with a lifetime.
pub trait AsScalarRef<'a, T: 'a> {
    fn as_scalar_ref(&'a self) -> T;
}

/// Implement downcasting a `PyScalar` per the subclass in the marker trait.
impl<'a, T: ScalarSubclass> AsScalarRef<'a, <T as ScalarSubclass>::Scalar<'a>> for PyRef<'a, T>
where
    for<'b> <T as ScalarSubclass>::Scalar<'b>: TryFrom<&'b Scalar, Error = VortexError>,
{
    fn as_scalar_ref(&self) -> <T as ScalarSubclass>::Scalar<'_> {
        <<T as ScalarSubclass>::Scalar<'_>>::try_from(self.as_super().inner())
            .vortex_expect("Failed to downcast scalar")
    }
}

impl PyScalar {
    /// Initialize a [`PyScalar`] from a Vortex [`Scalar`], ensuring the correct subclass is
    /// returned.
    pub fn init(py: Python, scalar: Scalar) -> PyResult<Bound<PyScalar>> {
        // TODO(ngates): Bound::as_super would be great, but it's in newer PyO3.
        match scalar.dtype() {
            DType::Bool(_) => Self::with_subclass(py, scalar, PyBoolScalar),
            DType::Primitive(..) => Self::with_subclass(py, scalar, PyPrimitiveScalar),
            _ => unreachable!(),
        }
    }

    /// Initialize a [`PyScalar`] from a Vortex [`Scalar`], with the given subclass.
    /// We keep this a private method to ensure we correctly match on the scalar DType in init.
    fn with_subclass<S: PyClass<BaseType = PyScalar>>(
        py: Python,
        scalar: Scalar,
        subclass: S,
    ) -> PyResult<Bound<PyScalar>> {
        Ok(Bound::new(
            py,
            PyClassInitializer::from(PyScalar(scalar)).add_subclass(subclass),
        )?
        .into_any()
        .downcast_into::<PyScalar>()?)
    }

    /// Return the inner [`Scalar`] value.
    pub fn inner(&self) -> &Scalar {
        &self.0
    }

    /// Return the inner [`Scalar`] value.
    #[allow(dead_code)]
    pub fn into_inner(self) -> Scalar {
        self.0
    }
}

/// Define the interface methods of a `PyScalar`. Note that all children should override these
/// methods and there's currently no good way to do this in PyO3.
#[pymethods]
impl PyScalar {
    pub fn as_py(&self) -> PyResult<PyScalar> {
        Err(PyNotImplementedError::new_err(
            "Scalar subclass should implement as_py",
        ))
    }
}

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
    inner: ByteBuffer,
}

impl PyBuffer {
    pub fn new(inner: ByteBuffer) -> PyBuffer {
        PyBuffer { inner }
    }

    pub fn new_bound(py: Python, inner: ByteBuffer) -> PyResult<Bound<PyBuffer>> {
        Bound::new(py, Self::new(inner))
    }

    pub fn new_pyobject(py: Python, inner: ByteBuffer) -> PyResult<PyObject> {
        let bound = Bound::new(py, Self::new(inner))?;
        Ok(bound.into_py(py))
    }

    pub fn unwrap(&self) -> &ByteBuffer {
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
        .vortex_expect("non-null")
        .into_iter()
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
    // TODO(ngates): rename this as_py() to match Arrow
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
