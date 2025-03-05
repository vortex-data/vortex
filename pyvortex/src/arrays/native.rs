use std::ops::Deref;

use pyo3::PyClass;
use pyo3::prelude::*;
use vortex::dtype::{DType, PType};
use vortex::error::VortexExpect;
use vortex::nbytes::NBytes;
use vortex::{Array, ArrayRef, Encoding};

use crate::arrays::PyArray;
use crate::arrays::typed::{
    PyBinaryTypeArray, PyBoolTypeArray, PyExtensionTypeArray, PyFloat16TypeArray,
    PyFloat32TypeArray, PyFloat64TypeArray, PyFloatTypeArray, PyInt8TypeArray, PyInt16TypeArray,
    PyInt32TypeArray, PyInt64TypeArray, PyIntTypeArray, PyIntegerTypeArray, PyListTypeArray,
    PyNullTypeArray, PyPrimitiveTypeArray, PyStructTypeArray, PyUInt8TypeArray, PyUInt16TypeArray,
    PyUInt32TypeArray, PyUInt64TypeArray, PyUIntTypeArray, PyUtf8TypeArray,
};
use crate::dtype::PyDType;

#[pyclass(name = "NativeArray", module = "vortex", extends=PyArray, sequence, subclass, frozen)]
pub struct PyNativeArray(ArrayRef);

impl Deref for PyNativeArray {
    type Target = ArrayRef;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PyNativeArray {
    /// Initialize a [`PyArray`] from a Vortex [`ArrayRef`], ensuring we return the correct typed
    /// subclass array.
    pub fn init(py: Python, array: ArrayRef) -> PyResult<Bound<PyNativeArray>> {
        fn unsigned(array: ArrayRef) -> PyClassInitializer<PyUIntTypeArray> {
            PyClassInitializer::from(PyArray)
                .add_subclass(PyNativeArray(array))
                .add_subclass(PyPrimitiveTypeArray)
                .add_subclass(PyIntegerTypeArray)
                .add_subclass(PyUIntTypeArray)
        }

        fn signed(array: ArrayRef) -> PyClassInitializer<PyIntTypeArray> {
            PyClassInitializer::from(PyArray)
                .add_subclass(PyNativeArray(array))
                .add_subclass(PyPrimitiveTypeArray)
                .add_subclass(PyIntegerTypeArray)
                .add_subclass(PyIntTypeArray)
        }

        fn float(array: ArrayRef) -> PyClassInitializer<PyFloatTypeArray> {
            PyClassInitializer::from(PyArray)
                .add_subclass(PyNativeArray(array))
                .add_subclass(PyPrimitiveTypeArray)
                .add_subclass(PyFloatTypeArray)
        }

        match array.dtype() {
            DType::Null => Self::with_subclass(py, array, PyNullTypeArray),
            DType::Bool(_) => Self::with_subclass(py, array, PyBoolTypeArray),
            DType::Primitive(ptype, _) => match ptype {
                PType::U8 => Self::with_subclass_initializer(
                    py,
                    unsigned(array).add_subclass(PyUInt8TypeArray),
                ),
                PType::U16 => Self::with_subclass_initializer(
                    py,
                    unsigned(array).add_subclass(PyUInt16TypeArray),
                ),
                PType::U32 => Self::with_subclass_initializer(
                    py,
                    unsigned(array).add_subclass(PyUInt32TypeArray),
                ),
                PType::U64 => Self::with_subclass_initializer(
                    py,
                    unsigned(array).add_subclass(PyUInt64TypeArray),
                ),
                PType::I8 => {
                    Self::with_subclass_initializer(py, signed(array).add_subclass(PyInt8TypeArray))
                }
                PType::I16 => Self::with_subclass_initializer(
                    py,
                    signed(array).add_subclass(PyInt16TypeArray),
                ),
                PType::I32 => Self::with_subclass_initializer(
                    py,
                    signed(array).add_subclass(PyInt32TypeArray),
                ),
                PType::I64 => Self::with_subclass_initializer(
                    py,
                    signed(array).add_subclass(PyInt64TypeArray),
                ),
                PType::F16 => Self::with_subclass_initializer(
                    py,
                    float(array).add_subclass(PyFloat16TypeArray),
                ),
                PType::F32 => Self::with_subclass_initializer(
                    py,
                    float(array).add_subclass(PyFloat32TypeArray),
                ),
                PType::F64 => Self::with_subclass_initializer(
                    py,
                    float(array).add_subclass(PyFloat64TypeArray),
                ),
            },
            DType::Utf8(_) => Self::with_subclass(py, array, PyUtf8TypeArray),
            DType::Binary(_) => Self::with_subclass(py, array, PyBinaryTypeArray),
            DType::Struct(..) => Self::with_subclass(py, array, PyStructTypeArray),
            DType::List(..) => Self::with_subclass(py, array, PyListTypeArray),
            DType::Extension(_) => Self::with_subclass(py, array, PyExtensionTypeArray),
        }
    }

    fn with_subclass<S: PyClass<BaseType = PyNativeArray>>(
        py: Python,
        array: ArrayRef,
        subclass: S,
    ) -> PyResult<Bound<PyNativeArray>> {
        Ok(Bound::new(
            py,
            PyClassInitializer::from(PyArray)
                .add_subclass(PyNativeArray(array))
                .add_subclass(subclass),
        )?
        .into_any()
        .downcast_into::<PyNativeArray>()?)
    }

    fn with_subclass_initializer<S: PyClass>(
        py: Python,
        intializer: PyClassInitializer<S>,
    ) -> PyResult<Bound<PyNativeArray>> {
        Ok(Bound::new(py, intializer)?
            .into_any()
            .downcast_into::<PyNativeArray>()?)
    }

    pub fn inner(&self) -> &ArrayRef {
        &self.0
    }

    pub fn into_inner(self) -> ArrayRef {
        self.0
    }
}

#[pymethods]
impl PyNativeArray {
    fn __len__(&self) -> usize {
        self.len()
    }

    fn __str__(&self) -> String {
        format!("{}", self.0)
    }

    /// Returns the encoding ID of this array.
    #[getter]
    fn id(&self) -> String {
        self.0.encoding().to_string()
    }

    /// Returns the number of bytes used by this array.
    #[getter]
    fn nbytes(&self) -> usize {
        self.0.nbytes()
    }

    #[getter]
    fn dtype(self_: PyRef<Self>) -> PyResult<Bound<PyDType>> {
        PyDType::init(self_.py(), self_.0.dtype().clone())
    }
}

/// A marker trait indicating a PyO3 class is a subclass of Vortex `Array`.
pub trait EncodingSubclass: PyClass<BaseType = PyNativeArray> {
    type Encoding: Encoding;
}

/// Unwrap a downcasted Vortex array from a `PyRef<ArraySubclass>`.
pub trait AsArrayRef<T> {
    fn as_array_ref(&self) -> &T;
}

impl<A: EncodingSubclass> AsArrayRef<<A::Encoding as Encoding>::Array> for PyRef<'_, A> {
    fn as_array_ref(&self) -> &<A::Encoding as Encoding>::Array {
        self.as_super()
            .inner()
            .as_any()
            .downcast_ref::<<A::Encoding as Encoding>::Array>()
            .vortex_expect("Failed to downcast array")
    }
}
