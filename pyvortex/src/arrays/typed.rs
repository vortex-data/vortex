use pyo3::prelude::*;

use crate::arrays::PyArray;

/// Concrete class for arrays of :class:`~vortex.NullDType`.
#[pyclass(name = "NullTypeArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyNullTypeArray;

/// Concrete class for arrays of :class:`~vortex.BoolDType`.
#[pyclass(name = "BoolTypeArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyBoolTypeArray;

/// Concrete class for arrays of any primitive type :class:`~vortex.PrimitiveDType`.
#[pyclass(name = "PrimitiveTypeArray", module = "vortex", extends=PyArray, frozen, subclass)]
pub(crate) struct PyPrimitiveTypeArray;

/// Concrete class for arrays of any primitive signed or unsigned integer type :class:`~vortex.PrimitiveDType`.
#[pyclass(name = "IntegerTypeArray", module = "vortex", extends=PyPrimitiveTypeArray, frozen, subclass)]
pub(crate) struct PyIntegerTypeArray;

/// Concrete class for arrays of any primitive unsigned integer type :class:`~vortex.PrimitiveDType`.
#[pyclass(name = "UIntTypeArray", module = "vortex", extends=PyIntegerTypeArray, frozen, subclass)]
pub(crate) struct PyUIntTypeArray;

/// Concrete class for arrays of u8 :class:`~vortex.PrimitiveDType`.
#[pyclass(name = "UInt8TypeArray", module = "vortex", extends=PyUIntTypeArray, frozen)]
pub(crate) struct PyUInt8TypeArray;

/// Concrete class for arrays of u16 :class:`~vortex.PrimitiveDType`.
#[pyclass(name = "UInt16TypeArray", module = "vortex", extends=PyUIntTypeArray, frozen)]
pub(crate) struct PyUInt16TypeArray;

/// Concrete class for arrays of u32 :class:`~vortex.PrimitiveDType`.
#[pyclass(name = "UInt32TypeArray", module = "vortex", extends=PyUIntTypeArray, frozen)]
pub(crate) struct PyUInt32TypeArray;

/// Concrete class for arrays of u64 :class:`~vortex.PrimitiveDType`.
#[pyclass(name = "UInt64TypeArray", module = "vortex", extends=PyUIntTypeArray, frozen)]
pub(crate) struct PyUInt64TypeArray;

/// Concrete class for arrays of any primitive signed integer type :class:`~vortex.PrimitiveDType`.
#[pyclass(name = "IntTypeArray", module = "vortex", extends=PyIntegerTypeArray, frozen, subclass)]
pub(crate) struct PyIntTypeArray;

/// Concrete class for arrays of i8 :class:`~vortex.PrimitiveDType`.
#[pyclass(name = "Int8TypeArray", module = "vortex", extends=PyIntTypeArray, frozen)]
pub(crate) struct PyInt8TypeArray;

/// Concrete class for arrays of i16 :class:`~vortex.PrimitiveDType`.
#[pyclass(name = "Int16TypeArray", module = "vortex", extends=PyIntTypeArray, frozen)]
pub(crate) struct PyInt16TypeArray;

/// Concrete class for arrays of i32 :class:`~vortex.PrimitiveDType`.
#[pyclass(name = "Int32TypeArray", module = "vortex", extends=PyIntTypeArray, frozen)]
pub(crate) struct PyInt32TypeArray;

/// Concrete class for arrays of i64 :class:`~vortex.PrimitiveDType`.
#[pyclass(name = "Int64TypeArray", module = "vortex", extends=PyIntTypeArray, frozen)]
pub(crate) struct PyInt64TypeArray;

/// Concrete class for arrays of any primitive floating point type :class:`~vortex.PrimitiveDType`.
#[pyclass(name = "FloatTypeArray", module = "vortex", extends=PyPrimitiveTypeArray, frozen, subclass)]
pub(crate) struct PyFloatTypeArray;

/// Concrete class for arrays of f16 :class:`~vortex.PrimitiveDType`.
#[pyclass(name = "Float16TypeArray", module = "vortex", extends=PyFloatTypeArray, frozen)]
pub(crate) struct PyFloat16TypeArray;

/// Concrete class for arrays of f32 :class:`~vortex.PrimitiveDType`.
#[pyclass(name = "Float32TypeArray", module = "vortex", extends=PyFloatTypeArray, frozen)]
pub(crate) struct PyFloat32TypeArray;

/// Concrete class for arrays of f64 :class:`~vortex.PrimitiveDType`.
#[pyclass(name = "Float64TypeArray", module = "vortex", extends=PyFloatTypeArray, frozen)]
pub(crate) struct PyFloat64TypeArray;

/// Concrete class for arrays of :class:`~vortex.Utf8DType`.
#[pyclass(name = "Utf8TypeArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyUtf8TypeArray;

/// Concrete class for arrays of :class:`~vortex.BinaryDType`.
#[pyclass(name = "BinaryTypeArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyBinaryTypeArray;

/// Concrete class for arrays of :class:`~vortex.StructDType`.
#[pyclass(name = "StructTypeArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyStructTypeArray;

/// Concrete class for arrays of :class:`~vortex.ListDType`.
#[pyclass(name = "ListTypeArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyListTypeArray;

/// Concrete class for arrays of :class:`~vortex.ExtensionDType`.
#[pyclass(name = "ExtensionTypeArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyExtensionTypeArray;
