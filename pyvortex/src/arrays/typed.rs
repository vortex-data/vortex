use pyo3::prelude::*;

use crate::arrays::PyArray;

/// Concrete class for arrays of :class:`~vortex.NullDType`.
#[pyclass(name = "NullTypeArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyNullTypeArray;

/// Concrete class for arrays of :class:`~vortex.BoolDType`.
#[pyclass(name = "BoolTypeArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyBoolTypeArray;

/// Concrete class for arrays of :class:`~vortex.PrimitiveDType`.
// TODO(ngates): should we explode this into each PType? Probably, yes.
#[pyclass(name = "PrimitiveTypeArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyPrimitiveTypeArray;

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
