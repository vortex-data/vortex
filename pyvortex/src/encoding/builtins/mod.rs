mod chunked;
mod constant;
mod struct_;

pub(crate) use chunked::*;
pub(crate) use constant::*;
use pyo3::prelude::*;
pub(crate) use struct_::*;

use crate::arrays::PyArray;

/// Concrete class for arrays with `vortex.null` encoding.
#[pyclass(name = "NullEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyNullEncoding;

/// Concrete class for arrays with `vortex.bool` encoding.
#[pyclass(name = "BoolEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyBoolEncoding;

/// Concrete class for arrays with `vortex.bytebool` encoding.
#[pyclass(name = "ByteBoolEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyByteBoolEncoding;

/// Concrete class for arrays with `vortex.primitive` encoding.
#[pyclass(name = "PrimitiveEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyPrimitiveEncoding;

/// Concrete class for arrays with `vortex.varbin` encoding.
#[pyclass(name = "VarBinEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyVarBinEncoding;

/// Concrete class for arrays with `vortex.varbinview` encoding.
#[pyclass(name = "VarBinViewEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyVarBinViewEncoding;

/// Concrete class for arrays with `vortex.list` encoding.
#[pyclass(name = "ListEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyListEncoding;

/// Concrete class for arrays with `vortex.ext` encoding.
#[pyclass(name = "ExtensionEncoding", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyExtensionEncoding;
