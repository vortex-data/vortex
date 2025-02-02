mod struct_;

use pyo3::prelude::*;
pub(crate) use struct_::*;

use crate::arrays::PyArray;

/// Concrete class for arrays with `vortex.null` encoding.
#[pyclass(name = "NullArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyNullArray;

/// Concrete class for arrays with `vortex.bool` encoding.
#[pyclass(name = "BoolArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyBoolArray;

/// Concrete class for arrays with `vortex.primitive` encoding.
#[pyclass(name = "PrimitiveArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyPrimitiveArray;

/// Concrete class for arrays with `vortex.varbinview` encoding.
#[pyclass(name = "VarBinViewArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyVarBinViewArray;

/// Concrete class for arrays with `vortex.list` encoding.
#[pyclass(name = "ListArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyListArray;

/// Concrete class for arrays with `vortex.ext` encoding.
#[pyclass(name = "ExtensionArray", module = "vortex", extends=PyArray, frozen)]
pub(crate) struct PyExtensionArray;
