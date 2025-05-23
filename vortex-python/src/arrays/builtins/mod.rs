mod chunked;
mod constant;
mod decimal;
mod primitive;
mod struct_;

pub(crate) use chunked::*;
pub(crate) use constant::*;
pub(crate) use decimal::*;
pub(crate) use primitive::*;
use pyo3::prelude::*;
pub(crate) use struct_::*;
use vortex::arrays::{
    BoolVTable, ExtensionVTable, ListVTable, NullVTable, VarBinVTable, VarBinViewVTable,
};
use vortex::encodings::bytebool::ByteBoolVTable;

use crate::arrays::native::{EncodingSubclass, PyNativeArray};

/// Concrete class for arrays with `vortex.null` encoding.
#[pyclass(name = "NullArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyNullArray;

impl EncodingSubclass for PyNullArray {
    type VTable = NullVTable;
}

/// Concrete class for arrays with `vortex.bool` encoding.
#[pyclass(name = "BoolArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyBoolArray;

impl EncodingSubclass for PyBoolArray {
    type VTable = BoolVTable;
}

/// Concrete class for arrays with `vortex.bytebool` encoding.
#[pyclass(name = "ByteBoolArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyByteBoolArray;

impl EncodingSubclass for PyByteBoolArray {
    type VTable = ByteBoolVTable;
}

/// Concrete class for arrays with `vortex.varbin` encoding.
#[pyclass(name = "VarBinArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyVarBinArray;

impl EncodingSubclass for PyVarBinArray {
    type VTable = VarBinVTable;
}

/// Concrete class for arrays with `vortex.varbinview` encoding.
#[pyclass(name = "VarBinViewArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyVarBinViewArray;

impl EncodingSubclass for PyVarBinViewArray {
    type VTable = VarBinViewVTable;
}

/// Concrete class for arrays with `vortex.list` encoding.
#[pyclass(name = "ListArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyListArray;

impl EncodingSubclass for PyListArray {
    type VTable = ListVTable;
}

/// Concrete class for arrays with `vortex.ext` encoding.
#[pyclass(name = "ExtensionArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyExtensionArray;

impl EncodingSubclass for PyExtensionArray {
    type VTable = ExtensionVTable;
}
