mod chunked;
mod constant;
mod primitive;
mod struct_;

pub(crate) use chunked::*;
pub(crate) use constant::*;
pub(crate) use primitive::*;
use pyo3::prelude::*;
pub(crate) use struct_::*;
use vortex::arrays::{
    BoolEncoding, ExtensionEncoding, ListEncoding, NullEncoding, VarBinEncoding, VarBinViewEncoding,
};
use vortex::encodings::bytebool::ByteBoolEncoding;

use crate::arrays::native::{EncodingSubclass, PyNativeArray};

/// Concrete class for arrays with `vortex.null` encoding.
#[pyclass(name = "NullArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyNullArray;

impl EncodingSubclass for PyNullArray {
    type Encoding = NullEncoding;
}

/// Concrete class for arrays with `vortex.bool` encoding.
#[pyclass(name = "BoolArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyBoolArray;

impl EncodingSubclass for PyBoolArray {
    type Encoding = BoolEncoding;
}

/// Concrete class for arrays with `vortex.bytebool` encoding.
#[pyclass(name = "ByteBoolArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyByteBoolArray;

impl EncodingSubclass for PyByteBoolArray {
    type Encoding = ByteBoolEncoding;
}

/// Concrete class for arrays with `vortex.varbin` encoding.
#[pyclass(name = "VarBinArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyVarBinArray;

impl EncodingSubclass for PyVarBinArray {
    type Encoding = VarBinEncoding;
}

/// Concrete class for arrays with `vortex.varbinview` encoding.
#[pyclass(name = "VarBinViewArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyVarBinViewArray;

impl EncodingSubclass for PyVarBinViewArray {
    type Encoding = VarBinViewEncoding;
}

/// Concrete class for arrays with `vortex.list` encoding.
#[pyclass(name = "ListArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyListArray;

impl EncodingSubclass for PyListArray {
    type Encoding = ListEncoding;
}

/// Concrete class for arrays with `vortex.ext` encoding.
#[pyclass(name = "ExtensionArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyExtensionArray;

impl EncodingSubclass for PyExtensionArray {
    type Encoding = ExtensionEncoding;
}
