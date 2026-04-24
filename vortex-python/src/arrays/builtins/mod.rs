// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use vortex::array::arrays::Bool;
use vortex::array::arrays::Extension;
use vortex::array::arrays::FixedSizeList;
use vortex::array::arrays::List;
use vortex::array::arrays::Null;
use vortex::array::arrays::VarBin;
use vortex::array::arrays::VarBinView;
use vortex::encodings::bytebool::ByteBool;

use crate::arrays::native::EncodingSubclass;
use crate::arrays::native::PyNativeArray;

/// Concrete class for arrays with `vortex.null` encoding.
#[pyclass(name = "NullArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyNullArray;

impl EncodingSubclass for PyNullArray {
    type VTable = Null;
}

/// Concrete class for arrays with `vortex.bool` encoding.
#[pyclass(name = "BoolArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyBoolArray;

impl EncodingSubclass for PyBoolArray {
    type VTable = Bool;
}

/// Concrete class for arrays with `vortex.bytebool` encoding.
#[pyclass(name = "ByteBoolArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyByteBoolArray;

impl EncodingSubclass for PyByteBoolArray {
    type VTable = ByteBool;
}

/// Concrete class for arrays with `vortex.varbin` encoding.
#[pyclass(name = "VarBinArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyVarBinArray;

impl EncodingSubclass for PyVarBinArray {
    type VTable = VarBin;
}

/// Concrete class for arrays with `vortex.varbinview` encoding.
#[pyclass(name = "VarBinViewArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyVarBinViewArray;

impl EncodingSubclass for PyVarBinViewArray {
    type VTable = VarBinView;
}

/// Concrete class for arrays with `vortex.list` encoding.
#[pyclass(name = "ListArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyListArray;

impl EncodingSubclass for PyListArray {
    type VTable = List;
}

/// Concrete class for arrays with `vortex.fixed_size_list` encoding.
#[pyclass(name = "FixedSizeListArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyFixedSizeListArray;

impl EncodingSubclass for PyFixedSizeListArray {
    type VTable = FixedSizeList;
}

/// Concrete class for arrays with `vortex.ext` encoding.
#[pyclass(name = "ExtensionArray", module = "vortex", extends=PyNativeArray, frozen)]
pub(crate) struct PyExtensionArray;

impl EncodingSubclass for PyExtensionArray {
    type VTable = Extension;
}
