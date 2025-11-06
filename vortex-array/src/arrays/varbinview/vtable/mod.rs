// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::varbinview::VarBinViewArray;
use crate::vtable::{NotSupported, VTable, ValidityVTableFromValidityHelper};
use crate::{EncodingId, EncodingRef, vtable};

mod array;
mod canonical;
mod operations;
mod operator;
mod serde;
mod validity;
mod visitor;

vtable!(VarBinView);

impl VTable for VarBinViewVTable {
    type Array = VarBinViewArray;
    type Encoding = VarBinViewEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type OperatorVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.varbinview")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(VarBinViewEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct VarBinViewEncoding;
