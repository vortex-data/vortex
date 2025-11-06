// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::varbin::VarBinArray;
use crate::vtable::{NotSupported, VTable, ValidityVTableFromValidityHelper};
use crate::{EncodingId, EncodingRef, vtable};

mod array;
mod canonical;
mod operations;
mod operator;
mod serde;
mod validity;
mod visitor;

vtable!(VarBin);

impl VTable for VarBinVTable {
    type Array = VarBinArray;
    type Encoding = VarBinEncoding;
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
        EncodingId::new_ref("vortex.varbin")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(VarBinEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct VarBinEncoding;
