// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::PrimitiveArray;
use crate::vtable::{NotSupported, VTable, ValidityVTableFromValidityHelper};
use crate::{EncodingId, EncodingRef, vtable};

mod array;
mod canonical;
mod operations;
mod operator;
mod serde;
mod validity;
mod visitor;

vtable!(Primitive);

impl VTable for PrimitiveVTable {
    type Array = PrimitiveArray;
    type Encoding = PrimitiveEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;
    type OperatorVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.primitive")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(PrimitiveEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct PrimitiveEncoding;
