// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod canonical;
mod operations;
mod serde;
mod validity;

use crate::arrays::masked::MaskedArray;
use crate::vtable::{NotSupported, VTable, ValidityVTableFromValidityHelper};
use crate::{EncodingId, EncodingRef, vtable};

vtable!(Masked);

#[derive(Clone, Debug)]
pub struct MaskedEncoding;

impl VTable for MaskedVTable {
    type Array = MaskedArray;
    type Encoding = MaskedEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;
    type OperatorVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.masked")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(MaskedEncoding.as_ref())
    }
}
