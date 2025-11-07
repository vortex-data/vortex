// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::vtable::{NotSupported, VTable, ValidityVTableFromValidityHelper};
use vortex_array::{EncodingId, EncodingRef, vtable};

use crate::BitPackedArray;

mod array;
mod canonical;
mod encode;
mod operations;
mod serde;
mod validity;
mod visitor;

vtable!(BitPacked);

impl VTable for BitPackedVTable {
    type Array = BitPackedArray;
    type Encoding = BitPackedEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;
    type OperatorVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("fastlanes.bitpacked")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(BitPackedEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct BitPackedEncoding;
