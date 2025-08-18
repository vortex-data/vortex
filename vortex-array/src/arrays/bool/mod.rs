// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub mod compute;
mod ops;
mod patch;
mod serde;
#[cfg(feature = "test-harness")]
mod test;

pub use array::*;
// Re-export the BooleanBuffer type on our API surface.
pub use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};

use crate::vtable::{NotSupported, VTable, ValidityVTableFromValidityHelper};
use crate::{EncodingId, EncodingRef, vtable};

vtable!(Bool);

impl VTable for BoolVTable {
    type Array = BoolArray;
    type Encoding = BoolEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    // Enable serde for this encoding
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.bool")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(BoolEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct BoolEncoding;
