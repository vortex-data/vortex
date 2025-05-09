mod array;
pub mod compute;
mod ops;
mod patch;
mod serde;

use arcref::ArcRef;
pub use array::*;
// Re-export the BooleanBuffer type on our API surface.
pub use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};

use crate::vtable::{VTable, ValidityVTableFromValidityChild};
use crate::{EncodingRef, vtable};

vtable!(Bool);

impl VTable for BoolVTable {
    type Array = BoolArray;
    type Encoding = BoolEncoding;

    type ArrayVTable = Self;
    type DecodeVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityChild;
    type VisitorVTable = Self;
    // Enable serde for this encoding
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> ArcRef<str> {
        ArcRef::new_ref("vortex.bool")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        ArcRef::new_ref(&BoolEncoding)
    }
}

#[derive(Debug)]
pub struct BoolEncoding;
