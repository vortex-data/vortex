mod array;
mod compute;
mod decode;
mod ops;
mod serde;

use arcref::ArcRef;
pub use array::*;

use crate::vtable::VTable;
use crate::{EncodingRef, vtable};

vtable!(Chunked);

impl VTable for ChunkedVTable {
    type Array = ChunkedArray;
    type Encoding = ChunkedEncoding;

    type ArrayVTable = Self;
    type DecodeVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> ArcRef<str> {
        ArcRef::new_ref("vortex.chunked")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        ArcRef::new_ref(&ChunkedEncoding)
    }
}

#[derive(Debug)]
pub struct ChunkedEncoding;
