mod array;
mod compute;
mod decode;
mod ops;
mod serde;

pub use array::*;

use crate::vtable::{NotSupported, VTable};
use crate::{EncodingId, EncodingRef, vtable};

vtable!(Chunked);

impl VTable for ChunkedVTable {
    type Array = ChunkedArray;
    type Encoding = ChunkedEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = Self;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.chunked")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ChunkedEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct ChunkedEncoding;
