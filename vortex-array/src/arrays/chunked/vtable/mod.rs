// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::ChunkedArray;
use crate::vtable::{NotSupported, VTable};
use crate::{vtable, EncodingId, EncodingRef};

mod array;
mod canonical;
mod compute;
mod operations;
mod serde;
mod validity;
mod visitor;

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
    type OperatorVTable = NotSupported;
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
