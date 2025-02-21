use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};

use crate::serde::ArrayParts;
use crate::vtable::EncodingVTable;
use crate::{Array, ArrayRef, ContextRef, Encoding};

pub trait SerdeVTable<A> {
    /// Encode an array into an [`ArrayParts`],
    fn encode(&self, array: &A) -> VortexResult<ArrayParts> {
        vortex_bail!("Encoding not supported for encoding")
    }

    /// Decode an [`ArrayParts`] into an [`ArrayRef`] of this encoding.
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!("Decoding not supported for encoding")
    }
}

impl<'a, E: Encoding> SerdeVTable<&'a dyn Array> for E
where
    E: SerdeVTable<&'a E::Array>,
{
    fn encode(&self, array: &&'a dyn Array) -> VortexResult<ArrayParts> {
        self.encode(array.downcast_ref::<E::Array>().unwrap())
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        self.decode(parts, ctx, dtype, len)
    }
}
