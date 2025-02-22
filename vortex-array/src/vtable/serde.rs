use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::serde::ArrayParts;
use crate::vtable::EncodingVTable;
use crate::{Array, ArrayRef, ContextRef, Encoding};

pub trait SerdeVTable<A: ?Sized> {
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

impl<E: Encoding> SerdeVTable<dyn Array> for E
where
    E: SerdeVTable<E::Array>,
{
    fn encode(&self, array: &dyn Array) -> VortexResult<ArrayParts> {
        SerdeVTable::encode(
            self,
            array
                .as_any()
                .downcast_ref::<E::Array>()
                .vortex_expect("Failed to downcast array"),
        )
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        <E as SerdeVTable<E::Array>>::decode(self, parts, ctx, dtype, len)
    }
}
