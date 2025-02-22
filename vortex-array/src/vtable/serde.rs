use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::serde::ArrayParts;
use crate::vtable::EncodingVTable;
use crate::{Array, ArrayRef, ContextRef, Encoding};

pub trait SerdeVTable<Array> {
    /// Encode the metadata of an array.
    ///
    /// Note that there are no alignment guarantees for the metadata buffer during deserialization,
    /// therefore this function returns a [`Vec<u8>`] instead of a [`ByteBuffer`].
    ///
    /// Returning `None` indicates that the encoding does not require encoded metadata.
    fn encode(&self, array: Array) -> Option<Vec<u8>> {
        None
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
    fn encode(&self, array: &'a dyn Array) -> Option<Vec<u8>> {
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
        <E as SerdeVTable<&'a E::Array>>::decode(self, parts, ctx, dtype, len)
    }
}
