use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::serde::ArrayParts;
use crate::vtable::EncodingVTable;
use crate::{Array, ArrayRef, ContextRef, Encoding};

// TODO(ngates): need a new name for this VTable.
pub trait SerdeVTable<Array> {
    /// Decode an [`ArrayParts`] into an [`ArrayRef`] of this encoding.
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!(
            "Decoding not supported for encoding {}",
            ctx.lookup_encoding(parts.encoding_id())
                .vortex_expect("Encoding already validated")
                .id()
        )
    }
}

impl<'a, E: Encoding> SerdeVTable<&'a dyn Array> for E
where
    E: SerdeVTable<&'a E::Array>,
{
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        <E as SerdeVTable<&'a E::Array>>::decode(self, parts, ctx, dtype, len)
    }
}
