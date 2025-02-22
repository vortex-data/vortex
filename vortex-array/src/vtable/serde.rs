use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::serde::ArrayParts;
use crate::visitor::ArrayVisitor;
use crate::vtable::EncodingVTable;
use crate::{Array, ArrayRef, ContextRef, Encoding};

// TODO(ngates): need a new name for this VTable.
pub trait SerdeVTable<Array> {
    /// Encode the metadata of an array.
    ///
    /// Note that there are no alignment guarantees for the metadata buffer during deserialization,
    /// therefore this function returns a [`Vec<u8>`] instead of a [`ByteBuffer`].
    ///
    /// Returning `None` indicates that the encoding does not require encoded metadata.
    fn metadata(&self, array: Array) -> Option<Vec<u8>> {
        None
    }

    /// Get the children of an array.
    fn children(&self, array: Array, visitor: &dyn Fn(&str, &dyn crate::Array)) {}

    /// Get the number of children of an array.
    fn nchildren(&self, array: Array) -> usize {
        let mut n = 0;
        self.children(array, &|_, _| n += 1);
        n
    }

    /// Get the buffers of the array.
    fn buffers(&self, array: Array, visitor: &dyn Fn(&ByteBuffer)) {}

    /// Get the number of buffers of the array.
    fn nbuffers(&self, array: Array) -> usize {
        let mut n = 0;
        self.buffers(array, &|_| n += 1);
        n
    }

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
    fn metadata(&self, array: &'a dyn Array) -> Option<Vec<u8>> {
        SerdeVTable::metadata(
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
        ctx: &ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        <E as SerdeVTable<&'a E::Array>>::decode(self, parts, ctx, dtype, len)
    }
}
