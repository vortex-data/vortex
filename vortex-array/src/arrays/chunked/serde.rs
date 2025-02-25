use itertools::Itertools;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::arrays::{ChunkedArray, ChunkedEncoding, PrimitiveArray};
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::SerdeVTable;
use crate::{
    Array, ArrayChildVisitor, ArrayRef, ArrayVisitorImpl, ContextRef, EmptyMetadata, ToCanonical,
};

impl ArrayVisitorImpl for ChunkedArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        let chunk_offsets = PrimitiveArray::new(self.chunk_offsets.clone(), Validity::NonNullable);
        visitor.visit_child("chunk_offsets", &chunk_offsets);

        for (idx, chunk) in self.chunks().iter().enumerate() {
            visitor.visit_child(format!("chunks[{}]", idx).as_str(), chunk);
        }
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}

impl SerdeVTable<&ChunkedArray> for ChunkedEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
        dtype: DType,
        // TODO(ngates): should we avoid storing the final chunk offset and push the length instead?
        _len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nchildren() == 0 {
            vortex_bail!("Chunked array needs at least one child");
        }

        let nchunks = parts.nchildren() - 1;

        // The first child contains the row offsets of the chunks
        let chunk_offsets = parts
            .child(0)
            .decode(
                ctx,
                DType::Primitive(PType::U64, Nullability::NonNullable),
                // 1 extra offset for the end of the last chunk
                nchunks + 1,
            )?
            .to_primitive()?
            .buffer::<u64>();

        // The remaining children contain the actual data of the chunks
        let chunks = chunk_offsets
            .iter()
            .tuple_windows()
            .enumerate()
            .map(|(idx, (start, end))| {
                let chunk_len =
                    usize::try_from(end - start).vortex_expect("chunk length exceeds usize");
                parts.child(idx + 1).decode(ctx, dtype.clone(), chunk_len)
            })
            .try_collect()?;

        // Unchecked because we just created each chunk with the same DType.
        Ok(ChunkedArray::new_unchecked(chunks, dtype).into_array())
    }
}
