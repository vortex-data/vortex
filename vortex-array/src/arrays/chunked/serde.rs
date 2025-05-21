use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use super::ChunkedEncoding;
use crate::arrays::{ChunkedArray, ChunkedVTable, PrimitiveArray};
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::{SerdeVTable, VisitorVTable};
use crate::{ArrayBufferVisitor, ArrayChildVisitor, EmptyMetadata, ToCanonical};

impl SerdeVTable<ChunkedVTable> for ChunkedVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &ChunkedArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(EmptyMetadata))
    }

    fn build(
        _encoding: &ChunkedEncoding,
        dtype: &DType,
        _len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ChunkedArray> {
        if children.is_empty() {
            vortex_bail!("Chunked array needs at least one child");
        }

        let nchunks = children.len() - 1;

        // The first child contains the row offsets of the chunks
        let chunk_offsets = children
            .get(
                0,
                &DType::Primitive(PType::U64, Nullability::NonNullable),
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
                children.get(idx + 1, dtype, chunk_len)
            })
            .try_collect()?;

        // Unchecked because we just created each chunk with the same DType.
        Ok(ChunkedArray::new_unchecked(chunks, dtype.clone()))
    }
}

impl VisitorVTable<ChunkedVTable> for ChunkedVTable {
    fn visit_buffers(_array: &ChunkedArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ChunkedArray, visitor: &mut dyn ArrayChildVisitor) {
        let chunk_offsets =
            PrimitiveArray::new(array.chunk_offsets().clone(), Validity::NonNullable);
        visitor.visit_child("chunk_offsets", chunk_offsets.as_ref());

        for (idx, chunk) in array.chunks().iter().enumerate() {
            visitor.visit_child(format!("chunks[{idx}]").as_str(), chunk);
        }
    }
}
