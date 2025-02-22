use itertools::Itertools;
use vortex_dtype::{DType, Nullability, PType, TryFromBytes};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::arrays::{ChunkedArray, ChunkedEncoding};
use crate::serde::ArrayParts;
use crate::vtable::SerdeVTable;
use crate::{
    encoding_ids, Array, ArrayRef, ContextRef, EmptyMetadata, Encoding, EncodingId, ToCanonical,
};

impl SerdeVTable<&ChunkedArray> for ChunkedEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nchildren() == 0 {
            vortex_bail!("Chunked array needs at least one child");
        }

        let nchunks = parts.nchildren() - 1;
        let children = parts.children();

        // The first child contains the row offsets of the chunks
        let chunk_offsets = children[0]
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
            .tuples()
            .enumerate()
            .map(|(idx, (start, end))| {
                let chunk_len =
                    usize::try_from(end - start).vortex_expect("chunk length exceeds usize");
                children[idx + 1].decode(ctx, dtype.clone(), chunk_len)
            })
            .try_collect()?;

        // Unchecked because we just created each chunk with the same DType.
        Ok(ChunkedArray::new_unchecked(chunks, dtype).into_array())
    }
}
