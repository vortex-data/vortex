// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for chunked arrays: each chunk streams in turn, with its rows
//! shifted by the chunk's offset.

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::chunk_iter::ChunkMut;
use crate::chunk_iter::ChunkSink;

pub(super) fn supports_decompress_chunks(array: ArrayView<'_, Chunked>) -> bool {
    array
        .iter_chunks()
        .all(|chunk| chunk.supports_decompress_chunks())
}

pub(super) fn decompress_chunks(
    array: ArrayView<'_, Chunked>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    let mut adapter = OffsetSink {
        start: 0,
        inner: sink,
    };
    for chunk in array.non_empty_chunks() {
        chunk.decompress_chunks(ctx, &mut adapter)?;
        adapter.start += chunk.len();
    }
    Ok(())
}

/// Shifts the rows of a chunk's stream to the chunk's position in the array.
struct OffsetSink<'a> {
    start: usize,
    inner: &'a mut dyn ChunkSink,
}

impl ChunkSink for OffsetSink<'_> {
    #[inline]
    fn accept(&mut self, chunk: ChunkMut<'_>, rows: Range<usize>) -> VortexResult<()> {
        self.inner
            .accept(chunk, self.start + rows.start..self.start + rows.end)
    }

    fn destination(&mut self, rows: Range<usize>) -> Option<ChunkMut<'_>> {
        self.inner
            .destination(self.start + rows.start..self.start + rows.end)
    }

    fn accept_written(&mut self, rows: Range<usize>) -> VortexResult<()> {
        self.inner
            .accept_written(self.start + rows.start..self.start + rows.end)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::test_harness::assert_streams_like_execute;

    #[test]
    fn chunked_streams_like_execute() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let chunks = [
            PrimitiveArray::from_option_iter((0..1500i32).map(|i| (i % 3 != 0).then_some(i)))
                .into_array(),
            PrimitiveArray::from_option_iter(Vec::<Option<i32>>::new()).into_array(),
            ConstantArray::new(Some(-7i32), 2100).into_array(),
            PrimitiveArray::new(buffer![1i32, 2, 3], crate::validity::Validity::AllValid)
                .into_array(),
        ];
        let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let array = ChunkedArray::try_new(chunks, dtype)?.into_array();
        assert_streams_like_execute(&array, &mut ctx)
    }
}
