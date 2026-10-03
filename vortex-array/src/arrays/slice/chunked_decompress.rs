// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for lazy slices.
//!
//! A slice stays lazy when its child can only be sliced with an execution context, e.g. to slice
//! its patches. Streaming first resolves it through the child's slice kernel, as the executor
//! would, so only the sliced rows are decoded. Otherwise the child streams in full and only the
//! rows in the slice are forwarded.

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Slice;
use crate::arrays::slice::SliceArraySlotsExt;
use crate::chunk_iter::ChunkMut;
use crate::chunk_iter::ChunkSink;

pub(super) fn supports_decompress_chunks(array: ArrayView<'_, Slice>) -> bool {
    array.child().supports_decompress_chunks()
}

pub(super) fn decompress_chunks(
    array: ArrayView<'_, Slice>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    if let Some(resolved) = array.array().try_execute_parent_kernels(ctx)?
        && resolved.supports_decompress_chunks()
    {
        return resolved.decompress_child_chunks(ctx, sink);
    }
    let mut adapter = TrimSink {
        range: array.slice_range().clone(),
        inner: sink,
    };
    array.child().decompress_child_chunks(ctx, &mut adapter)
}

/// Forwards only the rows of the child's stream that fall in `range`, renumbered from its start.
struct TrimSink<'a> {
    range: Range<usize>,
    inner: &'a mut dyn ChunkSink,
}

impl ChunkSink for TrimSink<'_> {
    #[inline]
    fn accept(&mut self, chunk: ChunkMut<'_>, rows: Range<usize>) -> VortexResult<()> {
        let start = rows.start.max(self.range.start);
        let end = rows.end.min(self.range.end);
        if start >= end {
            return Ok(());
        }
        self.inner.accept(
            chunk.narrow(start - rows.start..end - rows.start),
            start - self.range.start..end - self.range.start,
        )
    }

    /// Only chunks wholly inside the slice can be written in place.
    fn destination(&mut self, rows: Range<usize>) -> Option<ChunkMut<'_>> {
        (self.range.start <= rows.start && rows.end <= self.range.end).then_some(())?;
        self.inner
            .destination(rows.start - self.range.start..rows.end - self.range.start)
    }

    fn accept_written(&mut self, rows: Range<usize>) -> VortexResult<()> {
        self.inner
            .accept_written(rows.start - self.range.start..rows.end - self.range.start)
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::SliceArray;
    use crate::test_harness::assert_streams_like_execute;

    /// Slices with no slice kernel stream their child and forward only the sliced rows.
    #[test]
    fn slice_streams_like_execute() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let values =
            PrimitiveArray::from_option_iter((0..5000i32).map(|i| (i % 7 != 0).then_some(i)));
        for range in [517..4013, 0..1024, 1500..1600, 4999..5000] {
            let array = SliceArray::new(values.clone().into_array(), range).into_array();
            assert!(array.try_execute_parent_kernels(&mut ctx)?.is_none());
            assert_streams_like_execute(&array, &mut ctx)?;
        }
        let array = SliceArray::new(ConstantArray::new(3u8, 3000).into_array(), 1000..2900);
        assert_streams_like_execute(&array.into_array(), &mut ctx)
    }
}
