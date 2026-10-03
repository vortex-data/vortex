// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for filtered arrays.
//!
//! The child streams its decompressed blocks; each block is compacted **in place** against the
//! bits of the filter mask covering that block's rows, and the surviving prefix is forwarded
//! downstream. Compaction in place is sound because within a block the output index is always
//! `<=` the input index.
//!
//! Each block compacts with the SIMD compress kernel the level-wise filter would pick for the
//! mask, straight from the mask's bitmap, so streaming never builds the mask's indices or runs and
//! never materializes the child's full decompressed buffer.

use std::ops::Range;

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Filter;
use crate::arrays::filter::FilterArraySlotsExt;
use crate::arrays::filter::execute::buffer::ChunkCompactor;
use crate::chunk_iter::ChunkMut;
use crate::chunk_iter::ChunkSink;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;

pub(crate) fn supports_decompress_chunks(array: ArrayView<'_, Filter>) -> bool {
    array.child().supports_decompress_chunks()
}

pub(crate) fn decompress_chunks(
    array: ArrayView<'_, Filter>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    // A zero-length mask is both all-true and all-false, so check the empty case first.
    match array.filter_mask() {
        Mask::AllFalse(_) | Mask::AllTrue(0) => Ok(()),
        // Nothing is filtered out: forward the child's chunks untouched.
        Mask::AllTrue(_) => array.child().decompress_child_chunks(ctx, sink),
        Mask::Values(values) => match_each_native_ptype!(array.dtype().as_ptype(), |T| {
            let mut adapter = FilterChunkSink::<T> {
                bits: values.bit_buffer(),
                compactor: ChunkCompactor::new(values.density()),
                out_row: 0,
                inner: sink,
            };
            array.child().decompress_child_chunks(ctx, &mut adapter)
        }),
    }
}

/// Sink adapter that compacts each child chunk down to its surviving rows.
struct FilterChunkSink<'a, T> {
    /// The filter mask's bits, one per child row.
    bits: &'a BitBuffer,
    compactor: ChunkCompactor<T>,
    /// Number of rows already emitted downstream (i.e. the parent row cursor).
    out_row: usize,
    inner: &'a mut dyn ChunkSink,
}

impl<T: NativePType> ChunkSink for FilterChunkSink<'_, T> {
    fn accept(&mut self, mut chunk: ChunkMut<'_>, child_rows: Range<usize>) -> VortexResult<()> {
        let values = chunk.as_slice_mut::<T>();
        let kept = self.compactor.compact(values, &self.bits.slice(child_rows));
        if kept == 0 {
            // Nothing survived in this block: emit nothing rather than a zero-length chunk.
            return Ok(());
        }
        let start = self.out_row;
        self.out_row += kept;
        self.inner
            .accept(ChunkMut::new(&mut values[..kept]), start..self.out_row)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::FilterArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::NativePType;
    use crate::test_harness::assert_streams_like_execute;

    const LEN: usize = 5000;

    /// A primitive child whose chunks the filter compacts, of a width per SIMD kernel.
    fn values<T: NativePType>(cast: impl Fn(usize) -> T) -> ArrayRef {
        PrimitiveArray::from_iter((0..LEN).map(cast)).into_array()
    }

    /// Masks from all-but-empty to all-but-full, with runs of every shape, so every kernel and
    /// the scalar bitmap walk run, including on the partial last chunk.
    #[rstest]
    #[case::sparse(|i: usize| i % 97 == 3)]
    #[case::quarter(|i: usize| i % 16 < 4)]
    #[case::alternating(|i: usize| i.is_multiple_of(2))]
    #[case::dense(|i: usize| i % 16 != 7)]
    #[case::runs(|i: usize| (i / 300) % 3 != 1)]
    #[case::one_chunk(|i: usize| (1024..2048).contains(&i))]
    fn filter_streams_like_execute(#[case] keep: fn(usize) -> bool) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mask = Mask::from_iter((0..LEN).map(keep));
        #[expect(
            clippy::cast_possible_truncation,
            reason = "test values wrap on purpose"
        )]
        let children = [
            values(|i| i as u8),
            values(|i| i as i16),
            values(|i| i as u32),
            values(|i| i as f64),
        ];
        for child in children {
            let filtered = FilterArray::try_new(child.clone(), mask.clone())?.into_array();
            assert_streams_like_execute(&filtered, &mut ctx)?;
            // A sliced mask starts its bits mid-byte, as filters of sliced arrays do.
            let sliced = FilterArray::try_new(child.slice(13..LEN)?, mask.slice(13..LEN))?;
            assert_streams_like_execute(&sliced.into_array(), &mut ctx)?;
        }
        Ok(())
    }
}
