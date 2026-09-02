// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for filtered arrays.
//!
//! The child streams its decompressed blocks; each block is compacted **in place** against the
//! slice of the filter mask covering that block's rows, and the surviving prefix is forwarded
//! downstream. Compaction in place is sound because within a block the output index is always
//! `<=` the input index.
//!
//! The mask's run-slice representation is computed once up front and walked with a cursor, so no
//! per-chunk mask slicing or allocation happens on the hot path. This mirrors the kernel used by
//! the non-streaming path ([`filter_slice_mut_by_slices`](super::execute::slice)), but avoids
//! materializing the child's full decompressed buffer first.

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskIter;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Filter;
use crate::arrays::filter::FilterArraySlotsExt;
use crate::chunk_iter::ChunkMut;
use crate::chunk_iter::ChunkSink;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;

/// Below this mean run length, per-index gathering beats copying runs during in-place
/// compaction. Measured on `Filter(BitPacked)` at 64K rows: runs of 1 favor gathering by ~2x,
/// runs of 8 favor run-copying by ~1.7x.
const MIN_RUN_LEN_FOR_RUN_COPY: f64 = 4.0;

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
        Mask::AllTrue(_) => array.child().decompress_chunks(ctx, sink),
        Mask::Values(values) => {
            // Compaction here is in-place, so runs are preferred (fewer loop iterations and
            // branches than per-index gathers, as in `filter_slice_mut_by_mask_values`) — except
            // when runs are so short that run-copying degenerates into one-element `copy_within`
            // calls, where gathering indices measures faster.
            let runs = values.slices();
            let mean_run_len = values.true_count() as f64 / (runs.len().max(1) as f64);
            let selection = if mean_run_len < MIN_RUN_LEN_FOR_RUN_COPY {
                MaskIter::Indices(values.indices())
            } else {
                MaskIter::Slices(runs)
            };
            let mut adapter = FilterChunkSink {
                selection,
                cursor: 0,
                out_row: 0,
                inner: sink,
            };
            array.child().decompress_chunks(ctx, &mut adapter)
        }
    }
}

/// Sink adapter that compacts each child chunk down to its surviving rows.
struct FilterChunkSink<'a> {
    /// The mask's selected rows in child coordinates, as indices (sparse) or runs (dense).
    selection: MaskIter<'a>,
    /// Index of the first index/run that may still fall in a future chunk.
    cursor: usize,
    /// Number of rows already emitted downstream (i.e. the parent row cursor).
    out_row: usize,
    inner: &'a mut dyn ChunkSink,
}

impl ChunkSink for FilterChunkSink<'_> {
    fn accept(&mut self, mut chunk: ChunkMut<'_>, child_rows: Range<usize>) -> VortexResult<()> {
        match_each_native_ptype!(chunk.ptype(), |T| {
            compact_and_forward::<T>(
                &mut chunk,
                child_rows,
                &self.selection,
                &mut self.cursor,
                &mut self.out_row,
                self.inner,
            )
        })
    }
}

fn compact_and_forward<T: NativePType>(
    chunk: &mut ChunkMut<'_>,
    child_rows: Range<usize>,
    selection: &MaskIter<'_>,
    cursor: &mut usize,
    out_row: &mut usize,
    inner: &mut dyn ChunkSink,
) -> VortexResult<()> {
    let values = chunk.as_slice_mut::<T>();
    let out = match selection {
        MaskIter::Indices(indices) => compact_by_indices(values, &child_rows, indices, cursor),
        MaskIter::Slices(runs) => compact_by_runs(values, &child_rows, runs, cursor),
    };

    if out == 0 {
        // Nothing survived in this block: emit nothing rather than a zero-length chunk.
        return Ok(());
    }

    let start = *out_row;
    *out_row += out;
    inner.accept(ChunkMut::new(&mut values[..out]), start..*out_row)
}

/// Gather selected rows for sparse masks. In-place is sound because `out <= idx - start`.
fn compact_by_indices<T: Copy>(
    values: &mut [T],
    child_rows: &Range<usize>,
    indices: &[usize],
    cursor: &mut usize,
) -> usize {
    let mut out = 0usize;
    while let Some(&idx) = indices.get(*cursor) {
        if idx < child_rows.start {
            *cursor += 1;
            continue;
        }
        if idx >= child_rows.end {
            break;
        }
        values[out] = values[idx - child_rows.start];
        out += 1;
        *cursor += 1;
    }
    out
}

/// Copy selected runs for dense masks, clipping each run to the chunk.
fn compact_by_runs<T: Copy>(
    values: &mut [T],
    child_rows: &Range<usize>,
    runs: &[(usize, usize)],
    run_cursor: &mut usize,
) -> usize {
    let mut out = 0usize;

    while let Some(&(run_start, run_end)) = runs.get(*run_cursor) {
        if run_end <= child_rows.start {
            // Fully behind this chunk (only reachable if a chunk was empty).
            *run_cursor += 1;
            continue;
        }
        if run_start >= child_rows.end {
            break;
        }

        let from = run_start.max(child_rows.start) - child_rows.start;
        let to = run_end.min(child_rows.end) - child_rows.start;
        // In-place left-compaction: `out <= from` always holds because runs are ordered and
        // disjoint, so this never overwrites a value it has not yet copied.
        values.copy_within(from..to, out);
        out += to - from;

        if run_end <= child_rows.end {
            *run_cursor += 1;
        } else {
            // This run continues into the next chunk; keep the cursor on it.
            break;
        }
    }

    out
}
