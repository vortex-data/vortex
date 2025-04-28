use std::iter;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use itertools::Itertools;
use vortex_array::ArrayContext;
use vortex_error::{VortexExpect, VortexResult, vortex_panic};

use crate::layouts::chunked::ChunkedLayout;
use crate::reader::LayoutReader;
use crate::segments::SegmentSource;
use crate::{Layout, LayoutVTable};

#[derive(Clone)]
pub struct ChunkedReader {
    layout: Layout,
    pub(crate) segment_source: Arc<dyn SegmentSource>,
    ctx: ArrayContext,

    /// Shared lazy chunk scanners
    chunk_readers: Arc<[OnceLock<Arc<dyn LayoutReader>>]>,
    /// Row offset for each chunk
    chunk_offsets: Vec<u64>,
}

impl ChunkedReader {
    pub(super) fn try_new(
        layout: Layout,
        segment_source: Arc<dyn SegmentSource>,
        ctx: ArrayContext,
    ) -> VortexResult<Self> {
        if layout.vtable().id() != ChunkedLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        // The number of chunks
        let nchunks = layout.nchildren();

        // Construct a lazy scan for each chunk of the layout.
        let chunk_readers = (0..nchunks).map(|_| OnceLock::new()).collect();

        // Generate the cumulative chunk offsets, relative to the layout's row offset, with an
        // additional offset corresponding to the length.
        let chunk_offsets = iter::once(0)
            .chain(
                (0..nchunks)
                    .map(|i| layout.child_row_count(i))
                    .scan(0, |state, x| {
                        *state += x;
                        Some(*state)
                    }),
            )
            .collect();

        Ok(Self {
            layout,
            segment_source,
            ctx,
            chunk_readers,
            chunk_offsets,
        })
    }

    /// Return the child reader for the chunk.
    pub(crate) fn child(&self, idx: usize) -> VortexResult<&Arc<dyn LayoutReader>> {
        self.chunk_readers[idx].get_or_try_init(|| {
            let child_layout =
                self.layout
                    .child(idx, self.layout.dtype().clone(), format!("[{}]", idx))?;
            child_layout.reader(&self.segment_source, &self.ctx)
        })
    }

    pub(crate) fn chunk_offset(&self, idx: usize) -> u64 {
        self.chunk_offsets[idx]
    }

    pub(crate) fn chunk_range(&self, row_range: &Range<u64>) -> Range<usize> {
        let start_chunk = self
            .chunk_offsets
            .binary_search(&row_range.start)
            .unwrap_or_else(|x| x - 1);
        let end_chunk = self
            .chunk_offsets
            .binary_search(&row_range.end)
            .unwrap_or_else(|x| x);
        start_chunk..end_chunk
    }

    pub(crate) fn ranges<'a>(
        &'a self,
        row_range: &'a Range<u64>,
    ) -> impl Iterator<Item = (usize, Range<u64>, Range<usize>)> + 'a {
        self.chunk_range(row_range).map(move |chunk_idx| {
            // Figure out the chunk row range relative to the mask's row range.
            let chunk_row_range = self.chunk_offset(chunk_idx)..self.chunk_offset(chunk_idx + 1);

            // Find the intersection of the mask and the chunk row ranges.
            let intersecting_row_range =
                row_range.start.max(chunk_row_range.start)..row_range.end.min(chunk_row_range.end);
            let intersecting_len =
                usize::try_from(intersecting_row_range.end - intersecting_row_range.start)
                    .vortex_expect("Invalid row range");

            // Figure out the offset into the mask.
            let mask_relative_start =
                usize::try_from(intersecting_row_range.start - row_range.start)
                    .vortex_expect("Invalid row range");
            let mask_relative_end = mask_relative_start + intersecting_len;
            let mask_range = mask_relative_start..mask_relative_end;

            // Figure out the row range within the chunk.
            let chunk_relative_start = intersecting_row_range.start - chunk_row_range.start;
            let chunk_relative_end = chunk_relative_start + intersecting_len as u64;
            let chunk_range = chunk_relative_start..chunk_relative_end;

            (chunk_idx, chunk_range, mask_range)
        })
    }
}

impl LayoutReader for ChunkedReader {
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn children(&self) -> VortexResult<Vec<Arc<dyn LayoutReader>>> {
        (0..self.layout.nchildren())
            .map(|idx| self.child(idx).cloned())
            .try_collect()
    }
}
