use std::iter;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use vortex_array::ArrayContext;
use vortex_error::{VortexResult, vortex_panic};

use crate::layouts::chunked::ChunkedLayout;
use crate::reader::LayoutReader;
use crate::segments::SegmentReader;
use crate::{Layout, LayoutVTable};

#[derive(Clone)]
pub struct ChunkedReader {
    layout: Layout,
    ctx: ArrayContext,
    segment_reader: Arc<dyn SegmentReader>,

    /// Shared lazy chunk scanners
    chunk_readers: Arc<[OnceLock<Arc<dyn LayoutReader>>]>,
    /// Row offset for each chunk
    chunk_offsets: Vec<u64>,
}

impl ChunkedReader {
    pub(super) fn try_new(
        layout: Layout,
        ctx: ArrayContext,
        segment_reader: Arc<dyn SegmentReader>,
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
            ctx,
            segment_reader,
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
            child_layout.reader(self.segment_reader.clone(), self.ctx.clone())
        })
    }

    pub(crate) fn chunk_offset(&self, idx: usize) -> u64 {
        self.chunk_offsets[idx]
    }

    pub(crate) fn chunk_range(&self, row_range: Range<u64>) -> Range<usize> {
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
}

impl LayoutReader for ChunkedReader {
    fn layout(&self) -> &Layout {
        &self.layout
    }
}
