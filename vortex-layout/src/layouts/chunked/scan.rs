use vortex_array::{ArrayData, ContextRef};
use vortex_dtype::DType;
use vortex_error::{vortex_err, vortex_panic, VortexResult};

use crate::layouts::chunked::ChunkedLayout;
use crate::scanner::{LayoutScan, Poll, Scan, Scanner};
use crate::segments::SegmentReader;
use crate::{LayoutData, LayoutEncoding, RowMask};

#[derive(Clone)]
struct Reading {
    // The index of the chunk currently being read.
    chunk_idx: usize,
    // The layout of the chunk currently being read.
    chunk_layout: LayoutData,
    // The statistics table, if required
    statistics: Option<ArrayData>,
}

pub struct ChunkedScan {
    layout: LayoutData,
    scan: Scan,
    dtype: DType,
    ctx: ContextRef,
}

impl ChunkedScan {
    pub(super) fn try_new(layout: LayoutData, scan: Scan, ctx: ContextRef) -> VortexResult<Self> {
        if layout.encoding().id() != ChunkedLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }
        let dtype = scan.result_dtype(layout.dtype())?;
        Ok(Self {
            layout,
            scan,
            dtype,
            ctx,
        })
    }
}

impl LayoutScan for ChunkedScan {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn create_scanner(&self, mask: RowMask) -> VortexResult<Box<dyn Scanner>> {
        Ok(Box::new(ChunkedScanner {
            layout: self.layout.clone(),
            mask,
            state: State::Initial,
        }) as _)
    }
}

#[derive(Clone)]
enum State {
    Initial,
    Reading(Reading),
}

struct ChunkedScanner {
    layout: LayoutData,
    mask: RowMask,
    state: State,
}

impl ChunkedScanner {
    /// Returns the [`LayoutData`] for the given chunk.
    fn chunk_layout(&self, chunk_idx: usize) -> VortexResult<LayoutData> {
        self.layout
            .child(chunk_idx, self.layout.dtype().clone())
            .ok_or_else(|| vortex_err!("Chunk index out of bounds"))
    }
}

impl Scanner for ChunkedScanner {
    fn poll(&mut self, _segments: &dyn SegmentReader) -> VortexResult<Poll> {
        loop {
            match self.state {
                State::Initial => {
                    // TODO(ngates): decide whether to read the stats table. We should read it if:
                    //  * The scan's filter expression exists and is prune-able,
                    //  * The scan's mask spans more than a single chunk

                    // We always start at chunk zero. The reading state will skip if there's
                    // no work based on the mask.
                    let chunk_idx = 0;
                    let chunk_layout = self.chunk_layout(chunk_idx)?;
                    self.state = State::Reading(Reading {
                        chunk_idx,
                        chunk_layout,
                        statistics: None,
                    });
                }
                State::Reading(Reading { .. }) => {
                    // self.mask.is_disjoint(self.chunk_ranges)
                    //     self.state = State::Reading(Reading {
                    //         chunk_idx: chunk_idx + 1,
                    //         chunk_layout: self.chunk_layout(chunk_idx + 1)?,
                    //         statistics: None,
                    //     });s
                    todo!()
                }
            }
        }
    }
}
