use std::sync::{Arc, OnceLock};

use itertools::Itertools;
use vortex_array::ArrayContext;
use vortex_error::VortexResult;

use crate::LayoutData;
use crate::reader::LayoutReader;
use crate::segments::SegmentSource;

#[derive(Clone)]
pub struct ChunkedReader {
    pub(super) layout: LayoutData,
    pub(super) segment_source: Arc<dyn SegmentSource>,
    pub(super) ctx: ArrayContext,

    /// Shared lazy chunk scanners
    pub(super) chunk_readers: Arc<[OnceLock<Arc<dyn LayoutReader>>]>,
    /// Row offset for each chunk
    pub(super) chunk_offsets: Vec<u64>,
}

impl LayoutReader for ChunkedReader {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }

    fn children(&self) -> VortexResult<Vec<Arc<dyn LayoutReader>>> {
        (0..self.layout.nchildren())
            .map(|idx| self.child(idx).cloned())
            .try_collect()
    }
}
