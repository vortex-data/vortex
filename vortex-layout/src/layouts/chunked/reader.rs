use std::sync::{Arc, OnceLock};

use async_once_cell::OnceCell;
use vortex_array::stats::{stats_from_bitset_bytes, Stat};
use vortex_array::ContextRef;
use vortex_error::{vortex_panic, VortexResult};
use vortex_expr::Identity;
use vortex_scan::RowMask;

use crate::layouts::chunked::stats_table::StatsTable;
use crate::layouts::chunked::ChunkedLayout;
use crate::reader::LayoutReader;
use crate::segments::AsyncSegmentReader;
use crate::{ExprEvaluator, LayoutData, LayoutEncoding};

#[derive(Clone)]
pub struct ChunkedReader {
    layout: LayoutData,
    ctx: ContextRef,
    segments: Arc<dyn AsyncSegmentReader>,
    /// Shared stats table
    stats_table: Arc<OnceCell<Option<StatsTable>>>,
    /// Shared lazy chunk scanners
    chunk_readers: Arc<[OnceLock<Arc<dyn LayoutReader>>]>,
}

impl ChunkedReader {
    pub(super) fn try_new(
        layout: LayoutData,
        ctx: ContextRef,
        segments: Arc<dyn AsyncSegmentReader>,
    ) -> VortexResult<Self> {
        if layout.encoding().id() != ChunkedLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        // The number of chunks
        let mut nchunks = layout.nchildren();
        if layout.metadata().is_some() {
            // The final child is the statistics table.
            nchunks -= 1;
        }

        // Construct a lazy scan for each chunk of the layout.
        let chunk_readers = (0..nchunks).map(|_| OnceLock::new()).collect();

        Ok(Self {
            layout,
            ctx,
            segments,
            stats_table: Arc::new(OnceCell::new()),
            chunk_readers,
        })
    }

    /// Get or initialize the stats table.
    ///
    /// Only the first successful caller will initialize the stats table, all other callers will
    /// resolve to the same result.
    pub(crate) async fn stats_table(&self) -> VortexResult<Option<&StatsTable>> {
        self.stats_table
            .get_or_try_init(async {
                // The number of chunks
                let mut nchunks = self.layout.nchildren();
                if self.layout.metadata().is_some() {
                    // The final child is the statistics table.
                    nchunks -= 1;
                }

                Ok(match self.layout.metadata() {
                    None => None,
                    Some(metadata) => {
                        // Figure out which stats are present
                        let present_stats: Arc<[Stat]> =
                            Arc::from(stats_from_bitset_bytes(metadata.as_ref()));

                        let layout_dtype = self.layout.dtype().clone();
                        let stats_dtype =
                            StatsTable::dtype_for_stats_table(&layout_dtype, &present_stats);
                        let stats_layout = self
                            .layout
                            .child(self.layout.nchildren() - 1, stats_dtype)?;

                        let stats_array = stats_layout
                            .reader(self.segments.clone(), self.ctx.clone())?
                            .evaluate_expr(
                                RowMask::new_valid_between(0, nchunks as u64),
                                Identity::new_expr(),
                            )
                            .await?;

                        Some(StatsTable::try_new(
                            layout_dtype.clone(),
                            stats_array,
                            present_stats.clone(),
                        )?)
                    }
                })
            })
            .await
            .map(|opt| opt.as_ref())
    }

    /// Return the number of chunks
    pub(crate) fn nchunks(&self) -> usize {
        self.chunk_readers.len()
    }

    /// Return the child reader for the chunk.
    pub(crate) fn child(&self, idx: usize) -> VortexResult<&Arc<dyn LayoutReader>> {
        self.chunk_readers[idx].get_or_try_init(|| {
            let child_layout = self.layout.child(idx, self.layout.dtype().clone())?;
            child_layout.reader(self.segments.clone(), self.ctx.clone())
        })
    }
}

impl LayoutReader for ChunkedReader {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }
}
