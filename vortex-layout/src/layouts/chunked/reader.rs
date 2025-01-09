use std::sync::{Arc, OnceLock};

use futures::future::{ready, LocalBoxFuture, Shared};
use futures::lock::{Mutex, MutexGuard};
use futures::FutureExt;
use vortex_array::stats::{stats_from_bitset_bytes, Stat};
use vortex_array::ContextRef;
use vortex_error::{vortex_panic, VortexError, VortexResult};
use vortex_expr::Identity;
use vortex_scan::RowMask;

use crate::layouts::chunked::stats_table::StatsTable;
use crate::layouts::chunked::ChunkedLayout;
use crate::reader::LayoutReader;
use crate::segments::AsyncSegmentReader;
use crate::{LayoutData, LayoutEncoding};

#[derive(Clone)]
pub struct ChunkedReader {
    layout: LayoutData,
    ctx: ContextRef,
    segments: Arc<dyn AsyncSegmentReader>,
    /// Shared stats table operation and cache of the result
    /// So we shouldn't store the future. We should instead let the first caller to ask
    /// for the stats table construct their own future, provided they promise to store the result
    /// into the cache. Perhaps using <https://docs.rs/async-once-cell/latest/async_once_cell/struct.OnceCell.html>
    stats_table_fut: Arc<Mutex<StatsTableFut>>,
    /// Shared lazy chunk scanners
    chunk_readers: Arc<[OnceLock<Arc<dyn LayoutReader>>]>,
}

type StatsTableFut = Shared<LocalBoxFuture<'static, Option<StatsTable>>>;

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

        // Figure out which stats are present
        let present_stats: Arc<[Stat]> = layout
            .metadata()
            .map(|m| stats_from_bitset_bytes(m.as_ref()))
            .unwrap_or_default()
            .into();

        let stats_table_fut = layout
            .metadata()
            .is_some()
            .then(|| {
                let layout_dtype = layout.dtype().clone();
                let stats_dtype = StatsTable::dtype_for_stats_table(&layout_dtype, &present_stats);
                let stats_layout = layout.child(layout.nchildren() - 1, stats_dtype)?;
                let reader = stats_layout.reader(segments.clone(), ctx.clone())?;

                Ok::<_, VortexError>(
                    async move {
                        reader
                            .evaluate(
                                RowMask::new_valid_between(0, nchunks as u64),
                                Identity::new_expr(),
                            )
                            .map(move |stats_array| {
                                stats_array
                                    .and_then(|stats_array| {
                                        StatsTable::try_new(
                                            layout_dtype.clone(),
                                            stats_array,
                                            present_stats.clone(),
                                        )
                                    })
                                    // FIXME(ngates): we should map_err into a cloneable error type
                                    .ok()
                            })
                            .await
                    }
                    .boxed_local(),
                )
            })
            .transpose()?
            .unwrap_or_else(|| ready(None).boxed())
            .shared();

        // Construct a lazy scan for each chunk of the layout.
        let chunk_scans = (0..nchunks).map(|_| OnceLock::new()).collect();

        Ok(Self {
            layout,
            ctx,
            segments,
            stats_table_fut: Arc::new(Mutex::new(stats_table_fut)),
            chunk_readers: chunk_scans,
        })
    }

    /// Get the stats table future.
    pub(crate) async fn stats_table_fut(&self) -> MutexGuard<StatsTableFut> {
        self.stats_table_fut.lock().await
    }

    /// Return the number of chunks
    pub(crate) fn nchunks(&self) -> usize {
        self.chunk_readers.len()
    }

    /// Return the child reader for the chunk.
    pub(crate) fn child(&self, idx: usize) -> VortexResult<Arc<dyn LayoutReader>> {
        self.chunk_readers[idx]
            .get_or_try_init(|| {
                let child_layout = self.layout.child(idx, self.layout.dtype().clone())?;
                child_layout.reader(self.segments.clone(), self.ctx.clone())
            })
            .cloned()
    }
}

impl LayoutReader for ChunkedReader {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }
}
