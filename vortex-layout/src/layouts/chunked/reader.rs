use std::sync::{Arc, OnceLock, RwLock};

use arrow_buffer::BooleanBuffer;
use async_once_cell::OnceCell;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::stats::stats_from_bitset_bytes;
use vortex_array::{ContextRef, IntoArrayVariant};
use vortex_error::{vortex_err, vortex_panic, VortexResult};
use vortex_expr::pruning::PruningPredicate;
use vortex_expr::{ExprRef, Identity};
use vortex_scan::RowMask;

use crate::layouts::chunked::stats_table::StatsTable;
use crate::layouts::chunked::ChunkedLayout;
use crate::reader::LayoutReader;
use crate::segments::AsyncSegmentReader;
use crate::{ExprEvaluator, Layout, LayoutVTable};

type PruningCache = Arc<OnceCell<Option<BooleanBuffer>>>;

#[derive(Clone)]
pub struct ChunkedReader {
    layout: Layout,
    ctx: ContextRef,
    segments: Arc<dyn AsyncSegmentReader>,
    /// A cache of expr -> optional pruning result (applying the pruning expr to the stats table)
    pruning_result: Arc<RwLock<HashMap<ExprRef, PruningCache>>>,
    /// Shared stats table
    stats_table: Arc<OnceCell<Option<StatsTable>>>,
    /// Shared lazy chunk scanners
    chunk_readers: Arc<[OnceLock<Arc<dyn LayoutReader>>]>,
}

impl ChunkedReader {
    pub(super) fn try_new(
        layout: Layout,
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
            pruning_result: Arc::new(RwLock::new(HashMap::new())),
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
                Ok(match self.layout.metadata() {
                    None => None,
                    Some(metadata) => {
                        // The final child is the statistics table.
                        let nchunks = self.layout.nchildren() - 1;

                        // Figure out which stats are present
                        let present_stats = stats_from_bitset_bytes(metadata.as_ref());

                        let layout_dtype = self.layout.dtype();
                        let stats_dtype =
                            StatsTable::dtype_for_stats_table(layout_dtype, &present_stats);
                        let stats_layout = self.layout.child(nchunks, stats_dtype)?;

                        let stats_array = stats_layout
                            .reader(self.segments.clone(), self.ctx.clone())?
                            .evaluate_expr(
                                RowMask::new_valid_between(0, nchunks as u64),
                                Identity::new_expr(),
                            )
                            .await?;

                        Some(StatsTable::try_new_unchecked(
                            layout_dtype.clone(),
                            stats_array,
                            present_stats.into(),
                        ))
                    }
                })
            })
            .await
            .map(|opt| opt.as_ref())
    }

    pub(crate) async fn pruning_mask(&self, expr: &ExprRef) -> VortexResult<Option<BooleanBuffer>> {
        let cell = self
            .pruning_result
            .write()
            .map_err(|_| vortex_err!("poisoned lock"))?
            .entry(expr.clone())
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();

        cell.get_or_try_init(async {
            let pruning_predicate = PruningPredicate::try_new(expr);
            Ok(if let Some(stats_table) = self.stats_table().await? {
                if let Some(predicate) = pruning_predicate {
                    predicate
                        .evaluate(stats_table.array())?
                        .map(|mask| mask.into_bool())
                        .transpose()?
                        .map(|mask| mask.boolean_buffer())
                } else {
                    None
                }
            } else {
                None
            })
        })
        .await
        .cloned()
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
    fn layout(&self) -> &Layout {
        &self.layout
    }
}
