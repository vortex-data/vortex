use std::sync::{Arc, OnceLock, RwLock, RwLockWriteGuard};

use vortex_array::stats::{stats_from_bitset_bytes, Stat};
use vortex_array::ContextRef;
use vortex_error::{vortex_err, vortex_panic, VortexError, VortexResult};
use vortex_expr::{ExprRef, Identity};

use crate::layouts::chunked::evaluator::ChunkedEvaluator;
use crate::layouts::chunked::stats_table::StatsTable;
use crate::layouts::chunked::ChunkedLayout;
use crate::operations::cached::CachedOperation;
use crate::operations::{resolved, Operation, OperationExt};
use crate::reader::{EvalOp, LayoutReader};
use crate::{LayoutData, LayoutEncoding, RowMask};

pub struct ChunkedReader {
    layout: LayoutData,
    ctx: ContextRef,
    /// Shared stats table operation and cache of the result
    stats_table_op: RwLock<StatsTableOp>,
    /// Shared lazy chunk scanners
    // TODO(ngates): consider an LRU cache here so we don't indefinitely hold onto chunk readers.
    //  If we do this, then we could also cache ArrayData in a FlatLayout since we know that this
    //  cache will eventually be evicted.
    chunk_readers: Vec<OnceLock<Arc<dyn LayoutReader>>>,
}

type StatsTableOp = CachedOperation<Box<dyn Operation<Output = Option<StatsTable>>>>;

impl ChunkedReader {
    pub(super) fn try_new(layout: LayoutData, ctx: ContextRef) -> VortexResult<Self> {
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

        let stats_table_op = layout
            .metadata()
            .is_some()
            .then(|| {
                let column_dtype = layout.dtype().clone();
                let stats_dtype = StatsTable::dtype_for_stats_table(&column_dtype, &present_stats);
                let stats_layout = layout.child(layout.nchildren() - 1, stats_dtype)?;
                let op = stats_layout
                    .reader(ctx.clone())?
                    .create_evaluator(
                        RowMask::new_valid_between(0, nchunks as u64),
                        Identity::new_expr(),
                    )?
                    .map(move |stats_array| {
                        StatsTable::try_new(
                            column_dtype.clone(),
                            stats_array,
                            present_stats.clone(),
                        )
                        .map(Some)
                    })
                    .boxed();
                Ok::<_, VortexError>(op)
            })
            .transpose()?
            .unwrap_or_else(|| resolved(None).boxed())
            .cached();

        // Construct a lazy scan for each chunk of the layout.
        let chunk_scans = (0..nchunks).map(|_| OnceLock::new()).collect();

        Ok(Self {
            layout,
            ctx,
            stats_table_op: RwLock::new(stats_table_op),
            chunk_readers: chunk_scans,
        })
    }

    /// Get the stats table operation.
    pub(crate) fn stats_table_op(&self) -> VortexResult<RwLockWriteGuard<StatsTableOp>> {
        self.stats_table_op
            .write()
            .map_err(|_| vortex_err!("poisoned"))
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
                child_layout.reader(self.ctx.clone())
            })
            .cloned()
    }
}

impl LayoutReader for ChunkedReader {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }

    fn create_evaluator(self: Arc<Self>, row_mask: RowMask, expr: ExprRef) -> VortexResult<EvalOp> {
        Ok(ChunkedEvaluator::new(self, row_mask, expr).boxed())
    }
}
