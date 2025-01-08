use std::sync::{Arc, OnceLock, RwLock, RwLockWriteGuard};

use arrow_buffer::BooleanBuffer;
use vortex_array::array::StructArray;
use vortex_array::stats::{stats_from_bitset_bytes, Stat};
use vortex_array::validity::Validity;
use vortex_array::{ContextRef, IntoArrayData};
use vortex_dtype::DType;
use vortex_error::{vortex_err, vortex_panic, VortexExpect, VortexResult};

use crate::layouts::chunked::eval::ChunkedEvalOp;
use crate::layouts::chunked::stats_table::StatsTable;
use crate::layouts::chunked::ChunkedLayout;
use crate::operations::cached::CachedOperation;
use crate::operations::{resolved, OperationExt};
use crate::reader::LayoutReader;
use crate::scanner::{EvalOp, Scan};
use crate::{LayoutData, LayoutEncoding, RowMask};

/// Captures the scan state of a chunked layout.
///
/// This scan is used to generate multiple scanners, one per row-range. It may even be the case
/// that the caller requests different row ranges for filter operations as for projection
/// operations. As such, it's beneficial for us to re-use some state across different scanners.
///
/// The obvious state to re-use is that of the statistics table. Each range scan polls the same
/// underlying statistics scanner since a scanner must continue to return its result for subsequent
/// polls.
///
/// There is a question about whether we should lazily construct and share all chunk scanners too.
/// We currently create a new chunk scanner for every chunk read by each range scan. This is sort
/// of fine if we assume row ranges are non-overlapping (although if a range overlaps a chunk
/// boundary we will read the chunk twice). However, if we have overlapping row ranges, as can
/// happen if the parent is performing multiple scans (filter + projection), then we may read the
/// same chunk many times.
pub struct ChunkedReader {
    layout: LayoutData,
    scan: Scan,
    dtype: DType,
    ctx: ContextRef,
    // Shared state table operation
    stats_table: RwLock<Box<dyn CachedOperation<Output = StatsTable>>>,
    // Cached pruning mask for the scan
    pruning_mask: OnceLock<Option<BooleanBuffer>>,
    // Shared lazy chunk scanners
    chunk_scans: Vec<OnceLock<Arc<dyn LayoutReader>>>,
    // The stats that are present in the layout
    present_stats: Arc<[Stat]>,
}

impl ChunkedReader {
    pub(super) fn try_new(layout: LayoutData, scan: Scan, ctx: ContextRef) -> VortexResult<Self> {
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

        let stats_table = layout
            .metadata()
            .is_some()
            .then(|| {
                let stats_dtype = StatsTable::dtype_for_stats_table(layout.dtype(), &present_stats);
                let stats_layout = layout.child(layout.nchildren() - 1, stats_dtype)?;
                stats_layout
                    .new_scan(Scan::all(), ctx.clone())?
                    .create_eval(RowMask::new_valid_between(0, nchunks as u64))
            })
            .transpose()?
            .map(|scanner| {
                scanner
                    .map(|stats_array| {
                        StatsTable::try_new(
                            layout.dtype().clone(),
                            stats_array,
                            present_stats.clone(),
                        )
                    })
                    .cached()
                    .cached_box()
            })
            .unwrap_or_else(|| {
                // Otherwise we create a default stats table with no columns.
                resolved(
                    StatsTable::try_new(
                        layout.dtype().clone(),
                        StructArray::try_new(vec![].into(), vec![], nchunks, Validity::NonNullable)
                            .vortex_expect("cannot fail")
                            .into_array(),
                        vec![].into(),
                    )
                    .vortex_expect("cannot fail"),
                )
                .cached()
                .cached_box()
            });

        // Construct a lazy scan for each chunk of the layout.
        let chunk_scans = (0..nchunks).map(|_| OnceLock::new()).collect();

        // Compute the dtype of the scan.
        let dtype = scan.result_dtype(layout.dtype())?;

        Ok(Self {
            layout,
            scan,
            dtype,
            ctx,
            stats_table: RwLock::new(stats_table),
            pruning_mask: OnceLock::new(),
            chunk_scans,
            present_stats,
        })
    }
}

impl ChunkedReader {
    /// Get the stats table operation.
    pub(crate) fn stats_table(
        &self,
    ) -> VortexResult<RwLockWriteGuard<Box<dyn CachedOperation<Output = StatsTable>>>> {
        self.stats_table
            .write()
            .map_err(|_| vortex_err!("poisoned"))
    }
}

impl LayoutReader for ChunkedReader {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn create_eval(self: Arc<Self>, mask: RowMask) -> VortexResult<EvalOp> {
        Ok(ChunkedEvalOp::new(self, mask).boxed())
    }
}
