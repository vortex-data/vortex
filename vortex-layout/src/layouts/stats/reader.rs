use std::ops::Range;
use std::sync::{Arc, RwLock};

use async_once_cell::OnceCell;
use vortex_array::ArrayContext;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::stats::{Stat, stats_from_bitset_bytes};
use vortex_dtype::TryFromBytes;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err, vortex_panic};
use vortex_expr::pruning::PruningPredicate;
use vortex_expr::{ExprRef, Identity};
use vortex_mask::Mask;

use crate::layouts::stats::StatsLayout;
use crate::layouts::stats::stats_table::StatsTable;
use crate::reader::LayoutReader;
use crate::segments::AsyncSegmentReader;
use crate::{ExprEvaluator, Layout, LayoutVTable, RowMask};

type PruningCache = Arc<OnceCell<Option<Mask>>>;

#[derive(Clone)]
pub struct StatsReader {
    layout: Layout,
    ctx: ArrayContext,
    segment_reader: Arc<dyn AsyncSegmentReader>,

    /// The number of blocks
    nblocks: usize,
    /// The size of each block (except possibly the last)
    block_len: usize,
    /// The stats present in the layout
    present_stats: Arc<[Stat]>,
    /// A cache of expr -> optional pruning result (applying the pruning expr to the stats table)
    pruning_result: Arc<RwLock<HashMap<ExprRef, PruningCache>>>,
    /// Shared stats table
    stats_table: Arc<OnceCell<StatsTable>>,
    /// Child layout reader
    child: Arc<dyn LayoutReader>,
}

impl StatsReader {
    pub(super) fn try_new(
        layout: Layout,
        ctx: ArrayContext,
        segment_reader: Arc<dyn AsyncSegmentReader>,
    ) -> VortexResult<Self> {
        if layout.vtable().id() != StatsLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        let child = layout
            .child(0, layout.dtype().clone(), "data")?
            .reader(segment_reader.clone(), ctx.clone())?;

        let metadata = layout
            .metadata()
            .vortex_expect("Stats layout must have metadata");

        let block_len = u32::try_from_le_bytes(&metadata[0..4])? as usize;
        let present_stats = stats_from_bitset_bytes(&metadata[4..]).into();
        let nblocks = usize::try_from(layout.row_count().div_ceil(block_len as u64))?;

        Ok(Self {
            layout,
            ctx,
            segment_reader,
            nblocks,
            block_len,
            present_stats,
            pruning_result: Arc::new(RwLock::new(HashMap::new())),
            stats_table: Arc::new(OnceCell::new()),
            child,
        })
    }

    /// Get or initialize the stats table.
    ///
    /// Only the first successful caller will initialize the stats table, all other callers will
    /// resolve to the same result.
    pub(crate) async fn stats_table(&self) -> VortexResult<&StatsTable> {
        self.stats_table
            .get_or_try_init(async {
                let layout_dtype = self.layout.dtype();
                let stats_dtype =
                    StatsTable::dtype_for_stats_table(layout_dtype, &self.present_stats);
                let stats_layout = self.layout.child(1, stats_dtype.clone(), "stats_table")?;

                let stats_array = stats_layout
                    .reader(self.segment_reader.clone(), self.ctx.clone())?
                    .evaluate_expr(
                        RowMask::new_valid_between(0, self.nblocks as u64),
                        Identity::new_expr(),
                    )
                    .await?;

                if &stats_dtype != stats_array.dtype() {
                    vortex_bail!(
                        "Expected stats DType {stats_dtype} doesn't match read stats dtype {}",
                        stats_array.dtype()
                    )
                }

                // SAFETY: This is only fine to call because we perform validation above
                Ok(StatsTable::unchecked_new(
                    stats_array,
                    self.present_stats.clone(),
                ))
            })
            .await
    }

    /// Returns a pruning mask where `true` means the chunk _can be pruned_.
    pub(crate) async fn pruning_mask(&self, expr: &ExprRef) -> VortexResult<Option<Mask>> {
        let cell = self
            .pruning_result
            .write()
            .map_err(|_| vortex_err!("poisoned lock"))?
            .entry(expr.clone())
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();

        cell.get_or_try_init(async {
            let pruning_predicate = PruningPredicate::try_new(expr);
            if let Some(p) = &pruning_predicate {
                log::debug!("Constructed pruning predicate for expr: {}: {}", expr, p);
            }

            let stats_table = self.stats_table().await?;
            Ok(if let Some(predicate) = pruning_predicate {
                predicate
                    .evaluate(stats_table.array())?
                    .map(|a| Mask::try_from(a.as_ref()))
                    .transpose()?
            } else {
                None
            })
        })
        .await
        .cloned()
    }

    /// Return the child reader for the chunk.
    pub(crate) fn child(&self) -> &Arc<dyn LayoutReader> {
        &self.child
    }

    /// Return the block range covered by a row mask.
    pub(crate) fn block_range(&self, row_mask: &RowMask) -> Range<usize> {
        let block_start = usize::try_from(row_mask.begin() / self.block_len as u64)
            .vortex_expect("Invalid block start");
        let block_end = usize::try_from(row_mask.end().div_ceil(self.block_len as u64))
            .vortex_expect("Invalid block end");
        block_start..block_end
    }

    /// Return the row offset of a given block.
    pub(crate) fn block_offset(&self, block_idx: usize) -> u64 {
        ((block_idx * self.block_len) as u64).min(self.layout.child_row_count(0))
    }
}

impl LayoutReader for StatsReader {
    fn layout(&self) -> &Layout {
        &self.layout
    }
}
