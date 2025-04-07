use std::ops::Range;
use std::sync::{Arc, OnceLock, RwLock};

use futures::future::{BoxFuture, Shared};
use futures::{FutureExt, TryFutureExt};
use vortex_array::ArrayContext;
use vortex_array::aliases::hash_map::{Entry, HashMap};
use vortex_array::stats::{Stat, stats_from_bitset_bytes};
use vortex_dtype::TryFromBytes;
use vortex_error::{SharedVortexResult, VortexExpect, VortexResult, vortex_panic};
use vortex_expr::pruning::PruningPredicate;
use vortex_expr::{ExprRef, Identity};
use vortex_mask::Mask;

use crate::layouts::stats::StatsLayout;
use crate::layouts::stats::stats_table::StatsTable;
use crate::reader::LayoutReader;
use crate::segments::SegmentSource;
use crate::{ExprEvaluator, Layout, LayoutVTable};

pub(crate) type SharedStatsTable = Shared<BoxFuture<'static, SharedVortexResult<StatsTable>>>;
pub(crate) type SharedPruningResult = Shared<BoxFuture<'static, SharedVortexResult<Option<Mask>>>>;
pub(crate) type PredicateCache = Arc<OnceLock<Option<PruningPredicate>>>;

pub struct StatsReader {
    layout: Layout,

    /// Data layout reader
    pub(crate) data_child: Arc<dyn LayoutReader>,
    /// Stats table layout reader.
    pub(crate) stats_child: Arc<dyn LayoutReader>,

    /// The number of zones
    pub(crate) nzones: usize,
    /// The number of rows in each zone (except possibly the last)
    pub(crate) zone_len: usize,
    /// The stats that are present in the table
    pub(crate) present_stats: Arc<[Stat]>,

    /// A cache of expr -> optional pruning result (applying the pruning expr to the stats table)
    pruning_result: RwLock<HashMap<ExprRef, Option<SharedPruningResult>>>,

    /// Shared stats table
    stats_table: OnceLock<SharedStatsTable>,

    /// A cache of expr -> optional pruning predicate.
    pub(crate) pruning_predicates: Arc<RwLock<HashMap<ExprRef, PredicateCache>>>,
}

impl StatsReader {
    pub(super) fn try_new(
        layout: Layout,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<Self> {
        if layout.vtable().id() != StatsLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        let metadata = layout
            .metadata()
            .vortex_expect("Stats layout must have metadata");

        let zone_len = u32::try_from_le_bytes(&metadata[0..4])? as usize;
        let present_stats: Arc<[Stat]> = stats_from_bitset_bytes(&metadata[4..]).into();
        let nzones = usize::try_from(layout.row_count().div_ceil(zone_len as u64))?;

        let data_child = layout
            .child(0, layout.dtype().clone(), "data")?
            .reader(segment_source, ctx)?;

        let stats_dtype = StatsTable::dtype_for_stats_table(layout.dtype(), &present_stats);
        let stats_child = layout
            .child(1, stats_dtype, "stats_table")?
            .reader(segment_source, ctx)?;

        Ok(Self {
            layout,
            data_child,
            stats_child,
            nzones,
            zone_len,
            present_stats,
            pruning_result: Default::default(),
            stats_table: Default::default(),
            pruning_predicates: Default::default(),
        })
    }

    /// Get or create the pruning predicate for a given expression.
    pub(crate) fn pruning_predicate(&self, expr: ExprRef) -> Option<PruningPredicate> {
        self.pruning_predicates
            .write()
            .vortex_expect("poisoned lock")
            .entry(expr.clone())
            .or_default()
            .get_or_init(move || PruningPredicate::try_new(&expr))
            .clone()
    }

    /// Get or initialize the stats table.
    ///
    /// Only the first successful caller will initialize the stats table, all other callers will
    /// resolve to the same result.
    pub(crate) fn stats_table(&self) -> SharedStatsTable {
        self.stats_table
            .get_or_init(move || {
                let nzones = self.nzones;
                let present_stats = self.present_stats.clone();

                let stats_eval = self
                    .stats_child
                    .projection_evaluation(&(0..nzones as u64), &Identity::new_expr())
                    .vortex_expect("Failed construct stats table evaluation");

                async move {
                    let stats_array = stats_eval.invoke(Mask::new_true(nzones)).await?;
                    // SAFETY: This is only fine to call because we perform validation above
                    Ok(StatsTable::unchecked_new(stats_array, present_stats))
                }
                .map_err(Arc::new)
                .boxed()
                .shared()
            })
            .clone()
    }

    /// Returns a pruning mask where `true` means the chunk _can be pruned_.
    pub(crate) fn pruning_mask_future(&self, expr: ExprRef) -> Option<SharedPruningResult> {
        match self
            .pruning_result
            .write()
            .vortex_expect("poisoned lock")
            .entry(expr.clone())
        {
            Entry::Occupied(e) => e.get().clone(),
            Entry::Vacant(e) => e
                .insert(match self.pruning_predicate(expr.clone()) {
                    None => {
                        log::debug!("No pruning predicate for expr: {}", expr);
                        None
                    }
                    Some(pred) => {
                        log::debug!("Constructed pruning predicate for expr: {}: {}", expr, pred);
                        Some(
                            self.stats_table()
                                .map(move |stats_table| {
                                    stats_table.and_then(move |stats_table| {
                                        pred.evaluate(stats_table.array())?
                                            .map(|a| Mask::try_from(a.as_ref()))
                                            .transpose()
                                            .map_err(Arc::new)
                                    })
                                })
                                .boxed()
                                .shared(),
                        )
                    }
                })
                .clone(),
        }
    }

    /// Return the zone range covered by a row range.
    pub(crate) fn zone_range(&self, row_range: &Range<u64>) -> Range<usize> {
        let zone_start = usize::try_from(row_range.start / self.zone_len as u64)
            .vortex_expect("Invalid zone start");
        let zone_end = usize::try_from(row_range.end.div_ceil(self.zone_len as u64))
            .vortex_expect("Invalid zone end");
        zone_start..zone_end
    }

    /// Return the row offset of a given zone.
    pub(crate) fn zone_offset(&self, zone_idx: usize) -> u64 {
        ((zone_idx * self.zone_len) as u64).min(self.layout.child_row_count(0))
    }
}

impl LayoutReader for StatsReader {
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn children(&self) -> VortexResult<Vec<Arc<dyn LayoutReader>>> {
        Ok(vec![self.data_child.clone(), self.stats_child.clone()])
    }
}
