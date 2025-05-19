use std::ops::{BitAnd, Deref, Range, Sub};
use std::sync::{Arc, OnceLock};

use arrow_buffer::BooleanBufferBuilder;
use async_trait::async_trait;
use futures::future::{BoxFuture, Shared};
use futures::{FutureExt, TryFutureExt};
use itertools::Itertools;
use parking_lot::RwLock;
use vortex_array::aliases::hash_map::{Entry, HashMap};
use vortex_array::{ArrayContext, ToCanonical};
use vortex_error::{SharedVortexResult, VortexError, VortexExpect, VortexResult};
use vortex_expr::pruning::PruningPredicate;
use vortex_expr::{ExprRef, Identity};
use vortex_mask::Mask;

use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::zone_map::ZoneMap;
use crate::segments::SegmentSource;
use crate::{ArrayEvaluation, Layout, LayoutReader, MaskEvaluation, PruningEvaluation};

pub(crate) type SharedZoneMap = Shared<BoxFuture<'static, SharedVortexResult<ZoneMap>>>;
pub(crate) type SharedPruningResult = Shared<BoxFuture<'static, SharedVortexResult<Option<Mask>>>>;
pub(crate) type PredicateCache = Arc<OnceLock<Option<PruningPredicate>>>;

pub struct ZonedReader {
    layout: ZonedLayout,
    name: Arc<str>,

    /// Data layout reader
    data_child: Arc<dyn LayoutReader>,
    /// Zone map layout reader.
    zones_child: Arc<dyn LayoutReader>,

    /// A cache of expr -> optional pruning result (applying the pruning expr to the stats table)
    pruning_result: RwLock<HashMap<ExprRef, Option<SharedPruningResult>>>,

    /// Shared zone map
    zone_map: OnceLock<SharedZoneMap>,

    /// A cache of expr -> optional pruning predicate.
    pruning_predicates: Arc<RwLock<HashMap<ExprRef, PredicateCache>>>,
}

impl Deref for ZonedReader {
    type Target = dyn Layout;

    fn deref(&self) -> &Self::Target {
        self.layout.deref()
    }
}

impl ZonedReader {
    pub(super) fn try_new(
        layout: ZonedLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        ctx: ArrayContext,
    ) -> VortexResult<Self> {
        let data_child = layout.data.new_reader(&name, &segment_source, &ctx)?;
        let zones_child =
            layout
                .zones
                .new_reader(&format!("{}.zones", name).into(), &segment_source, &ctx)?;

        Ok(Self {
            layout,
            name,
            data_child,
            zones_child,
            pruning_result: Default::default(),
            zone_map: Default::default(),
            pruning_predicates: Default::default(),
        })
    }

    /// Get or create the pruning predicate for a given expression.
    fn pruning_predicate(&self, expr: ExprRef) -> Option<PruningPredicate> {
        self.pruning_predicates
            .write()
            .entry(expr.clone())
            .or_default()
            .get_or_init(move || PruningPredicate::try_new(&expr))
            .clone()
    }

    /// Get or initialize the stats table.
    ///
    /// Only the first successful caller will initialize the stats table, all other callers will
    /// resolve to the same result.
    fn stats_table(&self) -> SharedZoneMap {
        self.zone_map
            .get_or_init(move || {
                let nzones = self.layout.nzones();
                let present_stats = self.layout.present_stats.clone();

                let zones_eval = self
                    .zones_child
                    .projection_evaluation(&(0..nzones as u64), &Identity::new_expr())
                    .vortex_expect("Failed construct stats table evaluation");

                async move {
                    let zones_array = zones_eval
                        .invoke(Mask::new_true(nzones))
                        .await?
                        .to_struct()?;
                    // SAFETY: This is only fine to call because we perform validation above
                    Ok(ZoneMap::unchecked_new(zones_array, present_stats))
                }
                .map_err(Arc::new)
                .boxed()
                .shared()
            })
            .clone()
    }

    /// Returns a pruning mask where `true` means the chunk _can be pruned_.
    fn pruning_mask_future(&self, expr: ExprRef) -> Option<SharedPruningResult> {
        match self.pruning_result.write().entry(expr.clone()) {
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
                                        pred.evaluate(stats_table.array().as_ref())?
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
        let zone_start = usize::try_from(row_range.start / self.layout.zone_len as u64)
            .vortex_expect("Invalid zone start");
        let zone_end = usize::try_from(row_range.end.div_ceil(self.layout.zone_len as u64))
            .vortex_expect("Invalid zone end");
        zone_start..zone_end
    }

    /// Return the row offset of a given zone.
    pub(crate) fn zone_offset(&self, zone_idx: usize) -> u64 {
        (zone_idx as u64 * self.layout.zone_len as u64).min(self.data_child.row_count())
    }
}

impl LayoutReader for ZonedReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        log::debug!("Stats pruning evaluation: {} - {}", &self.name, expr);
        let data_eval = self.data_child.pruning_evaluation(row_range, expr)?;

        let Some(pruning_mask_future) = self.pruning_mask_future(expr.clone()) else {
            log::debug!("Stats pruning evaluation: not prune-able {}", expr);
            return Ok(data_eval);
        };

        let zone_range = self.zone_range(row_range);
        let zone_lengths = zone_range
            .clone()
            .map(|zone_idx| {
                // Figure out the range in the mask that corresponds to the zone
                let start =
                    usize::try_from(self.zone_offset(zone_idx).saturating_sub(row_range.start))?;
                let end = usize::try_from(
                    self.zone_offset(zone_idx + 1)
                        .sub(row_range.start)
                        .min(row_range.end - row_range.start),
                )?;
                Ok::<_, VortexError>(end - start)
            })
            .try_collect()?;

        Ok(Box::new(StatsPruningEvaluation {
            name: self.name.clone(),
            expr: expr.clone(),
            pruning_mask_future,
            zone_range,
            zone_lengths,
            data_eval,
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        self.data_child.filter_evaluation(row_range, expr)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        // TODO(ngates): there are some projection expressions that we may also be able to
        //  short-circuit with statistics.
        self.data_child.projection_evaluation(row_range, expr)
    }
}

struct StatsPruningEvaluation {
    name: Arc<str>,
    expr: ExprRef,
    pruning_mask_future: SharedPruningResult,
    // The range of zones that cover the evaluation's row range.
    zone_range: Range<usize>,
    // The lengths of each zone in the zone_range.
    zone_lengths: Vec<usize>,
    // The evaluation of the data child.
    data_eval: Box<dyn PruningEvaluation>,
}

#[async_trait]
impl PruningEvaluation for StatsPruningEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        log::debug!(
            "Invoking stats pruning evaluation {}: {}",
            self.name,
            self.expr,
        );
        let Some(pruning_mask) = self.pruning_mask_future.clone().await? else {
            // If the expression is not prune-able, we just return the input mask.
            return Ok(mask);
        };

        let mut builder = BooleanBufferBuilder::new(mask.len());
        for (zone_idx, zone_length) in self.zone_range.clone().zip_eq(&self.zone_lengths) {
            builder.append_n(*zone_length, !pruning_mask.value(zone_idx));
        }

        let stats_mask = Mask::from(builder.finish());
        assert_eq!(stats_mask.len(), mask.len(), "Mask length mismatch");

        // Intersect the masks.
        let mut stats_mask = mask.bitand(&stats_mask);

        // Forward to data child for further pruning.
        if !stats_mask.all_false() {
            let data_mask = self.data_eval.invoke(stats_mask.clone()).await?;
            stats_mask = stats_mask.bitand(&data_mask);
        }

        log::debug!(
            "Stats evaluation approx {} - {} (mask = {}) => {}",
            self.name,
            self.expr,
            mask.density(),
            stats_mask.density(),
        );

        Ok(stats_mask)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arcref::ArcRef;
    use futures::executor::block_on;
    use rstest::{fixture, rstest};
    use vortex_array::{ArrayContext, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, PType};
    use vortex_expr::{Identity, gt, lit};
    use vortex_mask::Mask;

    use crate::LayoutRef;
    use crate::layouts::chunked::writer::ChunkedLayoutWriter;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::zoned::writer::{ZonedLayoutOptions, ZonedLayoutWriter};
    use crate::segments::{SegmentSource, TestSegments};
    use crate::writer::LayoutWriterExt;

    #[fixture]
    /// Create a stats layout with three chunks of primitive arrays.
    fn stats_layout() -> (ArrayContext, Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();
        let mut segments = TestSegments::default();
        let layout = ZonedLayoutWriter::new(
            ctx.clone(),
            &DType::Primitive(PType::I32, NonNullable),
            ChunkedLayoutWriter::new(
                ctx.clone(),
                DType::Primitive(PType::I32, NonNullable),
                Default::default(),
            )
            .boxed(),
            ArcRef::new_arc(Arc::new(FlatLayoutStrategy::default())),
            ZonedLayoutOptions {
                block_size: 3,
                ..Default::default()
            },
        )
        .push_all(
            &mut segments,
            [
                Ok(buffer![1, 2, 3].into_array()),
                Ok(buffer![4, 5, 6].into_array()),
                Ok(buffer![7, 8, 9].into_array()),
            ],
        )
        .unwrap();
        (ctx, Arc::new(segments), layout)
    }

    #[rstest]
    fn test_stats_evaluator(
        #[from(stats_layout)] (ctx, segments, layout): (
            ArrayContext,
            Arc<dyn SegmentSource>,
            LayoutRef,
        ),
    ) {
        block_on(async {
            let result = layout
                .new_reader(&"".into(), &segments, &ctx)
                .unwrap()
                .projection_evaluation(&(0..layout.row_count()), &Identity::new_expr())
                .unwrap()
                .invoke(Mask::new_true(layout.row_count().try_into().unwrap()))
                .await
                .unwrap()
                .to_primitive()
                .unwrap();

            assert_eq!(result.len(), 9);
            assert_eq!(result.as_slice::<i32>(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        })
    }

    #[rstest]
    fn test_stats_pruning_mask(
        #[from(stats_layout)] (ctx, segments, layout): (
            ArrayContext,
            Arc<dyn SegmentSource>,
            LayoutRef,
        ),
    ) {
        block_on(async {
            let row_count = layout.row_count();
            let reader = layout.new_reader(&"".into(), &segments, &ctx).unwrap();

            // Choose a prune-able expression
            let expr = gt(Identity::new_expr(), lit(7));

            let result = reader
                .pruning_evaluation(&(0..row_count), &expr)
                .unwrap()
                .invoke(Mask::new_true(row_count.try_into().unwrap()))
                .await
                .unwrap()
                .to_boolean_buffer()
                .iter()
                .collect::<Vec<_>>();

            assert_eq!(
                result.as_slice(),
                &[false, false, false, false, false, false, true, true, true]
            );
        })
    }
}
