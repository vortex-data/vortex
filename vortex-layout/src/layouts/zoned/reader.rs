// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::{BitAnd, Range};
use std::sync::{Arc, OnceLock};

use arrow_buffer::BooleanBufferBuilder;
use async_trait::async_trait;
use dashmap::DashMap;
use futures::future::{BoxFuture, Shared};
use futures::{FutureExt, TryFutureExt};
use itertools::Itertools;
use vortex_array::ToCanonical;
use vortex_array::stats::Precision;
use vortex_dtype::{DType, FieldMask, FieldPath, FieldPathSet};
use vortex_error::{SharedVortexResult, VortexError, VortexExpect, VortexResult};
use vortex_expr::pruning::checked_pruning_expr;
use vortex_expr::{ExprRef, Scope, root};
use vortex_mask::Mask;

use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::zone_map::ZoneMap;
use crate::segments::SegmentSource;
use crate::{ArrayEvaluation, LayoutReader, MaskEvaluation, PruningEvaluation};

pub(crate) type SharedZoneMap = Shared<BoxFuture<'static, SharedVortexResult<ZoneMap>>>;
pub(crate) type SharedPruningResult = Shared<BoxFuture<'static, SharedVortexResult<Option<Mask>>>>;
pub(crate) type PredicateCache = Arc<OnceLock<Option<ExprRef>>>;

pub struct ZonedReader {
    layout: ZonedLayout,
    name: Arc<str>,

    /// Data layout reader
    data_child: Arc<dyn LayoutReader>,
    /// Zone map layout reader.
    zones_child: Arc<dyn LayoutReader>,

    /// A cache of expr -> optional pruning result (applying the pruning expr to the stats table)
    pruning_result: DashMap<ExprRef, Option<SharedPruningResult>>,

    /// Shared zone map
    zone_map: OnceLock<SharedZoneMap>,

    /// A cache of expr -> optional pruning predicate.
    /// This also uses the present_stats from the `ZonedLayout`
    pruning_predicates: Arc<DashMap<ExprRef, PredicateCache>>,
}

impl ZonedReader {
    pub(super) fn try_new(
        layout: ZonedLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
    ) -> VortexResult<Self> {
        let data_child = layout
            .data
            .new_reader(name.clone(), segment_source.clone())?;
        let zones_child = layout
            .zones
            .new_reader(format!("{name}.zones").into(), segment_source)?;

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
    fn pruning_predicate(&self, expr: ExprRef) -> Option<ExprRef> {
        self.pruning_predicates
            .entry(expr.clone())
            .or_default()
            .get_or_init(move || {
                let field_path_set = FieldPathSet::from_iter(
                    self.layout
                        .present_stats
                        .iter()
                        .map(|s| FieldPath::from_name(s.name())),
                );
                checked_pruning_expr(&expr, &field_path_set).map(|(expr, _)| expr)
            })
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
                    .projection_evaluation(&(0..nzones as u64), &root())
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
        self.pruning_result
            .entry(expr.clone())
            .or_insert_with(|| match self.pruning_predicate(expr.clone()) {
                None => {
                    log::debug!("No pruning predicate for expr: {expr}");
                    None
                }
                Some(pred) => {
                    log::debug!("Constructed pruning predicate for expr: {expr}: {pred:?}");
                    Some(
                        self.stats_table()
                            .map(move |stats_table| {
                                stats_table.and_then(move |stats_table| {
                                    Mask::try_from(
                                        pred.evaluate(&Scope::new(stats_table.array().to_array()))?
                                            .as_ref(),
                                    )
                                    .map_err(Arc::new)
                                    .map(Some)
                                })
                            })
                            .boxed()
                            .shared(),
                    )
                }
            })
            .clone()
    }

    /// Get the range of zone IDs containing a row range.
    pub(crate) fn zone_range(&self, row_range: &Range<u64>) -> Range<u64> {
        let zone_start = row_range.start / self.layout.zone_len as u64;
        let zone_end = row_range.end.div_ceil(self.layout.zone_len as u64);
        zone_start..zone_end
    }

    /// Get the row index for the first row in a zone with the given `zone_index`.
    pub(crate) fn first_row_offset(&self, zone_idx: u64) -> u64 {
        (zone_idx * self.layout.zone_len as u64).min(self.layout.row_count())
    }
}

impl LayoutReader for ZonedReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.data_child.dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        self.data_child.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.data_child
            .register_splits(field_mask, row_offset, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        log::debug!("Stats pruning evaluation: {} - {}", &self.name, expr);
        let data_eval = self.data_child.pruning_evaluation(row_range, expr)?;

        let Some(pruning_mask_future) = self.pruning_mask_future(expr.clone()) else {
            log::debug!("Stats pruning evaluation: not prune-able {expr}");
            return Ok(data_eval);
        };

        let row_count = row_range.end - row_range.start;
        let zone_range = self.zone_range(row_range);
        let zone_lengths = zone_range
            .clone()
            .map(|zone_idx| {
                // Figure out the range in the mask that corresponds to the zone
                let start = usize::try_from(
                    self.first_row_offset(zone_idx)
                        .saturating_sub(row_range.start),
                )?;
                let end = usize::try_from(
                    self.first_row_offset(zone_idx + 1)
                        .saturating_sub(row_range.start)
                        .min(row_count),
                )?;
                Ok::<_, VortexError>(end - start)
            })
            .try_collect()?;

        Ok(Box::new(ZoneMapPruningEvaluation {
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

struct ZoneMapPruningEvaluation {
    name: Arc<str>,
    expr: ExprRef,
    /// A mask indicating zones which have no matching values.
    ///
    /// A false value indicates the corresponding zone may have a matching value.
    pruning_mask_future: SharedPruningResult,
    /// The set of zone IDs that are available to the evaluation.
    zone_range: Range<u64>,
    /// The lengths of each zone in the zone_range.
    zone_lengths: Vec<usize>,
    /// The evaluation of the data child.
    data_eval: Box<dyn PruningEvaluation>,
}

#[async_trait]
impl PruningEvaluation for ZoneMapPruningEvaluation {
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
        for (zone_idx, &zone_length) in self.zone_range.clone().zip_eq(&self.zone_lengths) {
            builder.append_n(zone_length, !pruning_mask.value(usize::try_from(zone_idx)?));
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
    use futures::stream;
    use rstest::{fixture, rstest};
    use vortex_array::stream::{ArrayStreamAdapter, ArrayStreamExt};
    use vortex_array::{ArrayContext, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, PType};
    use vortex_expr::{gt, lit, root};
    use vortex_mask::Mask;

    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::zoned::writer::{ZonedLayoutOptions, ZonedStrategy};
    use crate::scan::LocalExecutor;
    use crate::segments::{SegmentSource, SequenceWriter, TestSegments};
    use crate::{LayoutRef, LayoutStrategy};

    #[fixture]
    /// Create a stats layout with three chunks of primitive arrays.
    fn stats_layout() -> (Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();
        let segments = TestSegments::default();
        let sequence_writer = SequenceWriter::new(Box::new(segments.clone()));
        let strategy = ZonedStrategy::new(
            ArcRef::new_arc(Arc::new(ChunkedLayoutStrategy::default())),
            ArcRef::new_arc(Arc::new(FlatLayoutStrategy::default())),
            ZonedLayoutOptions {
                block_size: 3,
                ..Default::default()
            },
            Arc::new(LocalExecutor),
        );
        let array_stream =
            sequence_writer.new_sequential(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
                DType::Primitive(PType::I32, NonNullable),
                stream::iter([
                    Ok(buffer![1, 2, 3].into_array()),
                    Ok(buffer![4, 5, 6].into_array()),
                    Ok(buffer![7, 8, 9].into_array()),
                ]),
            )));
        let layout = block_on(strategy.write_stream(&ctx, sequence_writer, array_stream)).unwrap();
        (Arc::new(segments), layout)
    }

    #[rstest]
    fn test_stats_evaluator(
        #[from(stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        block_on(async {
            let result = layout
                .new_reader("".into(), segments)
                .unwrap()
                .projection_evaluation(&(0..layout.row_count()), &root())
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
        #[from(stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        block_on(async {
            let row_count = layout.row_count();
            let reader = layout.new_reader("".into(), segments).unwrap();

            // Choose a prune-able expression
            let expr = gt(root(), lit(7));

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
