// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::{BitAnd, Range};
use std::sync::{Arc, OnceLock};

use arrow_buffer::BooleanBufferBuilder;
use async_trait::async_trait;
use futures::future::{BoxFuture, Shared};
use futures::{FutureExt, TryFutureExt};
use itertools::Itertools;
use parking_lot::RwLock;
use vortex_array::ToCanonical;
use vortex_array::stats::Precision;
use vortex_dtype::{DType, FieldMask, FieldPath, FieldPathSet};
use vortex_error::{SharedVortexResult, VortexError, VortexExpect, VortexResult};
use vortex_expr::dynamic::DynamicExprUpdates;
use vortex_expr::pruning::checked_pruning_expr;
use vortex_expr::{ExprRef, root};
use vortex_mask::Mask;
use vortex_utils::aliases::dash_map::DashMap;

use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::zone_map::ZoneMap;
use crate::segments::SegmentSource;
use crate::{ArrayEvaluation, LayoutReader, MaskEvaluation, MaskFuture, PruningEvaluation};

type SharedZoneMap = Shared<BoxFuture<'static, SharedVortexResult<ZoneMap>>>;
type SharedPruningResult = Shared<BoxFuture<'static, SharedVortexResult<Arc<PruningResult>>>>;
type PredicateCache = Arc<OnceLock<Option<ExprRef>>>;

pub struct ZonedReader {
    layout: ZonedLayout,
    name: Arc<str>,

    /// Data layout reader
    data_child: Arc<dyn LayoutReader>,
    /// Zone map layout reader.
    zones_child: Arc<dyn LayoutReader>,

    /// A cache of expr -> optional pruning result (applying the pruning expr to the zone map)
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

    /// Get or initialize the zone map.
    ///
    /// Only the first successful caller will initialize the zone map, all other callers will
    /// resolve to the same result.
    fn zone_map(&self) -> SharedZoneMap {
        self.zone_map
            .get_or_init(move || {
                let nzones = self.layout.nzones();
                let present_stats = self.layout.present_stats.clone();

                let zones_eval = self
                    .zones_child
                    .projection_evaluation(&(0..nzones as u64), &root())
                    .vortex_expect("Failed construct zone map evaluation");

                async move {
                    let zones_array = zones_eval
                        .invoke(MaskFuture::new_true(nzones))
                        .await?
                        .to_struct();
                    // SAFETY: This is only fine to call because we perform validation above
                    Ok(ZoneMap::new_unchecked(zones_array, present_stats))
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
                Some(predicate) => {
                    log::debug!("Constructed pruning predicate for expr: {expr}: {predicate:?}");
                    let zone_map = self.zone_map();
                    let dynamic_updates = DynamicExprUpdates::new(&expr);

                    Some(
                        async move {
                            let zone_map = zone_map.await?;
                            let initial_mask = zone_map.prune(&predicate)?;
                            Ok(Arc::new(PruningResult {
                                zone_map,
                                predicate,
                                dynamic_updates,
                                latest_result: RwLock::new((0, initial_mask)),
                            }))
                        }
                        .boxed()
                        .shared(),
                    )
                }
            })
            .clone()
    }

    /// Get the range of zone IDs containing a row range.
    pub(crate) fn zone_range(&self, row_range: &Range<u64>) -> Range<u64> {
        // Zone length is guaranteed to be > 0 by ZonedLayout::new validation
        debug_assert!(self.layout.zone_len > 0, "zone_len must be > 0");
        let zone_len_u64 = self.layout.zone_len as u64;
        let zone_start = row_range.start / zone_len_u64;
        let zone_end = row_range.end.div_ceil(zone_len_u64);
        zone_start..zone_end
    }

    /// Get the row index for the first row in a zone with the given `zone_index`.
    pub(crate) fn first_row_offset(&self, zone_idx: u64) -> u64 {
        zone_idx
            .saturating_mul(self.layout.zone_len as u64)
            .min(self.layout.row_count())
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

        let pruning_mask = self.pruning_mask_future.clone().await?.mask()?;

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

/// A wrapper for the result of pruning an expression against a zone map such that we can refresh
/// it each time the dynamic expressions are updated.
struct PruningResult {
    zone_map: ZoneMap,
    predicate: ExprRef,
    dynamic_updates: Option<DynamicExprUpdates>,
    latest_result: RwLock<(u64, Mask)>,
}

impl PruningResult {
    /// Return the pruning mask, computed for _at least_ the given version.
    ///
    /// The version typically comes from the dynamic expression updates, but zero can be passed
    /// to fetch any version.
    fn mask(&self) -> VortexResult<Mask> {
        // If we're not dynamic, then the result is always the latest result.
        let Some(dynamic_updates) = &self.dynamic_updates else {
            return Ok(self.latest_result.read().1.clone());
        };

        // Compute the latest version of the dynamic expression values.
        let version = dynamic_updates.version();

        {
            let read_guard = self.latest_result.read();
            if read_guard.0 >= version {
                // We're up to date, so we can return the cached result.
                return Ok(read_guard.1.clone());
            }
        }

        // Otherwise, we re-compute the mask for the given version number.
        let mut guard = self.latest_result.write();

        // Once we've taken the write lock, we check again in case another thread has already
        // beaten us to it.
        if guard.0 >= version {
            return Ok(guard.1.clone());
        }

        log::debug!(
            "Re-computing pruning mask for version {version} on {}",
            self.predicate
        );

        let next_mask = self.zone_map.prune(&self.predicate)?;
        *guard = (version, next_mask.clone());

        Ok(next_mask)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

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
    use crate::segments::{SegmentSource, SequenceWriter, TestSegments};
    use crate::{LayoutRef, LayoutStrategy, LocalExecutor, MaskFuture};

    #[fixture]
    /// Create a stats layout with three chunks of primitive arrays.
    fn stats_layout() -> (Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();
        let segments = TestSegments::default();
        let sequence_writer = SequenceWriter::new(Box::new(segments.clone()));
        let strategy = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
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
                .invoke(MaskFuture::new_true(layout.row_count().try_into().unwrap()))
                .await
                .unwrap()
                .to_primitive();

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
