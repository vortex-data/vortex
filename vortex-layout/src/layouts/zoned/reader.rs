// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::OnceLock;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use itertools::Itertools;
use parking_lot::RwLock;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::ToCanonical;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::FieldPathSet;
use vortex_array::expr::Expression;
use vortex_array::expr::pruning::checked_pruning_expr;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_buffer::BitBufferMut;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::LazyReaderChildren;
use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::zone_map::ZoneMap;
use crate::segments::SegmentSource;

type SharedZoneMap = Shared<BoxFuture<'static, SharedVortexResult<ZoneMap>>>;
type SharedPruningResult = Shared<BoxFuture<'static, SharedVortexResult<Arc<PruningResult>>>>;
type PredicateCache = Arc<OnceLock<Option<Expression>>>;

pub struct ZonedReader {
    layout: ZonedLayout,
    name: Arc<str>,
    lazy_children: LazyReaderChildren,
    session: VortexSession,

    /// A cache of expr -> optional pruning result (applying the pruning expr to the zone map)
    pruning_result: LazyLock<DashMap<Expression, Option<SharedPruningResult>>>,

    /// Shared zone map
    zone_map: OnceLock<SharedZoneMap>,

    /// A cache of expr -> optional pruning predicate.
    /// This also uses the present_stats from the `ZonedLayout`
    pruning_predicates: LazyLock<Arc<DashMap<Expression, PredicateCache>>>,
}

impl ZonedReader {
    pub(super) fn try_new(
        layout: ZonedLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> VortexResult<Self> {
        let dtypes = vec![
            layout.dtype.clone(),
            ZoneMap::dtype_for_stats_table(layout.dtype(), layout.present_stats()),
        ];
        let names = vec![name.clone(), format!("{}.zones", name).into()];
        let lazy_children = LazyReaderChildren::new(
            layout.children.clone(),
            dtypes,
            names,
            segment_source.clone(),
            session.clone(),
        );

        Ok(Self {
            layout,
            name,
            lazy_children,
            session,
            pruning_result: Default::default(),
            zone_map: Default::default(),
            pruning_predicates: Default::default(),
        })
    }

    fn data_child(&self) -> VortexResult<&LayoutReaderRef> {
        self.lazy_children.get(0)
    }

    /// Get or create the pruning predicate for a given expression.
    fn pruning_predicate(&self, expr: Expression) -> Option<Expression> {
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
                    .lazy_children
                    .get(1)
                    .vortex_expect("failed to get zone child")
                    .projection_evaluation(
                        &(0..nzones as u64),
                        &root(),
                        MaskFuture::new_true(nzones),
                    )
                    .vortex_expect("Failed construct zone map evaluation");

                async move {
                    let zones_array = zones_eval.await?.to_struct();
                    // SAFETY: This is only fine to call because we perform validation above
                    Ok(unsafe { ZoneMap::new_unchecked(zones_array, present_stats) })
                }
                .map_err(Arc::new)
                .boxed()
                .shared()
            })
            .clone()
    }

    /// Returns a pruning mask where `true` means the chunk _can be pruned_.
    fn pruning_mask_future(&self, expr: Expression) -> Option<SharedPruningResult> {
        // Check cache first with read-only lock
        if let Some(result) = self.pruning_result.get(&expr) {
            return result.value().clone();
        }

        self.pruning_result
            .entry(expr.clone())
            .or_insert_with(|| match self.pruning_predicate(expr.clone()) {
                None => {
                    tracing::debug!("No pruning predicate for expr: {expr}");
                    None
                }
                Some(predicate) => {
                    tracing::debug!(
                        "Constructed pruning predicate for expr: {expr}: {predicate:?}"
                    );
                    let zone_map = self.zone_map();
                    let dynamic_updates = DynamicExprUpdates::new(&expr);
                    let session = self.session.clone();

                    Some(
                        async move {
                            let zone_map = zone_map.await?;
                            let initial_mask =
                                zone_map.prune(&predicate, &session).map_err(|err| {
                                    err.with_context(format!(
                                        "While evaluating pruning predicate {} (derived from {})",
                                        predicate, expr
                                    ))
                                })?;
                            Ok(Arc::new(PruningResult {
                                zone_map,
                                predicate,
                                dynamic_updates,
                                latest_result: RwLock::new((0, initial_mask)),
                                session,
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
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.data_child()?
            .register_splits(field_mask, row_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        tracing::debug!("Stats pruning evaluation: {} - {}", &self.name, expr);
        let data_eval = self
            .data_child()?
            .pruning_evaluation(row_range, expr, mask.clone())?;

        let Some(pruning_mask_future) = self.pruning_mask_future(expr.clone()) else {
            tracing::debug!("Stats pruning evaluation: not prune-able {expr}");
            return Ok(data_eval);
        };

        let row_count = row_range.end - row_range.start;
        let zone_range = self.zone_range(row_range);
        let zone_lengths: Vec<_> = zone_range
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

        let name = self.name.clone();
        let expr = expr.clone();

        Ok(MaskFuture::new(mask.len(), async move {
            tracing::debug!("Invoking stats pruning evaluation {}: {}", name, expr);

            let pruning_mask = pruning_mask_future.await?.mask()?;

            let mut builder = BitBufferMut::with_capacity(mask.len());
            for (zone_idx, &zone_length) in zone_range.clone().zip_eq(&zone_lengths) {
                builder.append_n(!pruning_mask.value(usize::try_from(zone_idx)?), zone_length);
            }

            let stats_mask = Mask::from(builder.freeze());
            assert_eq!(stats_mask.len(), mask.len(), "Mask length mismatch");

            // Intersect the masks.
            let mut stats_mask = mask.bitand(&stats_mask);

            // Forward to data child for further pruning.
            if !stats_mask.all_false() {
                let data_mask = data_eval.await?;
                stats_mask = stats_mask.bitand(&data_mask);
            }

            tracing::debug!(
                "Stats evaluation approx {} - {} (mask = {}) => {}",
                name,
                expr,
                mask.density(),
                stats_mask.density(),
            );

            Ok(stats_mask)
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        self.data_child()?.filter_evaluation(row_range, expr, mask)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        // TODO(ngates): there are some projection expressions that we may also be able to
        //  short-circuit with statistics.
        self.data_child()?
            .projection_evaluation(row_range, expr, mask)
    }
}

/// A wrapper for the result of pruning an expression against a zone map such that we can refresh
/// it each time the dynamic expressions are updated.
struct PruningResult {
    zone_map: ZoneMap,
    predicate: Expression,
    dynamic_updates: Option<DynamicExprUpdates>,
    latest_result: RwLock<(u64, Mask)>,
    session: VortexSession,
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

        tracing::debug!(
            "Re-computing pruning mask for version {version} on {}",
            self.predicate
        );

        let next_mask = self
            .zone_map
            .prune(&self.predicate, &self.session)
            .map_err(|err| {
                err.with_context(format!(
                    "While evaluating pruning predicate {}",
                    self.predicate
                ))
            })?;
        *guard = (version, next_mask.clone());

        Ok(next_mask)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use rstest::fixture;
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_buffer::buffer;
    use vortex_io::runtime::single::block_on;
    use vortex_mask::Mask;

    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::zoned::writer::ZonedLayoutOptions;
    use crate::layouts::zoned::writer::ZonedStrategy;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;

    #[fixture]
    /// Create a stats layout with three chunks of primitive arrays.
    fn stats_layout() -> (Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: 3,
                ..Default::default()
            },
        );
        let array_stream = ChunkedArray::from_iter([
            buffer![1, 2, 3].into_array(),
            buffer![4, 5, 6].into_array(),
            buffer![7, 8, 9].into_array(),
        ])
        .into_array()
        .to_array_stream()
        .sequenced(ptr);
        let layout = block_on(|handle| {
            strategy.write_stream(ctx, segments.clone(), array_stream, eof, handle)
        })
        .unwrap();
        (segments, layout)
    }

    #[rstest]
    fn test_stats_evaluator(
        #[from(stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        block_on(|_| async {
            let result = layout
                .new_reader("".into(), segments, &SESSION)
                .unwrap()
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &root(),
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();

            let expected = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
            assert_arrays_eq!(result, expected);
        })
    }

    #[rstest]
    fn test_stats_pruning_mask(
        #[from(stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        block_on(|_| async {
            let row_count = layout.row_count();
            let reader = layout.new_reader("".into(), segments, &SESSION).unwrap();

            // Choose a prune-able expression
            let expr = gt(root(), lit(7));

            let result = reader
                .pruning_evaluation(
                    &(0..row_count),
                    &expr,
                    Mask::new_true(row_count.try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();

            assert_eq!(
                result,
                Mask::from_iter([false, false, false, false, false, false, true, true, true])
            );
        })
    }
}
