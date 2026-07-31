// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::future::try_join_all;
use itertools::Itertools;
use tracing::trace;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::pack::Pack;
use vortex_array::scalar_fn::fns::stat::StatFn;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::Layout;
use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::LazyReaderChildren;
use crate::RowSplits;
use crate::SplitRange;
use crate::VTable;
use crate::layouts::zoned::ZonedData;
use crate::layouts::zoned::pruning::PruningState;
use crate::segments::SegmentSource;

/// Slice of projection's "row_range" covered by a single zone
struct ZoneSegment {
    zone_idx: u64,
    /// Range within "row_range", i.e. local_range.start is offset from
    /// row_range.start
    offset: Range<usize>,
    /// Absolute row range in file
    absolute: Range<u64>,
    /// True if zone is contained inside "row_range"
    fully_covered: bool,
}

pub struct ZonedReader {
    dtype: DType,
    row_count: u64,
    zone_len: usize,
    name: Arc<str>,
    lazy_children: Arc<LazyReaderChildren>,
    pruning: PruningState,
    session: VortexSession,
}

impl ZonedReader {
    pub(super) fn try_new<V>(
        layout: Layout<V>,
        zone_count: usize,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: crate::LayoutReaderContext,
    ) -> VortexResult<Self>
    where
        V: VTable<LayoutData = ZonedData>,
    {
        let aggregate_fns = layout.aggregate_fns();
        let dtypes = vec![layout.dtype().clone(), layout.stats_table_dtype.clone()];
        let names = vec![Arc::clone(&name), format!("{}.zones", name).into()];
        let lazy_children = Arc::new(LazyReaderChildren::new(
            Arc::clone(layout.children()),
            dtypes,
            names,
            Arc::clone(&segment_source),
            session.clone(),
            ctx,
        ));
        let dtype = layout.dtype().clone();
        let row_count = layout.row_count();
        let zone_len = layout.zone_len;

        Ok(Self {
            pruning: PruningState::new(
                layout,
                zone_count,
                aggregate_fns,
                Arc::clone(&lazy_children),
                session.clone(),
            ),
            dtype,
            row_count,
            zone_len,
            name,
            lazy_children,
            session,
        })
    }

    fn data_child(&self) -> VortexResult<&LayoutReaderRef> {
        self.lazy_children.get(0)
    }

    /// Get the range of zone IDs containing a row range.
    pub(crate) fn zone_range(&self, row_range: &Range<u64>) -> Range<u64> {
        // Callers rely on `zone_len > 0`. `new_reader` never constructs a `ZonedReader` for a
        // zero-length zone map (it reads the data child directly), so this holds by construction.
        debug_assert!(self.zone_len > 0, "zone_len must be > 0");

        let zone_len_u64 = self.zone_len as u64;
        let zone_start = row_range.start / zone_len_u64;
        let zone_end = row_range.end.div_ceil(zone_len_u64);
        zone_start..zone_end
    }

    /// Get the row index for the first row in a zone with the given `zone_index`.
    pub(crate) fn first_row_offset(&self, zone_idx: u64) -> u64 {
        zone_idx
            .saturating_mul(self.zone_len as u64)
            .min(self.row_count)
    }
}

impl LayoutReader for ZonedReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        self.data_child()?
            .register_splits(field_mask, split_range, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        trace!("Stats pruning evaluation: {} - {}", &self.name, expr);
        let data_eval = self
            .data_child()?
            .pruning_evaluation(row_range, expr, mask.clone())?;

        let Some(pruning_mask_future) = self.pruning.pruning_mask_future(expr.clone()) else {
            trace!("Stats pruning evaluation: not prune-able {expr}");
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

        let name = Arc::clone(&self.name);
        let expr = expr.clone();

        Ok(MaskFuture::new(mask.len(), async move {
            trace!("Invoking stats pruning evaluation {}: {}", name, expr);

            let pruning_mask = pruning_mask_future.await?.mask()?;

            let mut builder = BitBufferMut::with_capacity(mask.len());
            for (zone_idx, &zone_length) in zone_range.clone().zip_eq(&zone_lengths) {
                builder.append_n(!pruning_mask.value(usize::try_from(zone_idx)?), zone_length);
            }

            let stats_mask = Mask::from(builder.freeze());
            assert_eq!(stats_mask.len(), mask.len(), "Mask length mismatch");

            // Intersect the masks.
            let mask_density = mask.density();
            let mut stats_mask = mask.bitand(&stats_mask);

            // Forward to data child for further pruning.
            if !stats_mask.all_false() {
                let data_mask = data_eval.await?;
                stats_mask = stats_mask.bitand(&data_mask);
            }

            trace!(
                "Stats evaluation approx {} - {} (mask = {}) => {}",
                name,
                expr,
                mask_density,
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
        if self.zone_len > 0 {
            if is_zone_stat(expr) {
                let aggregate_fn = expr.as_::<StatFn>().aggregate_fn().clone();
                let dtype = expr.return_dtype(&self.dtype)?;
                return self.aggregate_projection(row_range, aggregate_fn, dtype, mask);
            }

            // Multiple aggregations over a column are packed together per aggregate.
            if expr.is::<Pack>()
                && (0..expr.as_::<Pack>().names.len()).any(|i| is_zone_stat(expr.child(i)))
            {
                return self.aggregate_pack_projection(row_range, expr, mask);
            }
        }

        self.data_child()?
            .projection_evaluation(row_range, expr, mask)
    }
}

/// True if "expr" is an aggregate stat of form "stat(root(), agg)"
fn is_zone_stat(expr: &Expression) -> bool {
    expr.is::<StatFn>() && is_root(expr.child(0))
}

impl ZonedReader {
    fn aggregate_pack_projection(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let options = expr.as_::<Pack>();
        let names = options.names.clone();
        let validity = options.nullability.into();

        let futures = (0..names.len())
            .map(|i| {
                let field = expr.child(i);
                if is_zone_stat(field) {
                    let aggregate_fn = field.as_::<StatFn>().aggregate_fn().clone();
                    let dtype = field.return_dtype(&self.dtype)?;
                    self.aggregate_projection(row_range, aggregate_fn, dtype, mask.clone())
                } else {
                    self.data_child()?
                        .projection_evaluation(row_range, field, mask.clone())
                }
            })
            .try_collect::<_, Vec<_>, _>()?;

        Ok(Box::pin(async move {
            let fields = try_join_all(futures).await?;
            let len = fields.first().map(|f| f.len()).unwrap_or(0);
            Ok(StructArray::new(names, fields, len, validity).into_array())
        }))
    }

    fn aggregate_projection(
        &self,
        row_range: &Range<u64>,
        aggregate_fn: AggregateFnRef,
        out_dtype: DType,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let zone_len = self.zone_len as u64;
        let row_count = self.row_count;
        let zone_range = self.zone_range(row_range);

        // For every zone, slice of "row_range" this zone covers.
        let segments: Vec<ZoneSegment> = zone_range
            .map(|zone_idx| {
                let zone_start = zone_idx.saturating_mul(zone_len).min(row_count);
                let zone_end = zone_idx
                    .saturating_add(1)
                    .saturating_mul(zone_len)
                    .min(row_count);

                let absolute_start = zone_start.max(row_range.start);
                let absolute_end = zone_end.min(row_range.end);
                let absolute = absolute_start..absolute_end;

                let offset_start: usize = usize::try_from(absolute_start - row_range.start)?;
                let offset_end: usize = usize::try_from(absolute_end - row_range.start)?;
                let offset = offset_start..offset_end;

                let fully_covered = zone_start >= row_range.start && zone_end <= row_range.end;

                Ok::<_, VortexError>(ZoneSegment {
                    zone_idx,
                    offset,
                    absolute,
                    fully_covered,
                })
            })
            .try_collect()?;

        let data_child = Arc::clone(self.data_child()?);
        let zone_map = self.pruning.shared_zone_map();
        let column_dtype = self.dtype.clone();
        let session = self.session.clone();

        Ok(Box::pin(async move {
            let mask = mask.await?;
            let true_count = mask.true_count();
            let mut ctx = session.create_execution_ctx();
            let mut accumulator = aggregate_fn.accumulator(&column_dtype)?;

            let zone_map = match zone_map.await {
                Ok(zone_map) if zone_map.supports_zone_partial(&aggregate_fn) => Some(zone_map),
                Ok(_) | Err(_) => None,
            };

            let covered = |segment: &ZoneSegment| {
                zone_map.is_some()
                    && segment.fully_covered
                    && mask.slice(segment.offset.clone()).all_true()
            };
            let mut i = 0;
            while i < segments.len() {
                let start_covered = covered(&segments[i]);
                let mut j = i + 1;
                while j < segments.len() && covered(&segments[j]) == start_covered {
                    j += 1;
                }

                if start_covered && let Some(zone_map) = zone_map.as_ref() {
                    let range = segments[i].zone_idx..segments[j - 1].zone_idx + 1;
                    let range = usize::try_from(range.start)?..usize::try_from(range.end)?;
                    zone_map.fold_zone_partials(
                        &aggregate_fn,
                        range,
                        &mut accumulator,
                        &session,
                    )?;
                } else {
                    let sub_mask = mask.slice(segments[i].offset.start..segments[j - 1].offset.end);
                    if !sub_mask.all_false() {
                        let sub_range = segments[i].absolute.start..segments[j - 1].absolute.end;
                        let array = data_child
                            .projection_evaluation(
                                &sub_range,
                                &root(),
                                MaskFuture::ready(sub_mask),
                            )?
                            .await?;
                        accumulator.accumulate(&array, &mut ctx)?;
                    }
                }
                i = j;
            }

            let partial = accumulator.partial_scalar()?.cast(&out_dtype)?;
            Ok(ConstantArray::new(partial, true_count).into_array())
        }))
    }
}

#[cfg(test)]
mod test {
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use rstest::fixture;
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::gt;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::expr::stats::Stat;
    use vortex_array::stats::expr::stat;
    use vortex_array::stats::expr::sum;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_io::runtime::Handle;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use crate::LayoutBuildContext;
    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::VTable;
    use crate::children::OwnedLayoutChildren;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::zoned::LegacyStats;
    use crate::layouts::zoned::LegacyStatsLayoutEncoding;
    use crate::layouts::zoned::LegacyStatsMetadata;
    use crate::layouts::zoned::Zoned;
    use crate::layouts::zoned::writer::ZonedLayoutOptions;
    use crate::layouts::zoned::writer::ZonedStrategy;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::session::LayoutSession;

    fn session_with_handle(handle: Handle) -> VortexSession {
        array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle)
    }

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
                block_size: NonZeroUsize::new(3).vortex_expect("non zero"),
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
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = session_with_handle(handle);
            strategy
                .write_stream(ctx.into(), segments2, array_stream, eof, &session)
                .await
        })
        .unwrap();
        (segments, layout)
    }

    #[rstest]
    fn test_stats_evaluator(
        #[from(stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        block_on(|handle| async {
            let mut ctx = array_session().create_execution_ctx();
            let session = session_with_handle(handle);
            let result = layout
                .new_reader("".into(), segments, &session, &Default::default())
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
            assert_arrays_eq!(result, expected, &mut ctx);
        })
    }

    /// Test aggregate projection answers from zone maps
    #[rstest]
    fn test_aggregate_projection(
        #[from(stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let rows = layout.row_count();
        let range = &(0..rows);
        let mask = MaskFuture::new_true(rows.try_into()?);

        block_on(|handle| async {
            let session = session_with_handle(handle);
            let reader = layout.new_reader("".into(), segments, &session, &Default::default())?;

            let sum = reader
                .projection_evaluation(range, &sum(root()), mask.clone())?
                .await?;
            assert_eq!(sum.len(), usize::try_from(rows)?);
            assert_eq!(
                sum.execute_scalar(0, &mut ctx)?
                    .as_primitive()
                    .typed_value::<i64>(),
                Some(45)
            );

            let min_expr = &stat(root(), Stat::Min.aggregate_fn().unwrap());
            let min = reader
                .projection_evaluation(range, min_expr, mask.clone())?
                .await?;
            assert_eq!(
                min.execute_scalar(0, &mut ctx)?
                    .as_primitive()
                    .typed_value::<i32>(),
                Some(1)
            );

            let max_expr = &stat(root(), Stat::Max.aggregate_fn().unwrap());
            let max = reader.projection_evaluation(range, max_expr, mask)?.await?;
            assert_eq!(
                max.execute_scalar(0, &mut ctx)?
                    .as_primitive()
                    .typed_value::<i32>(),
                Some(9)
            );
            Ok(())
        })
    }

    /// Test aggregate projection with filter decodes columns for zone maps
    /// which are partially taken
    #[rstest]
    fn test_aggregate_projection_hybrid(
        #[from(stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let rows = layout.row_count();

        // zones: [1,2,3][4,5,6][7,8,9]. Cover zone 0 and 2 fully but select
        // only row 3 (=4) of zone 1. We must count every selected row exactly
        // once: 1+2+3 + 4 + 7+8+9 = 34.
        let mask: Mask = [true, true, true, true, false, false, true, true, true]
            .into_iter()
            .collect();
        assert_eq!(mask.len(), usize::try_from(rows)?);

        block_on(|handle| async {
            let session = session_with_handle(handle);
            let reader = layout.new_reader("".into(), segments, &session, &Default::default())?;
            let sum = reader
                .projection_evaluation(&(0..rows), &sum(root()), MaskFuture::ready(mask))?
                .await?;

            assert_eq!(sum.len(), 7);
            assert_eq!(
                sum.execute_scalar(0, &mut ctx)?
                    .as_primitive()
                    .typed_value::<i64>(),
                Some(34)
            );
            Ok(())
        })
    }

    #[rstest]
    fn test_stats_pruning_mask(
        #[from(stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        block_on(|handle| async {
            let row_count = layout.row_count();
            let session = session_with_handle(handle);
            let reader = layout
                .new_reader("".into(), segments, &session, &Default::default())
                .unwrap();

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

    #[test]
    fn test_default_zoned_null_count_pruning_mask() {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: NonZeroUsize::new(3).vortex_expect("non zero"),
                ..Default::default()
            },
        );
        let array_stream = ChunkedArray::from_iter([
            PrimitiveArray::new(
                buffer![0i32, 0, 0],
                Validity::from_iter([false, false, false]),
            )
            .into_array(),
            PrimitiveArray::new(buffer![1i32, 2, 3], Validity::from_iter([true, true, true]))
                .into_array(),
            PrimitiveArray::new(
                buffer![0i32, 0, 0],
                Validity::from_iter([false, false, false]),
            )
            .into_array(),
        ])
        .into_array()
        .to_array_stream()
        .sequenced(ptr);
        let segments2 = Arc::<TestSegments>::clone(&segments);

        let layout = block_on(|handle| async move {
            let session = session_with_handle(handle);
            strategy
                .write_stream(ctx.into(), segments2, array_stream, eof, &session)
                .await
        })
        .unwrap();

        block_on(|handle| async {
            let row_count = layout.row_count();
            let session = session_with_handle(handle);
            let reader = layout
                .new_reader("".into(), segments, &session, &Default::default())
                .unwrap();

            let result = reader
                .pruning_evaluation(
                    &(0..row_count),
                    &is_not_null(root()),
                    Mask::new_true(row_count.try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();

            assert_eq!(
                result,
                Mask::from_iter([false, false, false, true, true, true, false, false, false])
            );
        })
    }

    #[rstest]
    #[case::zero_zone_len(0, [true; 9])]
    #[case::zoned_reader(
        3,
        [false, false, false, false, false, false, true, true, true]
    )]
    fn test_legacy_zoned_reader(
        #[from(stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
        #[case] zone_len: u32,
        #[case] expected: [bool; 9],
    ) -> VortexResult<()> {
        let zoned_layout = layout.as_::<Zoned>();
        let children = OwnedLayoutChildren::layout_children(vec![
            layout
                .slot(0)?
                .vortex_expect("ZonedLayout always has a data child"),
            layout
                .slot(1)?
                .vortex_expect("ZonedLayout always has a stats child"),
        ]);
        let session = array_session();
        let read_ctx = ReadContext::new([]);
        let build_ctx = LayoutBuildContext {
            session: &session,
            array_read_ctx: &read_ctx,
        };
        let legacy_layout = <LegacyStats as VTable>::build(
            &LegacyStatsLayoutEncoding,
            layout.dtype(),
            layout.row_count(),
            &LegacyStatsMetadata {
                zone_len,
                zone_map_schema: zoned_layout.zone_map_schema.clone(),
            },
            vec![],
            children.as_ref(),
            &build_ctx,
        )?
        .into_layout();

        block_on(|handle| async {
            let row_count = legacy_layout.row_count();
            let session = session_with_handle(handle);
            let reader =
                legacy_layout.new_reader("".into(), segments, &session, &Default::default())?;

            let result = reader
                .pruning_evaluation(
                    &(0..row_count),
                    &gt(root(), lit(7)),
                    Mask::new_true(row_count.try_into().unwrap()),
                )?
                .await?;

            assert_eq!(result, Mask::from_iter(expected));

            let projected = reader
                .projection_evaluation(
                    &(0..row_count),
                    &root(),
                    MaskFuture::new_true(row_count.try_into().unwrap()),
                )?
                .await?;
            let mut ctx = array_session().create_execution_ctx();
            assert_arrays_eq!(
                projected,
                buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9].into_array(),
                &mut ctx
            );
            Ok(())
        })
    }
}
