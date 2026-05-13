// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use itertools::Itertools;
use tracing::trace;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::Expression;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::LazyReaderChildren;
use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::pruning::PruningState;
use crate::layouts::zoned::schema::stats_table_dtype;
use crate::segments::SegmentSource;

pub struct ZonedReader {
    layout: ZonedLayout,
    name: Arc<str>,
    lazy_children: Arc<LazyReaderChildren>,
    pruning: PruningState,
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
            stats_table_dtype(layout.dtype(), layout.present_stats()),
        ];
        let names = vec![Arc::clone(&name), format!("{}.zones", name).into()];
        let lazy_children = Arc::new(LazyReaderChildren::new(
            Arc::clone(&layout.children),
            dtypes,
            names,
            Arc::clone(&segment_source),
            session.clone(),
        ));

        Ok(Self {
            pruning: PruningState::new(&layout, Arc::clone(&lazy_children), session),
            layout,
            name,
            lazy_children,
        })
    }

    fn data_child(&self) -> VortexResult<&LayoutReaderRef> {
        self.lazy_children.get(0)
    }

    /// Get the range of zone IDs containing a row range.
    pub(crate) fn zone_range(&self, row_range: &Range<u64>) -> Range<u64> {
        // Caller must ensure zone_len > 0. Legacy files may deserialize with zone_len == 0, but
        // pruning_evaluation disables zoned pruning for those layouts before calling this helper.
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
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
        trace!("Stats pruning evaluation: {} - {}", &self.name, expr);
        let data_eval = self
            .data_child()?
            .pruning_evaluation(row_range, expr, mask.clone())?;

        if self.layout.zone_len == 0 {
            trace!("Stats pruning evaluation: skipping zoned pruning for legacy zero-length zones");
            return Ok(data_eval);
        }

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
    use vortex_array::scalar_fn::session::ScalarFnSession;
    use vortex_array::session::ArraySession;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::Handle;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use crate::IntoLayout;
    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::VTable;
    use crate::children::OwnedLayoutChildren;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::zoned::Zoned;
    use crate::layouts::zoned::ZonedLayoutEncoding;
    use crate::layouts::zoned::ZonedMetadata;
    use crate::layouts::zoned::writer::ZonedLayoutOptions;
    use crate::layouts::zoned::writer::ZonedStrategy;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::session::LayoutSession;

    fn session_with_handle(handle: Handle) -> VortexSession {
        VortexSession::empty()
            .with::<ArraySession>()
            .with::<LayoutSession>()
            .with::<ScalarFnSession>()
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
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = session_with_handle(handle);
            strategy
                .write_stream(ctx, segments2, array_stream, eof, &session)
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
            let session = session_with_handle(handle);
            let result = layout
                .new_reader("".into(), segments, &session)
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
        block_on(|handle| async {
            let row_count = layout.row_count();
            let session = session_with_handle(handle);
            let reader = layout.new_reader("".into(), segments, &session).unwrap();

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

    #[rstest]
    fn test_legacy_zero_zone_len_skips_zoned_pruning(
        #[from(stats_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) -> VortexResult<()> {
        let zoned_layout = layout.as_::<Zoned>();
        let children =
            OwnedLayoutChildren::layout_children(vec![layout.child(0)?, layout.child(1)?]);
        let legacy_layout = <Zoned as VTable>::build(
            &ZonedLayoutEncoding,
            layout.dtype(),
            layout.row_count(),
            &ZonedMetadata {
                zone_len: 0,
                present_stats: Arc::clone(zoned_layout.present_stats()),
            },
            vec![],
            children.as_ref(),
            &ReadContext::new([]),
        )?
        .into_layout();

        block_on(|handle| async {
            let row_count = legacy_layout.row_count();
            let session = session_with_handle(handle);
            let reader = legacy_layout.new_reader("".into(), segments, &session)?;

            let result = reader
                .pruning_evaluation(
                    &(0..row_count),
                    &gt(root(), lit(7)),
                    Mask::new_true(row_count.try_into().unwrap()),
                )?
                .await?;

            assert!(result.all_true());
            Ok(())
        })
    }

    #[test]
    fn test_writer_rejects_zero_block_size() {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: 0,
                ..Default::default()
            },
        );
        let array_stream = ChunkedArray::from_iter([buffer![1, 2, 3].into_array()])
            .into_array()
            .to_array_stream()
            .sequenced(ptr);
        let segments2 = Arc::<TestSegments>::clone(&segments);

        let result = block_on(|handle| async move {
            let session = session_with_handle(handle);
            strategy
                .write_stream(ctx, segments2, array_stream, eof, &session)
                .await
        });

        assert!(result.is_err());
    }
}
