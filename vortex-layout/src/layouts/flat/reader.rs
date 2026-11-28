// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use tracing::trace;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_expr::ArrayExprExt;
use vortex_expr::BoundExpression;
use vortex_functions::variant_get::decode_for_repeated_extraction;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::layouts::SharedArrayFuture;
use crate::layouts::flat::FlatLayout;
use crate::reader::LayoutReader;
use crate::reader::RowSplits;
use crate::reader::SplitRange;
use crate::segments::SegmentSource;

/// The threshold of mask density below which we will evaluate the expression only over the
/// selected rows, and above which we evaluate the expression over all rows and then select
/// after.
// TODO(ngates): more experimentation is needed, and this should probably be dynamic based on the
//  actual expression? Perhaps all expressions are given a selection mask to decide for themselves?
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

pub struct FlatReader {
    layout: FlatLayout,
    name: Arc<str>,
    segment_source: Arc<dyn SegmentSource>,
    session: VortexSession,
}

impl FlatReader {
    pub(crate) fn new(
        layout: FlatLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> Self {
        Self {
            layout,
            name,
            segment_source,
            session,
        }
    }

    /// Return a future for the decoded array, registering the segment early when appropriate.
    fn array_future(&self, mask: &MaskFuture) -> SharedArrayFuture {
        let row_count =
            usize::try_from(self.layout.row_count()).vortex_expect("row count must fit in usize");

        let id = self.layout.segment_id();
        let defer_request = match mask.peek() {
            Some(mask) => mask.all_false(),
            // A pending filter can eliminate a Variant's residual payload. Keep other
            // columns eligible to coalesce with the filter's reads.
            None => self.layout.dtype().is_variant(),
        };
        let segment_fut = if defer_request {
            let source = Arc::clone(&self.segment_source);
            async move { source.request(id).await }.boxed()
        } else {
            self.segment_source.request(id)
        };

        let ctx = self.layout.array_ctx().clone();
        let session = self.session.clone();
        let dtype = self.layout.dtype().clone();
        let array_tree = self.layout.array_tree().cloned();
        async move {
            let segment = segment_fut.await?;
            let parts = if let Some(array_tree) = array_tree {
                // Use the pre-stored flatbuffer from layout metadata combined with segment buffers.
                SerializedArray::from_flatbuffer_and_segment(array_tree, segment)?
            } else {
                // Parse the flatbuffer from the segment itself.
                SerializedArray::try_from(segment)?
            };
            parts
                .decode(&dtype, row_count, &ctx, &session)
                .map_err(Arc::new)
        }
        .boxed()
        .shared()
    }
}

impl LayoutReader for FlatReader {
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
        _field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        split_range.check_bounds(self.layout.row_count())?;
        splits.push(split_range.root_row_range().end);
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let row_range = usize::try_from(row_range.start)
            .vortex_expect("Row range begin must fit within FlatLayout size")
            ..usize::try_from(row_range.end)
                .vortex_expect("Row range end must fit within FlatLayout size");
        if mask.peek().is_some_and(Mask::all_false) {
            return Ok(mask);
        }
        let name = Arc::clone(&self.name);
        let array = self.array_future(&mask);
        let expr = expr.clone();
        let session = self.session.clone();

        Ok(MaskFuture::new(mask.len(), async move {
            let mask = mask.await?;
            if mask.all_false() {
                return Ok(mask);
            }
            let mut array = array.await?;

            // Slice the array based on the row mask.
            if row_range.start > 0 || row_range.end < array.len() {
                array = array.slice(row_range.clone())?;
            }

            let mask_density = mask.density();
            let sparse = mask_density < EXPR_EVAL_THRESHOLD;
            let mut ctx = session.create_execution_ctx();
            // Filtering Variant rows before decoding bounds repeated path extraction by the
            // surviving payload. Other encodings can push the expression into the full array.
            let prefilter = sparse && array.dtype().is_variant();
            if prefilter {
                array = array.filter(mask.clone())?;
            }
            let array =
                decode_for_repeated_extraction(array, &expr, &mut ctx)?.apply_bound(&expr)?;
            let array = if sparse && !prefilter {
                array.filter(mask.clone())?
            } else {
                array
            };
            let result = array.fill_null(false)?.execute::<Mask>(&mut ctx)?;
            let array_mask = if sparse {
                mask.intersect_by_rank(&result)
            } else {
                mask.bitand(&result)
            };

            trace!(
                "Flat mask evaluation {} - {} (mask = {}) => {}",
                name,
                expr,
                mask_density,
                array_mask.density(),
            );

            Ok(array_mask)
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let row_range = usize::try_from(row_range.start)
            .vortex_expect("Row range begin must fit within FlatLayout size")
            ..usize::try_from(row_range.end)
                .vortex_expect("Row range end must fit within FlatLayout size");
        let name = Arc::clone(&self.name);
        let array = self.array_future(&mask);
        let expr = expr.clone();
        let session = self.session.clone();

        Ok(async move {
            trace!("Flat array evaluation {} - {}", name, expr);

            // Wait for the mask first: a chunk without selected rows needs no decode.
            let mask = mask.await?;
            if mask.all_false() {
                return Canonical::try_empty(expr.dtype()).map(IntoArray::into_array);
            }

            let mut array = array.clone().await?;

            // Slice the array based on the row mask.
            if row_range.start > 0 || row_range.end < array.len() {
                array = array.slice(row_range.clone())?;
            }

            // First apply the filter to the array.
            // NOTE(ngates): we *must* filter first before applying the expression, as the
            // expression may depend on the filtered rows being removed e.g.
            //  `CAST(a, u8) WHERE a < 256`
            if !mask.all_true() {
                array = array.filter(mask)?;
            }

            array =
                decode_for_repeated_extraction(array, &expr, &mut session.create_execution_ctx())?;

            // Evaluate the projection expression.
            array = array.apply_bound(&expr)?;

            Ok(array)
        }
        .boxed())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_expr::gt;
    use vortex_expr::is_not_null;
    use vortex_expr::lit;
    use vortex_expr::root;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_mask::Mask;
    use vortex_session::registry::ReadContext;

    use crate::LayoutStrategy;
    use crate::layouts::flat::FlatLayout;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::CountingSegmentSource;
    use crate::segments::SegmentId;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::new_session;

    #[rstest]
    #[case::ready_bool(DType::Bool(Nullability::Nullable), true, 0)]
    #[case::pending_bool(DType::Bool(Nullability::Nullable), false, 2)]
    #[case::pending_variant(DType::Variant(Nullability::Nullable), false, 0)]
    fn empty_mask_segment_requests(
        #[case] dtype: DType,
        #[case] ready: bool,
        #[case] expected_requests: usize,
    ) -> VortexResult<()> {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let segments = Arc::new(CountingSegmentSource::new(
                Arc::new(TestSegments::default()),
            ));
            let layout = FlatLayout::new(4, dtype, SegmentId::from(0), ReadContext::new([]));
            let reader = layout.into_layout().new_reader(
                "".into(),
                Arc::<CountingSegmentSource>::clone(&segments),
                &session,
                &Default::default(),
            )?;
            let mask = if ready {
                MaskFuture::ready(Mask::new_false(4))
            } else {
                MaskFuture::new(4, async { Ok(Mask::new_false(4)) })
            };
            let predicate = is_not_null(root()).bind(reader.dtype())?;
            let filter = reader.filter_evaluation(&(0..4), &predicate, mask.clone())?;
            let root = root().bind(reader.dtype())?;
            let projection = reader.projection_evaluation(&(0..4), &root, mask)?;
            assert_eq!(segments.requests(), expected_requests);
            assert!(filter.await?.all_false());
            let projected = projection.await?;
            assert_eq!(projected.dtype(), reader.dtype());
            assert!(projected.is_empty());
            assert_eq!(segments.requests(), expected_requests);
            Ok(())
        })
    }

    #[test]
    fn flat_identity() -> VortexResult<()> {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let mut ctx = session.create_execution_ctx();
            let array_ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array =
                PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid).into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    array_ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await?;

            assert_eq!(
                format!("{}", layout),
                "vortex.flat(i32?, rows=5, segments=[0])"
            );

            let reader = layout.new_reader("".into(), segments, &session, &Default::default())?;
            let expr = root().bind(reader.dtype())?;
            let result = reader
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &expr,
                    MaskFuture::new_true(layout.row_count().try_into()?),
                )?
                .await?;

            assert_arrays_eq!(result, array, &mut ctx);

            Ok(())
        })
    }

    #[test]
    fn flat_expr() {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let mut ctx = session.create_execution_ctx();
            let array_ctx = ArrayContext::empty();

            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array =
                PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid).into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    array_ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            let reader = layout
                .new_reader("".into(), segments, &session, &Default::default())
                .unwrap();
            let expr = gt(root(), lit(3i32)).bind(reader.dtype()).unwrap();
            let result = reader
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &expr,
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();

            let expected = BoolArray::from_iter([false, false, false, true, true].map(Some));
            assert_arrays_eq!(result, expected, &mut ctx);
        })
    }

    #[test]
    fn flat_unaligned_row_mask() {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let mut ctx = session.create_execution_ctx();
            let array_ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array =
                PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid).into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    array_ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            let reader = layout
                .new_reader("".into(), segments, &session, &Default::default())
                .unwrap();
            let expr = root().bind(reader.dtype()).unwrap();
            let result = reader
                .projection_evaluation(&(2..4), &expr, MaskFuture::new_true(2))
                .unwrap()
                .await
                .unwrap();

            let expected = PrimitiveArray::new(buffer![3i32, 4], Validity::AllValid).into_array();
            assert_arrays_eq!(result, expected, &mut ctx);
        })
    }
}
