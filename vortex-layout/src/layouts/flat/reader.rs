// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::FutureExt;
use futures::future::BoxFuture;
use tracing::trace;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::Expression;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
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
    array: OnceLock<SharedArrayFuture>,
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
            array: Default::default(),
        }
    }

    /// Register the segment request and return a future that would resolve into the deserialised array.
    fn array_future(&self) -> SharedArrayFuture {
        let row_count =
            usize::try_from(self.layout.row_count()).vortex_expect("row count must fit in usize");

        self.array
            .get_or_init(|| {
                let segment_fut = self.segment_source.request(self.layout.segment_id());

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
            })
            .clone()
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
        split_range.check_bounds(self.layout.row_count)?;
        splits.push(split_range.root_row_range().end);
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let row_range = usize::try_from(row_range.start)
            .vortex_expect("Row range begin must fit within FlatLayout size")
            ..usize::try_from(row_range.end)
                .vortex_expect("Row range end must fit within FlatLayout size");
        let name = Arc::clone(&self.name);
        let array = self.array_future();
        let expr = expr.clone();
        let session = self.session.clone();

        Ok(MaskFuture::new(mask.len(), async move {
            // TODO(ngates): if the mask density is low enough, or if the mask is dense within a range
            //  (as often happens with zone map pruning), then we could slice/filter the array prior
            //  to evaluating the expression.
            let mut array = array.clone().await?;
            let mask = mask.await?;

            // Slice the array based on the row mask.
            if row_range.start > 0 || row_range.end < array.len() {
                array = array.slice(row_range.clone())?;
            }

            let mask_density = mask.density();
            let array_mask = if mask_density < EXPR_EVAL_THRESHOLD {
                // We have the choice to apply the filter or the expression first, we apply the
                // expression first so that it can try pushing down itself and then the filter
                // after this.
                let array = array.apply(&expr)?;
                let array = array.filter(mask.clone())?;
                let mut ctx = session.create_execution_ctx();
                let array_mask = array.null_as_false().execute(&mut ctx)?;

                mask.intersect_by_rank(&array_mask)
            } else {
                // Run over the full array, with a simpler bitand at the end.
                let array = array.apply(&expr)?;
                let mut ctx = session.create_execution_ctx();
                let array_mask = array.null_as_false().execute(&mut ctx)?;

                mask.bitand(&array_mask)
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
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let row_range = usize::try_from(row_range.start)
            .vortex_expect("Row range begin must fit within FlatLayout size")
            ..usize::try_from(row_range.end)
                .vortex_expect("Row range end must fit within FlatLayout size");
        let name = Arc::clone(&self.name);
        let array = self.array_future();
        let expr = expr.clone();

        Ok(async move {
            trace!("Flat array evaluation {} - {}", name, expr);

            let mut array = array.clone().await?;
            let mask = mask.await?;

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

            // Evaluate the projection expression.
            array = array.apply(&expr)?;

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
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::byte_length;
    use vortex_array::expr::checked_add;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::not_eq;
    use vortex_array::expr::root;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::LayoutReaderRef;
    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::SegmentFuture;
    use crate::segments::SegmentId;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;

    #[derive(Clone)]
    struct CountingSegmentSource {
        inner: TestSegments,
        requests: Arc<AtomicUsize>,
    }

    impl SegmentSource for CountingSegmentSource {
        fn request(&self, id: SegmentId) -> SegmentFuture {
            self.requests.fetch_add(1, Ordering::Relaxed);
            self.inner.request(id)
        }
    }

    #[derive(Clone, Copy)]
    enum RegressedQueryPattern {
        ProjectionOnly,
        FilterOnly,
        FilteredProjection,
        ComputedProjection,
        StringFilteredProjection,
        StringFilteredComputedProjection,
    }

    async fn counted_flat_reader(
        session: &VortexSession,
        array: ArrayRef,
        requests: Arc<AtomicUsize>,
    ) -> VortexResult<LayoutReaderRef> {
        let array_ctx = ArrayContext::empty();
        let segments = TestSegments::default();
        let source: Arc<dyn SegmentSource> = Arc::new(CountingSegmentSource {
            inner: segments.clone(),
            requests,
        });

        let (ptr, eof) = SequenceId::root().split();
        let layout = FlatLayoutStrategy::default()
            .write_stream(
                array_ctx,
                Arc::new(segments),
                array.to_array_stream().sequenced(ptr),
                eof,
                session,
            )
            .await?;

        layout.new_reader("".into(), source, session, &Default::default())
    }

    #[test]
    fn flat_identity() -> VortexResult<()> {
        block_on(|handle| async {
            let session = SESSION.clone().with_handle(handle);
            let mut ctx = session.create_execution_ctx();
            let array_ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array =
                PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid).into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    array_ctx,
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

            let result = layout
                .new_reader("".into(), segments, &session, &Default::default())?
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &root(),
                    MaskFuture::new_true(layout.row_count().try_into()?),
                )?
                .await?;

            assert_arrays_eq!(result, array, &mut ctx);

            Ok(())
        })
    }

    #[rstest]
    #[case::clickbench_q3_q7_q9_q10_projection_only(RegressedQueryPattern::ProjectionOnly)]
    #[case::clickbench_q2_tpcds_q88_q96_filter_only(RegressedQueryPattern::FilterOnly)]
    #[case::tpcds_q19_q33_q56_numeric_filter_projection(RegressedQueryPattern::FilteredProjection)]
    #[case::clickbench_q19_tpcds_q50_q79_computed_projection(
        RegressedQueryPattern::ComputedProjection
    )]
    #[case::clickbench_q11_q25_string_filter_projection(
        RegressedQueryPattern::StringFilteredProjection
    )]
    #[case::clickbench_q29_tpcds_q34_q46_q68_string_computed_filter_projection(
        RegressedQueryPattern::StringFilteredComputedProjection
    )]
    fn flat_subrange_regressed_query_patterns_reuse_segment_request(
        #[case] pattern: RegressedQueryPattern,
    ) -> VortexResult<()> {
        block_on(|handle| async {
            let session = SESSION.clone().with_handle(handle);
            let mut ctx = session.create_execution_ctx();
            let requests = Arc::new(AtomicUsize::new(0));

            match pattern {
                RegressedQueryPattern::ProjectionOnly => {
                    let reader = counted_flat_reader(
                        &session,
                        buffer![1, 2, 3, 4, 5, 6].into_array(),
                        Arc::clone(&requests),
                    )
                    .await?;
                    let first =
                        reader.projection_evaluation(&(0..3), &root(), MaskFuture::new_true(3))?;
                    let second =
                        reader.projection_evaluation(&(3..6), &root(), MaskFuture::new_true(3))?;

                    let (first, second): (VortexResult<ArrayRef>, VortexResult<ArrayRef>) =
                        futures::join!(first, second);
                    assert_arrays_eq!(first?, buffer![1, 2, 3].into_array(), &mut ctx);
                    assert_arrays_eq!(second?, buffer![4, 5, 6].into_array(), &mut ctx);
                }
                RegressedQueryPattern::FilterOnly => {
                    let reader = counted_flat_reader(
                        &session,
                        buffer![0, 1, 2, 0, 4, 5].into_array(),
                        Arc::clone(&requests),
                    )
                    .await?;
                    let filter = not_eq(root(), lit(0_i32));
                    let first =
                        reader.filter_evaluation(&(0..3), &filter, MaskFuture::new_true(3))?;
                    let second =
                        reader.filter_evaluation(&(3..6), &filter, MaskFuture::new_true(3))?;

                    let (first, second) = futures::join!(first, second);
                    assert_eq!(first?.true_count(), 2);
                    assert_eq!(second?.true_count(), 2);
                }
                RegressedQueryPattern::FilteredProjection => {
                    let reader = counted_flat_reader(
                        &session,
                        buffer![0, 1, 2, 3, 0, 5].into_array(),
                        Arc::clone(&requests),
                    )
                    .await?;
                    let filter = gt(root(), lit(1_i32));
                    let first_mask =
                        reader.filter_evaluation(&(0..3), &filter, MaskFuture::new_true(3))?;
                    let second_mask =
                        reader.filter_evaluation(&(3..6), &filter, MaskFuture::new_true(3))?;
                    let first = reader.projection_evaluation(&(0..3), &root(), first_mask)?;
                    let second = reader.projection_evaluation(&(3..6), &root(), second_mask)?;

                    let (first, second): (VortexResult<ArrayRef>, VortexResult<ArrayRef>) =
                        futures::join!(first, second);
                    assert_arrays_eq!(first?, buffer![2].into_array(), &mut ctx);
                    assert_arrays_eq!(second?, buffer![3, 5].into_array(), &mut ctx);
                }
                RegressedQueryPattern::ComputedProjection => {
                    let projection = checked_add(root(), lit(10_i32));
                    let source = buffer![1, 2, 3, 4, 5, 6].into_array();
                    let expected = source.clone().apply(&projection)?;
                    let reader =
                        counted_flat_reader(&session, source, Arc::clone(&requests)).await?;
                    let first = reader.projection_evaluation(
                        &(0..3),
                        &projection,
                        MaskFuture::new_true(3),
                    )?;
                    let second = reader.projection_evaluation(
                        &(3..6),
                        &projection,
                        MaskFuture::new_true(3),
                    )?;

                    let (first, second): (VortexResult<ArrayRef>, VortexResult<ArrayRef>) =
                        futures::join!(first, second);
                    assert_arrays_eq!(first?, expected.slice(0..3)?, &mut ctx);
                    assert_arrays_eq!(second?, expected.slice(3..6)?, &mut ctx);
                }
                RegressedQueryPattern::StringFilteredProjection => {
                    let filter = not_eq(root(), lit(Scalar::utf8("", Nullability::Nullable)));
                    let source = VarBinArray::from_iter(
                        [
                            Some(""),
                            Some("abc"),
                            Some("de"),
                            Some(""),
                            Some("fghi"),
                            Some("j"),
                        ],
                        DType::Utf8(Nullability::Nullable),
                    )
                    .into_array();
                    let first_expected = source
                        .slice(0..3)?
                        .filter(Mask::from_iter([false, true, true]))?;
                    let second_expected = source
                        .slice(3..6)?
                        .filter(Mask::from_iter([false, true, true]))?;
                    let reader =
                        counted_flat_reader(&session, source, Arc::clone(&requests)).await?;
                    let first_mask =
                        reader.filter_evaluation(&(0..3), &filter, MaskFuture::new_true(3))?;
                    let second_mask =
                        reader.filter_evaluation(&(3..6), &filter, MaskFuture::new_true(3))?;
                    let first = reader.projection_evaluation(&(0..3), &root(), first_mask)?;
                    let second = reader.projection_evaluation(&(3..6), &root(), second_mask)?;

                    let (first, second): (VortexResult<ArrayRef>, VortexResult<ArrayRef>) =
                        futures::join!(first, second);
                    assert_arrays_eq!(first?, first_expected, &mut ctx);
                    assert_arrays_eq!(second?, second_expected, &mut ctx);
                }
                RegressedQueryPattern::StringFilteredComputedProjection => {
                    let projection = byte_length(root());
                    let filter = not_eq(root(), lit(Scalar::utf8("", Nullability::Nullable)));
                    let source = VarBinArray::from_iter(
                        [
                            Some(""),
                            Some("abc"),
                            Some("de"),
                            Some(""),
                            Some("fghi"),
                            Some("j"),
                        ],
                        DType::Utf8(Nullability::Nullable),
                    )
                    .into_array();
                    let first_expected = source
                        .slice(0..3)?
                        .filter(Mask::from_iter([false, true, true]))?
                        .apply(&projection)?;
                    let second_expected = source
                        .slice(3..6)?
                        .filter(Mask::from_iter([false, true, true]))?
                        .apply(&projection)?;
                    let reader =
                        counted_flat_reader(&session, source, Arc::clone(&requests)).await?;
                    let first_mask =
                        reader.filter_evaluation(&(0..3), &filter, MaskFuture::new_true(3))?;
                    let second_mask =
                        reader.filter_evaluation(&(3..6), &filter, MaskFuture::new_true(3))?;
                    let first = reader.projection_evaluation(&(0..3), &projection, first_mask)?;
                    let second = reader.projection_evaluation(&(3..6), &projection, second_mask)?;

                    let (first, second): (VortexResult<ArrayRef>, VortexResult<ArrayRef>) =
                        futures::join!(first, second);
                    assert_arrays_eq!(first?, first_expected, &mut ctx);
                    assert_arrays_eq!(second?, second_expected, &mut ctx);
                }
            }

            assert_eq!(requests.load(Ordering::Relaxed), 1);

            Ok(())
        })
    }

    #[test]
    fn flat_expr() {
        block_on(|handle| async {
            let session = SESSION.clone().with_handle(handle);
            let mut ctx = session.create_execution_ctx();
            let array_ctx = ArrayContext::empty();

            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array =
                PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid).into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    array_ctx,
                    Arc::<TestSegments>::clone(&segments),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            let expr = gt(root(), lit(3i32));
            let result = layout
                .new_reader("".into(), segments, &session, &Default::default())
                .unwrap()
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
            let session = SESSION.clone().with_handle(handle);
            let mut ctx = session.create_execution_ctx();
            let array_ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array =
                PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid).into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    array_ctx,
                    Arc::<TestSegments>::clone(&segments),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            let result = layout
                .new_reader("".into(), segments, &session, &Default::default())
                .unwrap()
                .projection_evaluation(&(2..4), &root(), MaskFuture::new_true(2))
                .unwrap()
                .await
                .unwrap();

            let expected = PrimitiveArray::new(buffer![3i32, 4], Validity::AllValid).into_array();
            assert_arrays_eq!(result, expected, &mut ctx);
        })
    }
}
