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
use vortex_array::expr::BoundExpression;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::layouts::SharedArrayFuture;
use crate::layouts::flat::FlatLayout;
use crate::layouts::flat::partial::PartialReadPlan;
use crate::layouts::flat::partial::RegisteredPartialRead;
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

#[derive(Clone)]
pub struct FlatReader {
    layout: FlatLayout,
    name: Arc<str>,
    segment_source: Arc<dyn SegmentSource>,
    session: VortexSession,
    partial_plan: Arc<OnceLock<Option<PartialReadPlan>>>,
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
            partial_plan: Arc::new(OnceLock::new()),
        }
    }

    fn register_partial(
        &self,
        row_range: &Range<usize>,
        mask: &Mask,
    ) -> Option<RegisteredPartialRead> {
        if !PartialReadPlan::supports_mask(mask) {
            return None;
        }
        let plan = self
            .partial_plan
            .get_or_init(|| match PartialReadPlan::try_new(&self.layout) {
                Ok(plan) => plan,
                Err(error) => {
                    tracing::debug!("Flat partial-read plan disabled: {error}");
                    None
                }
            });
        plan.as_ref()?.register(
            &self.segment_source,
            self.layout.segment_id(),
            usize::try_from(self.layout.row_count()).ok()?,
            row_range,
            mask,
        )
    }

    /// Register the segment request and return a future that would resolve into the deserialised array.
    fn array_future(&self) -> SharedArrayFuture {
        let row_count =
            usize::try_from(self.layout.row_count()).vortex_expect("row count must fit in usize");

        // We create the segment_fut here to ensure we give the segment reader visibility into
        // how to prioritize this segment, even if the `array` future has already been initialized.
        // This is gross... see the function's TODO for a maybe better solution?
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
        if !mask.partial_reads_allowed() {
            let name = Arc::clone(&self.name);
            let array = self.array_future();
            let expr = expr.clone();
            let session = self.session.clone();

            return Ok(MaskFuture::new(mask.len(), async move {
                let mut array = array.await?;
                let mask = mask.await?;

                if row_range.start > 0 || row_range.end < array.len() {
                    array = array.slice(row_range.clone())?;
                }

                let mask_density = mask.density();
                let array_mask = if mask_density < EXPR_EVAL_THRESHOLD {
                    let array = array.apply_bound(&expr)?;
                    let array = array.filter(mask.clone())?;
                    let mut ctx = session.create_execution_ctx();
                    let array_mask = array.null_as_false().execute(&mut ctx)?;
                    mask.intersect_by_rank(&array_mask)
                } else {
                    let array = array.apply_bound(&expr)?;
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
            }));
        }
        let name = Arc::clone(&self.name);
        let expr = expr.clone();
        let session = self.session.clone();
        let reader = self.clone();
        let partial_reads_allowed = mask.partial_reads_allowed();
        let registered = partial_reads_allowed
            .then(|| mask.upper_bound())
            .flatten()
            .and_then(|upper_bound| self.register_partial(&row_range, upper_bound));
        let eager_array =
            (mask.upper_bound_is_exact() && registered.is_none()).then(|| self.array_future());

        Ok(MaskFuture::new(mask.len(), async move {
            // TODO(ngates): if the mask density is low enough, or if the mask is dense within a range
            //  (as often happens with zone map pruning), then we could slice/filter the array prior
            //  to evaluating the expression.
            let mask = mask.await?;

            if let Some(registered) = registered.or_else(|| {
                partial_reads_allowed
                    .then(|| reader.register_partial(&row_range, &mask))
                    .flatten()
            }) {
                let array = registered
                    .resolve(
                        reader.layout.dtype(),
                        &row_range,
                        &mask,
                        reader.layout.array_ctx(),
                        &session,
                    )
                    .await?;
                let array = array.apply_bound(&expr)?;
                let mut ctx = session.create_execution_ctx();
                let array_mask = array.null_as_false().execute(&mut ctx)?;
                return Ok(mask.intersect_by_rank(&array_mask));
            }

            let mut array = match eager_array {
                Some(array) => array.await?,
                None => reader.array_future().await?,
            };

            // Slice the array based on the row mask.
            if row_range.start > 0 || row_range.end < array.len() {
                array = array.slice(row_range.clone())?;
            }

            let mask_density = mask.density();
            let array_mask = if mask_density < EXPR_EVAL_THRESHOLD {
                // We have the choice to apply the filter or the expression first, we apply the
                // expression first so that it can try pushing down itself and then the filter
                // after this.
                let array = array.apply_bound(&expr)?;
                let array = array.filter(mask.clone())?;
                let mut ctx = session.create_execution_ctx();
                let array_mask = array.null_as_false().execute(&mut ctx)?;

                mask.intersect_by_rank(&array_mask)
            } else {
                // Run over the full array, with a simpler bitand at the end.
                let array = array.apply_bound(&expr)?;
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
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let row_range = usize::try_from(row_range.start)
            .vortex_expect("Row range begin must fit within FlatLayout size")
            ..usize::try_from(row_range.end)
                .vortex_expect("Row range end must fit within FlatLayout size");
        if !mask.partial_reads_allowed() {
            let name = Arc::clone(&self.name);
            let array = self.array_future();
            let expr = expr.clone();

            return Ok(async move {
                trace!("Flat array evaluation {} - {}", name, expr);

                let mut array = array.await?;
                let mask = mask.await?;

                if row_range.start > 0 || row_range.end < array.len() {
                    array = array.slice(row_range.clone())?;
                }
                if !mask.all_true() {
                    array = array.filter(mask)?;
                }
                array = array.apply_bound(&expr)?;
                Ok(array)
            }
            .boxed());
        }
        let name = Arc::clone(&self.name);
        let expr = expr.clone();
        let reader = self.clone();
        let partial_reads_allowed = mask.partial_reads_allowed();
        let registered = partial_reads_allowed
            .then(|| mask.upper_bound())
            .flatten()
            .and_then(|upper_bound| self.register_partial(&row_range, upper_bound));
        let eager_array = ((!partial_reads_allowed || mask.upper_bound_is_exact())
            && registered.is_none())
        .then(|| self.array_future());

        Ok(async move {
            trace!("Flat array evaluation {} - {}", name, expr);

            let mask = mask.await?;

            if let Some(registered) = registered {
                let mut array = registered
                    .resolve(
                        reader.layout.dtype(),
                        &row_range,
                        &mask,
                        reader.layout.array_ctx(),
                        &reader.session,
                    )
                    .await?;
                array = array.apply_bound(&expr)?;
                return Ok(array);
            }

            let mut array = match eager_array {
                Some(array) => array.await?,
                None => reader.array_future().await?,
            };

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
    use std::ops::Range;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use parking_lot::Mutex;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_mask::Mask;

    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::SegmentFuture;
    use crate::segments::SegmentId;
    use crate::segments::SegmentSource;
    use crate::segments::SharedSegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::new_session;

    #[derive(Clone, Default)]
    struct RangedTestSource {
        inner: Arc<TestSegments>,
        ranges: Arc<Mutex<Vec<Range<u64>>>>,
        whole_requests: Arc<AtomicUsize>,
    }

    impl SegmentSource for RangedTestSource {
        fn preferred_read_size(&self) -> Option<u64> {
            Some(16)
        }

        fn segment_len(&self, id: SegmentId) -> Option<u64> {
            self.inner.segment_len(id)
        }

        fn request(&self, id: SegmentId) -> SegmentFuture {
            self.whole_requests.fetch_add(1, Ordering::Relaxed);
            self.inner.request(id)
        }

        fn request_range(&self, id: SegmentId, range: Range<u64>) -> SegmentFuture {
            self.ranges.lock().push(range.clone());
            self.inner.request_range(id, range)
        }
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

    #[test]
    fn sparse_projection_reads_only_virtual_pages() -> VortexResult<()> {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let mut ctx = session.create_execution_ctx();
            let array_ctx = ArrayContext::empty();
            let source = RangedTestSource::default();
            let (ptr, eof) = SequenceId::root().split();
            let array = PrimitiveArray::from_iter(0i32..64).into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    array_ctx.into(),
                    Arc::<TestSegments>::clone(&source.inner),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await?;

            let reader = layout.new_reader(
                "".into(),
                Arc::new(source.clone()),
                &session,
                &Default::default(),
            )?;
            let expr = root().bind(reader.dtype())?;
            let result = reader
                .projection_evaluation(
                    &(0..64),
                    &expr,
                    MaskFuture::ready(Mask::from_indices(64, [1, 10])),
                )?
                .await?;

            let expected = PrimitiveArray::from_iter([1i32, 10]).into_array();
            assert_arrays_eq!(result, expected, &mut ctx);
            assert_eq!(source.whole_requests.load(Ordering::Relaxed), 0);
            assert_eq!(*source.ranges.lock(), [0..16, 32..48]);

            let result = reader
                .projection_evaluation(
                    &(0..64),
                    &expr,
                    MaskFuture::ready(Mask::from_indices(64, [1, 10])),
                )?
                .await?;
            assert_arrays_eq!(result, expected, &mut ctx);
            assert_eq!(
                *source.ranges.lock(),
                [0..16, 32..48, 0..16, 32..48],
                "separate evaluations must not retain page data"
            );
            Ok(())
        })
    }

    #[test]
    fn dense_projection_chooses_whole_segment() -> VortexResult<()> {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let mut ctx = session.create_execution_ctx();
            let array_ctx = ArrayContext::empty();
            let source = RangedTestSource::default();
            let (ptr, eof) = SequenceId::root().split();
            let array = PrimitiveArray::from_iter(0i32..64).into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    array_ctx.into(),
                    Arc::<TestSegments>::clone(&source.inner),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await?;

            let reader = layout.new_reader(
                "".into(),
                Arc::new(source.clone()),
                &session,
                &Default::default(),
            )?;
            let expr = root().bind(reader.dtype())?;
            let mask = Mask::from_indices(64, (0..64).step_by(2));
            let result = reader
                .projection_evaluation(&(0..64), &expr, MaskFuture::ready(mask.clone()))?
                .await?;

            assert_arrays_eq!(result, array.filter(mask)?, &mut ctx);
            assert_eq!(source.whole_requests.load(Ordering::Relaxed), 1);
            assert!(source.ranges.lock().is_empty());
            Ok(())
        })
    }

    #[test]
    fn filter_and_projection_share_pages_while_scan_is_in_flight() -> VortexResult<()> {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let mut ctx = session.create_execution_ctx();
            let array_ctx = ArrayContext::empty();
            let source = RangedTestSource::default();
            let (ptr, eof) = SequenceId::root().split();
            let array = PrimitiveArray::from_iter(0i32..64).into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    array_ctx.into(),
                    Arc::<TestSegments>::clone(&source.inner),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await?;

            let reader = layout.new_reader(
                "".into(),
                Arc::new(SharedSegmentSource::new(source.clone())),
                &session,
                &Default::default(),
            )?;
            let projection_expr = root().bind(reader.dtype())?;
            let filter_expr = gt(root(), lit(-1i32)).bind(reader.dtype())?;
            let mask = Mask::from_indices(64, [1, 10]);

            let filter = reader.filter_evaluation(
                &(0..64),
                &filter_expr,
                MaskFuture::ready(mask.clone()),
            )?;
            let projection = reader.projection_evaluation(
                &(0..64),
                &projection_expr,
                MaskFuture::ready(mask),
            )?;

            let (filter_mask, result) = futures::try_join!(filter, projection)?;
            assert_eq!(
                filter_mask.indices(),
                Mask::from_indices(64, [1, 10]).indices()
            );
            let expected = PrimitiveArray::from_iter([1i32, 10]).into_array();
            assert_arrays_eq!(result, expected, &mut ctx);
            assert_eq!(
                *source.ranges.lock(),
                [0..16, 32..48],
                "one in-flight request should serve filter and projection"
            );
            Ok(())
        })
    }
}
