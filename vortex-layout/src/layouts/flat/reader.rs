// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

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

use crate::LayoutReader;
use crate::layouts::SharedArrayFuture;
use crate::layouts::flat::FlatLayout;
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
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        splits.insert(row_range.start + self.layout.row_count);
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

            let array_mask = if mask.density() < EXPR_EVAL_THRESHOLD {
                // We have the choice to apply the filter or the expression first, we apply the
                // expression first so that it can try pushing down itself and then the filter
                // after this.
                let array = array.apply(&expr)?;
                let array = array.filter(mask.clone())?;
                let mut ctx = session.create_execution_ctx();
                let array_mask = array.execute::<Mask>(&mut ctx)?;

                mask.intersect_by_rank(&array_mask)
            } else {
                // Run over the full array, with a simpler bitand at the end.
                let array = array.apply(&expr)?;
                let mut ctx = session.create_execution_ctx();
                let array_mask = array.execute::<Mask>(&mut ctx)?;

                mask.bitand(&array_mask)
            };

            trace!(
                "Flat mask evaluation {} - {} (mask = {}) => {}",
                name,
                expr,
                mask.density(),
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

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
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

    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;

    #[test]
    fn flat_identity() -> VortexResult<()> {
        block_on(|handle| async {
            let session = SESSION.clone().with_handle(handle);
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array =
                PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid).into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
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
                .new_reader("".into(), segments, &SESSION)?
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &root(),
                    MaskFuture::new_true(layout.row_count().try_into()?),
                )?
                .await?;

            assert_arrays_eq!(result, array);

            Ok(())
        })
    }

    #[test]
    fn flat_expr() {
        block_on(|handle| async {
            let session = SESSION.clone().with_handle(handle);
            let ctx = ArrayContext::empty();

            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array =
                PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid).into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    Arc::<TestSegments>::clone(&segments),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            let expr = gt(root(), lit(3i32));
            let result = layout
                .new_reader("".into(), segments, &SESSION)
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
            assert_arrays_eq!(result, expected);
        })
    }

    #[test]
    fn flat_unaligned_row_mask() {
        block_on(|handle| async {
            let session = SESSION.clone().with_handle(handle);
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array =
                PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid).into_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    Arc::<TestSegments>::clone(&segments),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            let result = layout
                .new_reader("".into(), segments, &SESSION)
                .unwrap()
                .projection_evaluation(&(2..4), &root(), MaskFuture::new_true(2))
                .unwrap()
                .await
                .unwrap();

            let expected = PrimitiveArray::new(buffer![3i32, 4], Validity::AllValid).into_array();
            assert_arrays_eq!(result, expected);
        })
    }
}
