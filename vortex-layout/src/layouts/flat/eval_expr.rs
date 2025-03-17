use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::compute::{filter, slice};
use vortex_array::serde::ArrayParts;
use vortex_array::{Array, ArrayRef};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_expr::{ExprRef, Identity};

use crate::layouts::flat::reader::FlatReader;
use crate::{ExprEvaluator, MaskFuture, RowMask};

#[async_trait]
impl ExprEvaluator for FlatReader {
    fn evaluate_expr2(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<Option<ArrayRef>>>> {
        if row_range.end > self.layout.row_count() {
            vortex_bail!("Row range exceeds layout row count");
        }

        let segment_id = self
            .layout
            .segment_id(0)
            .ok_or_else(|| vortex_err!("FlatLayout missing segment"))?;

        let row_count = usize::try_from(self.layout.row_count())
            .vortex_expect("FlatLayout row count does not fit within usize");

        let array_future = self
            .array2
            .get_or_init(|| {
                let segment_reader = self.segment_reader.clone();
                let ctx = self.ctx.clone();
                let dtype = self.layout.dtype().clone();

                // We create, but don't await, the segment request. This allows for prefetching.
                let segment_fut = segment_reader.get(segment_id);

                async move {
                    let segment = segment_fut.await?;
                    ArrayParts::try_from(segment)?
                        .decode(&ctx, dtype.clone(), row_count)
                        .map_err(|e| Arc::new(e))
                }
                .boxed()
                .shared()
            })
            .clone();

        let row_range = row_range.clone();
        let expr = expr.clone();

        Ok(Box::pin(async move {
            // Wait for the mask, and short-circuit if it's all false.
            let mask = mask.await?;
            if mask.all_false() {
                return Ok(None);
            }

            // Now we await the array .
            let mut array = array_future.await?;

            // Convert the range into an usize now that we expect it to fit.
            let start = usize::try_from(row_range.start)
                .vortex_expect("RowMask begin must fit within FlatLayout size");
            let end = usize::try_from(row_range.end)
                .vortex_expect("RowMask end must fit within FlatLayout size");
            let row_range = start..end;

            // Slice the array based on the row mask.
            if start > 0 || row_range.end < array.len() {
                array = slice(&array, start, end)?;
            }

            // Filter the array based on the row mask.
            if !mask.all_true() {
                array = filter(&array, &mask)?;
            }

            // Evaluate the projection expression.
            if !expr.as_any().is::<Identity>() {
                array = expr.evaluate(&array)?;
            }

            Ok(Some(array))
        }))
    }

    async fn evaluate_expr(
        self: &Self,
        row_mask: RowMask,
        expr: ExprRef,
    ) -> VortexResult<ArrayRef> {
        assert!(row_mask.true_count() > 0);

        let mut array = self.array().await?.clone();

        // TODO(ngates): what's the best order to apply the filter mask / expression?
        let begin = usize::try_from(row_mask.begin())
            .vortex_expect("RowMask begin must fit within FlatLayout size");

        // Slice the array based on the row mask.
        if begin > 0 || (begin + row_mask.len()) < array.len() {
            array = slice(&array, begin, begin + row_mask.len())?;
        }

        // Filter the array based on the row mask.
        if !row_mask.filter_mask().all_true() {
            array = filter(&array, row_mask.filter_mask())?;
        }

        // Evaluate the projection expression.
        if !expr.as_any().is::<Identity>() {
            array = expr.evaluate(&array)?;
        }

        Ok(array)
    }

    async fn refine_mask(&self, row_mask: RowMask, _expr: ExprRef) -> VortexResult<RowMask> {
        // No cheap pruning for us to do without fetching data.
        Ok(row_mask)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_buffer::BooleanBuffer;
    use futures::executor::block_on;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayContext, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_expr::{Identity, gt, ident, lit};

    use crate::RowMask;
    use crate::layouts::flat::writer::FlatLayoutWriter;
    use crate::segments::test::TestSegments;
    use crate::writer::LayoutWriterExt;

    #[test]
    fn flat_identity() {
        block_on(async {
            let ctx = ArrayContext::empty();
            let mut segments = TestSegments::default();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout =
                FlatLayoutWriter::new(ctx.clone(), array.dtype().clone(), Default::default())
                    .push_one(&mut segments, array.to_array().into_array())
                    .unwrap();

            let result = layout
                .reader(Arc::new(segments), ctx)
                .unwrap()
                .evaluate_expr(
                    RowMask::new_valid_between(0, layout.row_count()),
                    Identity::new_expr(),
                )
                .await
                .unwrap()
                .to_primitive()
                .unwrap();

            assert_eq!(array.as_slice::<i32>(), result.as_slice::<i32>());
        })
    }

    #[test]
    fn flat_expr() {
        block_on(async {
            let ctx = ArrayContext::empty();
            let mut segments = TestSegments::default();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout =
                FlatLayoutWriter::new(ctx.clone(), array.dtype().clone(), Default::default())
                    .push_one(&mut segments, array.into_array())
                    .unwrap();

            let expr = gt(Identity::new_expr(), lit(3i32));
            let result = layout
                .reader(Arc::new(segments), ctx)
                .unwrap()
                .evaluate_expr(RowMask::new_valid_between(0, layout.row_count()), expr)
                .await
                .unwrap()
                .to_bool()
                .unwrap();

            assert_eq!(
                &BooleanBuffer::from_iter([false, false, false, true, true]),
                result.boolean_buffer()
            );
        })
    }

    #[test]
    fn flat_unaligned_row_mask() {
        block_on(async {
            let ctx = ArrayContext::empty();
            let mut segments = TestSegments::default();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout =
                FlatLayoutWriter::new(ctx.clone(), array.dtype().clone(), Default::default())
                    .push_one(&mut segments, array.to_array().into_array())
                    .unwrap();

            let result = layout
                .reader(Arc::new(segments), ctx)
                .unwrap()
                .evaluate_expr(RowMask::new_valid_between(2, 4), ident())
                .await
                .unwrap()
                .to_primitive()
                .unwrap();

            assert_eq!(result.as_slice::<i32>(), &[3, 4],);
        })
    }
}
