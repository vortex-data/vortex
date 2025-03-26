use std::ops::{BitAnd, Range};

use async_trait::async_trait;
use vortex_array::compute::{filter, slice};
use vortex_array::{Array, ArrayRef};
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::{ExprRef, Identity};
use vortex_mask::Mask;

use crate::layouts::flat::reader::{FlatReader, SharedArray};
use crate::{ArrayEvaluation, ExprEvaluator, MaskEvaluation};

#[async_trait]
impl ExprEvaluator for FlatReader {
    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        let row_range = usize::try_from(row_range.start)
            .vortex_expect("RowMask begin must fit within FlatLayout size")
            ..usize::try_from(row_range.end)
                .vortex_expect("RowMask end must fit within FlatLayout size");

        Ok(Box::new(FlatEvaluation {
            array: self.array_future()?,
            row_range,
            expr: expr.clone(),
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        let row_range = usize::try_from(row_range.start)
            .vortex_expect("RowMask begin must fit within FlatLayout size")
            ..usize::try_from(row_range.end)
                .vortex_expect("RowMask end must fit within FlatLayout size");
        Ok(Box::new(FlatEvaluation {
            array: self.array_future()?,
            row_range,
            expr: expr.clone(),
        }))
    }
}

struct FlatEvaluation {
    array: SharedArray,
    row_range: Range<usize>,
    expr: ExprRef,
}

#[async_trait]
impl MaskEvaluation for FlatEvaluation {
    async fn exact(&self, mask: Mask) -> VortexResult<Mask> {
        // Now we await the array .
        let mut array = self.array.clone().await?;

        // Slice the array based on the row mask.
        if self.row_range.start > 0 || self.row_range.end < array.len() {
            array = slice(&array, self.row_range.start, self.row_range.end)?;
        }

        // TODO(ngates): if the input mask is sufficiently sparse, we may want to apply it before
        //  the expression, and then do a Mask::range_intersection.

        // Evaluate the projection expression.
        if !self.expr.as_any().is::<Identity>() {
            log::debug!(
                "Evaluating filter expr over {} true values of {} on array\n{}",
                mask.density(),
                mask.len(),
                array.tree_display(),
            );
            array = self.expr.evaluate(&array)?;
        }

        // Convert the array into a mask.
        let array_mask = Mask::try_from(array.as_ref())?;

        // Intersect the mask with the input mask.
        Ok(mask.bitand(&array_mask))
    }
}

#[async_trait]
impl ArrayEvaluation for FlatEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef> {
        // Now we await the array .
        let mut array = self.array.clone().await?;

        // Slice the array based on the row mask.
        if self.row_range.start > 0 || self.row_range.end < array.len() {
            array = slice(&array, self.row_range.start, self.row_range.end)?;
        }

        // Filter the array based on the row mask.
        if !mask.all_true() {
            array = filter(&array, &mask)?;
        }

        // Evaluate the projection expression.
        if !self.expr.as_any().is::<Identity>() {
            array = self.expr.evaluate(&array)?;
        }

        Ok(array)
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
