use std::ops::{BitAnd, Range};

use async_trait::async_trait;
use vortex_array::compute::filter;
use vortex_array::{Array, ArrayRef};
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::{ExprRef, Identity};
use vortex_mask::Mask;

use crate::layouts::SharedArrayFuture;
use crate::layouts::flat::reader::FlatReader;
use crate::{
    ArrayEvaluation, ExprEvaluator, Layout, LayoutReader, MaskEvaluation, NoOpPruningEvaluation,
    PruningEvaluation,
};

/// The threshold of mask density below which we will evaluate the expression only over the
/// selected rows, and above which we evaluate the expression over all rows and then select
/// after.
// TODO(ngates): more experimentation is needed, and this should probably be dynamic based on the
//  actual expression? Perhaps all expressions are given a selection mask to decide for themselves?
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

impl ExprEvaluator for FlatReader {
    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        Ok(Box::new(NoOpPruningEvaluation))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        let row_range = usize::try_from(row_range.start)
            .vortex_expect("Row range begin must fit within FlatLayout size")
            ..usize::try_from(row_range.end)
                .vortex_expect("Row range end must fit within FlatLayout size");

        Ok(Box::new(FlatEvaluation {
            layout: self.layout().clone(),
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
            .vortex_expect("Row range begin must fit within FlatLayout size")
            ..usize::try_from(row_range.end)
                .vortex_expect("Row range end must fit within FlatLayout size");
        Ok(Box::new(FlatEvaluation {
            layout: self.layout().clone(),
            array: self.array_future()?,
            row_range,
            expr: expr.clone(),
        }))
    }
}

struct FlatEvaluation {
    layout: Layout,
    array: SharedArrayFuture,
    row_range: Range<usize>,
    expr: ExprRef,
}

#[async_trait]
impl MaskEvaluation for FlatEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        // TODO(ngates): if the mask density is low enough, or if the mask is dense within a range
        //  (as often happens with zone map pruning), then we could slice/filter the array prior
        //  to evaluating the expression.

        // Now we await the array .
        let mut array = self.array.clone().await?;

        // Slice the array based on the row mask.
        if self.row_range.start > 0 || self.row_range.end < array.len() {
            array = array.slice(self.row_range.start, self.row_range.end)?;
        }

        // TODO(ngates): the mask may actually be dense within a range, as is often the case when
        //  we have approximate mask results from a zone map. In which case we could look at
        //  the true_count between the mask's first and last true positions.
        // TODO(ngates): we could also track runtime statistics about whether it's worth selecting
        //   or not.
        let array_mask = if mask.density() < EXPR_EVAL_THRESHOLD {
            // Evaluate only the selected rows of the mask.
            array = filter(&array, &mask)?;
            let array_mask = Mask::try_from(self.expr.evaluate(&array)?.as_ref())?;
            mask.intersect_by_rank(&array_mask)
        } else {
            // Evaluate all rows, avoiding the more expensive rank intersection.
            array = self.expr.evaluate(&array)?;
            let array_mask = Mask::try_from(array.as_ref())?;
            mask.bitand(&array_mask)
        };

        log::debug!(
            "Flat mask evaluation {} - {} (mask = {}) => {}",
            self.layout.name(),
            self.expr,
            mask.density(),
            array_mask.density(),
        );

        Ok(array_mask)
    }
}

#[async_trait]
impl ArrayEvaluation for FlatEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef> {
        log::debug!(
            "Flat array evaluation {} - {} (mask = {})",
            self.layout.name(),
            self.expr,
            mask.density(),
        );

        // Now we await the array .
        let mut array = self.array.clone().await?;

        // Slice the array based on the row mask.
        if self.row_range.start > 0 || self.row_range.end < array.len() {
            array = array.slice(self.row_range.start, self.row_range.end)?;
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
    use vortex_array::{ArrayContext, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_expr::{Identity, gt, ident, lit};
    use vortex_mask::Mask;

    use crate::ExprEvaluator;
    use crate::layouts::flat::writer::FlatLayoutWriter;
    use crate::segments::{SegmentSource, TestSegments};
    use crate::writer::LayoutWriterExt;

    #[test]
    fn flat_identity() {
        block_on(async {
            let ctx = ArrayContext::empty();
            let mut segments = TestSegments::default();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout =
                FlatLayoutWriter::new(ctx.clone(), array.dtype().clone(), Default::default())
                    .push_one(&mut segments, array.to_array())
                    .unwrap();

            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

            let result = layout
                .reader(&segments, &ctx)
                .unwrap()
                .projection_evaluation(&(0..layout.row_count()), &Identity::new_expr())
                .unwrap()
                .invoke(Mask::new_true(layout.row_count().try_into().unwrap()))
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

            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

            let expr = gt(Identity::new_expr(), lit(3i32));
            let result = layout
                .reader(&segments, &ctx)
                .unwrap()
                .projection_evaluation(&(0..layout.row_count()), &expr)
                .unwrap()
                .invoke(Mask::new_true(layout.row_count().try_into().unwrap()))
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
                    .push_one(&mut segments, array.to_array())
                    .unwrap();

            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

            let result = layout
                .reader(&segments, &ctx)
                .unwrap()
                .projection_evaluation(&(2..4), &ident())
                .unwrap()
                .invoke(Mask::new_true(2))
                .await
                .unwrap()
                .to_primitive()
                .unwrap();

            assert_eq!(result.as_slice::<i32>(), &[3, 4],);
        })
    }
}
