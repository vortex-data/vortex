use async_trait::async_trait;
use vortex_array::compute::{filter, slice};
use vortex_array::{Array, ArrayRef};
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::{ExprRef, Identity};

use crate::layouts::flat::reader::FlatReader;
use crate::{ExprEvaluator, RowMask};

#[async_trait]
impl ExprEvaluator for FlatReader {
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

    async fn prune_mask(&self, row_mask: RowMask, _expr: ExprRef) -> VortexResult<RowMask> {
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
