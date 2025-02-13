use async_trait::async_trait;
use vortex_array::Array;
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::{ExprRef, Identity};

use crate::layouts::flat::reader::FlatReader;
use crate::scan::ScanTask;
use crate::{ExprEvaluator, RowMask};

#[async_trait]
impl ExprEvaluator for FlatReader {
    async fn evaluate_expr(self: &Self, row_mask: RowMask, expr: ExprRef) -> VortexResult<Array> {
        assert!(row_mask.true_count() > 0);

        let array = self.array().await?.clone();

        // TODO(ngates): what's the best order to apply the filter mask / expression?
        let begin = usize::try_from(row_mask.begin())
            .vortex_expect("RowMask begin must fit within FlatLayout size");

        let mut tasks = Vec::with_capacity(3);

        // Slice the array based on the row mask.
        if begin > 0 || (begin + row_mask.len()) < array.len() {
            tasks.push(ScanTask::Slice(begin..begin + row_mask.len()));
        }

        // Filter the array based on the row mask.
        if !row_mask.filter_mask().all_true() {
            tasks.push(ScanTask::Filter(row_mask.filter_mask().clone()));
        }

        // Evaluate the projection expression.
        if !expr.as_any().is::<Identity>() {
            tasks.push(ScanTask::Expr(expr));
        }

        self.executor().evaluate(&array, &tasks).await
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
    use vortex_array::array::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArray, IntoArrayVariant};
    use vortex_buffer::buffer;
    use vortex_expr::{gt, ident, lit, Identity};

    use crate::layouts::flat::writer::FlatLayoutWriter;
    use crate::scan::ScanExecutor;
    use crate::segments::test::TestSegments;
    use crate::writer::LayoutWriterExt;
    use crate::RowMask;

    #[test]
    fn flat_identity() {
        block_on(async {
            let mut segments = TestSegments::default();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout = FlatLayoutWriter::new(array.dtype().clone(), Default::default())
                .push_one(&mut segments, array.clone().into_array())
                .unwrap();

            let result = layout
                .reader(ScanExecutor::inline(Arc::new(segments)), Default::default())
                .unwrap()
                .evaluate_expr(
                    RowMask::new_valid_between(0, layout.row_count()),
                    Identity::new_expr(),
                )
                .await
                .unwrap()
                .into_primitive()
                .unwrap();

            assert_eq!(array.as_slice::<i32>(), result.as_slice::<i32>());
        })
    }

    #[test]
    fn flat_expr() {
        block_on(async {
            let mut segments = TestSegments::default();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout = FlatLayoutWriter::new(array.dtype().clone(), Default::default())
                .push_one(&mut segments, array.into_array())
                .unwrap();

            let expr = gt(Identity::new_expr(), lit(3i32));
            let result = layout
                .reader(ScanExecutor::inline(Arc::new(segments)), Default::default())
                .unwrap()
                .evaluate_expr(RowMask::new_valid_between(0, layout.row_count()), expr)
                .await
                .unwrap()
                .into_bool()
                .unwrap();

            assert_eq!(
                BooleanBuffer::from_iter([false, false, false, true, true]),
                result.boolean_buffer()
            );
        })
    }

    #[test]
    fn flat_unaligned_row_mask() {
        block_on(async {
            let mut segments = TestSegments::default();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout = FlatLayoutWriter::new(array.dtype().clone(), Default::default())
                .push_one(&mut segments, array.clone().into_array())
                .unwrap();

            let result = layout
                .reader(ScanExecutor::inline(Arc::new(segments)), Default::default())
                .unwrap()
                .evaluate_expr(RowMask::new_valid_between(2, 4), ident())
                .await
                .unwrap()
                .into_primitive()
                .unwrap();

            assert_eq!(result.as_slice::<i32>(), &[3, 4],);
        })
    }
}
