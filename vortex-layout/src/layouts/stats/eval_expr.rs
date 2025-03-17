use std::ops::Range;

use async_trait::async_trait;
use futures::future::BoxFuture;
use vortex_array::arrays::ConstantArray;
use vortex_array::{Array, ArrayRef};
use vortex_error::VortexResult;
use vortex_expr::ExprRef;

use crate::layouts::stats::reader::StatsReader;
use crate::{ExprEvaluator, MaskFuture, RowMask};

#[async_trait]
impl ExprEvaluator for StatsReader {
    fn evaluate_expr2(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<Option<ArrayRef>>>> {
        let Some(pruning_mask) = self.pruning_mask(expr.clone()) else {
            // TODO(ngates): we should check if the predicate can be evaluated with the stats
            //  that are present.

            // Otherwise, we just delegate to the data child.
            return self.data_child.evaluate_expr2(row_range, expr, mask);
        };

        let zone_range = self.zone_range(&row_range);

        // We create an "un-pruned" future to ensure visibility into pre-fetching, although we
        // may never await this.
        let result = self
            .data_child
            .evaluate_expr2(&row_range, &expr, mask.clone())?;

        Ok(Box::pin(async move {
            let mask = mask.clone().await?;

            if let Some(pruning_mask) = pruning_mask.await? {
                if zone_range
                    .clone()
                    .all(|zone_idx| pruning_mask.value(zone_idx))
                {
                    // If all zones covering the row range are pruned, we can return a constant
                    // false response.
                    return Ok(Some(
                        ConstantArray::new(false, mask.true_count()).into_array(),
                    ));
                }
            }

            // Otherwise, we must delegate to the child.
            result.await
        }))
    }

    async fn evaluate_expr(
        self: &Self,
        row_mask: RowMask,
        expr: ExprRef,
    ) -> VortexResult<ArrayRef> {
        self.data_child.evaluate_expr(row_mask, expr).await
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use futures::executor::block_on;
    use rstest::{fixture, rstest};
    use vortex_array::arcref::ArcRef;
    use vortex_array::{Array, ArrayContext, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, PType};
    use vortex_expr::{Identity, gt, lit};

    use crate::layouts::chunked::writer::ChunkedLayoutWriter;
    use crate::layouts::flat::FlatLayout;
    use crate::layouts::stats::writer::{StatsLayoutOptions, StatsLayoutWriter};
    use crate::segments::AsyncSegmentReader;
    use crate::segments::test::TestSegments;
    use crate::writer::LayoutWriterExt;
    use crate::{ExprEvaluator, Layout, RowMask};

    #[fixture]
    /// Create a stats layout with three chunks of primitive arrays.
    fn stats_layout() -> (ArrayContext, Arc<dyn AsyncSegmentReader>, Layout) {
        let ctx = ArrayContext::empty();
        let mut segments = TestSegments::default();
        let layout = StatsLayoutWriter::try_new(
            ctx.clone(),
            &DType::Primitive(PType::I32, NonNullable),
            ChunkedLayoutWriter::new(
                ctx.clone(),
                &DType::Primitive(PType::I32, NonNullable),
                Default::default(),
            )
            .boxed(),
            ArcRef::new_ref(&FlatLayout),
            StatsLayoutOptions {
                block_size: 3,
                ..Default::default()
            },
        )
        .unwrap()
        .push_all(
            &mut segments,
            [
                Ok(buffer![1, 2, 3].into_array()),
                Ok(buffer![4, 5, 6].into_array()),
                Ok(buffer![7, 8, 9].into_array()),
            ],
        )
        .unwrap();
        (ctx, Arc::new(segments), layout)
    }

    #[rstest]
    fn test_stats_evaluator(
        #[from(stats_layout)] (ctx, segments, layout): (
            ArrayContext,
            Arc<dyn AsyncSegmentReader>,
            Layout,
        ),
    ) {
        block_on(async {
            let result = layout
                .reader(segments, ctx)
                .unwrap()
                .evaluate_expr(
                    RowMask::new_valid_between(0, layout.row_count()),
                    Identity::new_expr(),
                )
                .await
                .unwrap()
                .to_primitive()
                .unwrap();

            assert_eq!(result.len(), 9);
            assert_eq!(result.as_slice::<i32>(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        })
    }

    #[rstest]
    fn test_stats_pruning_mask(
        #[from(stats_layout)] (ctx, segments, layout): (
            ArrayContext,
            Arc<dyn AsyncSegmentReader>,
            Layout,
        ),
    ) {
        block_on(async {
            let row_count = layout.row_count();
            let reader = layout.reader(segments, ctx).unwrap();

            // Choose a prune-able expression
            let expr = gt(Identity::new_expr(), lit(7));

            let result = reader
                .refine_mask(RowMask::new_valid_between(0, row_count), expr.clone())
                .await
                .unwrap()
                .filter_mask()
                .to_boolean_buffer()
                .iter()
                .collect::<Vec<_>>();

            assert_eq!(
                result.as_slice(),
                &[false, false, false, false, false, false, true, true, true]
            );
        })
    }
}
