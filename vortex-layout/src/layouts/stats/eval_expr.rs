use std::ops::{BitAnd, Sub};

use arrow_buffer::BooleanBufferBuilder;
use async_trait::async_trait;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::layouts::stats::reader::StatsReader;
use crate::{ExprEvaluator, RowMask};

#[async_trait]
impl ExprEvaluator for StatsReader {
    async fn evaluate_expr(
        self: &Self,
        row_mask: RowMask,
        expr: ExprRef,
    ) -> VortexResult<ArrayRef> {
        self.child().evaluate_expr(row_mask, expr).await
    }

    async fn refine_mask(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<RowMask> {
        // Compute the pruning mask
        let Some(pruning_mask) = self.pruning_mask(&expr).await? else {
            // If there is no pruning mask, then we can't prune anything!
            log::debug!(
                "Cannot prune {} in chunked reader, returning mask {}",
                expr,
                row_mask.filter_mask().density()
            );
            return Ok(row_mask);
        };

        log::debug!(
            "Pruning mask for {} {}..{}: {:?}",
            expr,
            row_mask.begin(),
            row_mask.end(),
            pruning_mask
        );

        let mut builder = BooleanBufferBuilder::new(row_mask.len());

        for block_idx in self.block_range(&row_mask) {
            // Figure out the range in the mask that corresponds to the block
            let start = usize::try_from(
                self.block_offset(block_idx)
                    .saturating_sub(row_mask.begin()),
            )?;
            let end = usize::try_from(
                self.block_offset(block_idx + 1)
                    .sub(row_mask.begin())
                    .min(row_mask.len() as u64),
            )?;
            builder.append_n(end - start, !pruning_mask.value(block_idx));
        }

        let mask = Mask::from(builder.finish());
        assert_eq!(mask.len(), row_mask.len(), "Mask length mismatch");

        // Apply the mask to the row mask
        let mask = row_mask.filter_mask().bitand(&mask);

        Ok(RowMask::new(mask, row_mask.begin()))
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
