use std::ops::{BitAnd, Range, Sub};

use arrow_buffer::BooleanBufferBuilder;
use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::{FutureExt, TryFutureExt};
use vortex_array::ArrayRef;
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::layouts::stats::reader::StatsReader;
use crate::{ExprEvaluator, LayoutReader, MaskFuture, RowMask};

#[async_trait]
impl ExprEvaluator for StatsReader {
    fn evaluate_expr2(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> BoxFuture<'static, VortexResult<Option<ArrayRef>>> {
        let Some(pruning_mask) = self.pruning_mask(expr.clone()) else {
            // TODO(ngates): we should check if the predicate can be evaluated with the stats
            //  that are present.

            // Otherwise, we just delegate to the data child.
            return self.data_child.evaluate_expr2(row_range, expr, mask);
        };

        let row_range2 = row_range.clone();
        let zone_range = self.zone_range(&row_range);
        let row_count = self.layout().row_count();
        let zone_len = self.zone_len;

        // Refine the mask based on the result of the pruning mask.
        let mask: MaskFuture = pruning_mask
            .and_then(move |pruning_mask| {
                let row_range = row_range2.clone();
                let zone_range = zone_range.clone();

                async move {
                    // Await the original row mask.
                    let mask = mask.await?;

                    let Some(pruning_mask) = pruning_mask else {
                        // If the pruning mask couldn't be evaluated, we return the original mask.
                        return Ok(mask);
                    };

                    // A function to compute the offset of a given zone.
                    let zone_offset =
                        |zone_idx: usize| ((zone_idx * zone_len) as u64).min(row_count);

                    let mut builder = BooleanBufferBuilder::new(mask.len());
                    for zone_idx in zone_range {
                        // Figure out the range in the mask that corresponds to the zone
                        let start =
                            usize::try_from(zone_offset(zone_idx).saturating_sub(row_range.start))
                                .vortex_expect("Invalid usize row range");
                        let end = usize::try_from(
                            zone_offset(zone_idx + 1)
                                .sub(row_range.start)
                                .min(mask.len() as u64),
                        )
                        .vortex_expect("Invalid usize row range");
                        builder.append_n(end - start, !pruning_mask.value(zone_idx));
                    }

                    let pruning_row_mask = Mask::from(builder.finish());
                    assert_eq!(pruning_row_mask.len(), mask.len(), "Mask length mismatch");

                    // Apply the mask to the row mask
                    Ok(mask.bitand(&pruning_row_mask))
                }
            })
            .boxed()
            .shared();

        self.data_child.evaluate_expr2(&row_range, &expr, mask)
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
