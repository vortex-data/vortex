use std::ops::{BitAnd, Range, Sub};

use arrow_buffer::BooleanBufferBuilder;
use async_trait::async_trait;
use itertools::Itertools;
use vortex_error::{VortexError, VortexResult};
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::layouts::stats::reader::{SharedPruningResult, StatsReader};
use crate::{
    ArrayEvaluation, ExprEvaluator, Layout, LayoutReader, MaskEvaluation, PruningEvaluation,
};

impl ExprEvaluator for StatsReader {
    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        log::debug!(
            "Stats pruning evaluation: {} - {}",
            self.layout().name(),
            expr
        );
        let data_eval = self.data_child.pruning_evaluation(row_range, expr)?;

        let Some(pruning_mask_future) = self.pruning_mask_future(expr.clone()) else {
            log::debug!("Stats pruning evaluation: not prune-able {}", expr);
            return Ok(data_eval);
        };

        let zone_range = self.zone_range(row_range);
        let zone_lengths = zone_range
            .clone()
            .map(|zone_idx| {
                // Figure out the range in the mask that corresponds to the zone
                let start =
                    usize::try_from(self.zone_offset(zone_idx).saturating_sub(row_range.start))?;
                let end = usize::try_from(
                    self.zone_offset(zone_idx + 1)
                        .sub(row_range.start)
                        .min(row_range.end - row_range.start),
                )?;
                Ok::<_, VortexError>(end - start)
            })
            .try_collect()?;

        Ok(Box::new(StatsPruningEvaluation {
            layout: self.layout().clone(),
            expr: expr.clone(),
            pruning_mask_future,
            zone_range,
            zone_lengths,
            data_eval,
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        self.data_child.filter_evaluation(row_range, expr)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        // TODO(ngates): there are some projection expressions that we may also be able to
        //  short-circuit with statistics.
        self.data_child.projection_evaluation(row_range, expr)
    }
}

struct StatsPruningEvaluation {
    layout: Layout,
    expr: ExprRef,
    pruning_mask_future: SharedPruningResult,
    // The range of zones that cover the evaluation's row range.
    zone_range: Range<usize>,
    // The lengths of each zone in the zone_range.
    zone_lengths: Vec<usize>,
    // The evaluation of the data child.
    data_eval: Box<dyn PruningEvaluation>,
}

#[async_trait]
impl PruningEvaluation for StatsPruningEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        log::debug!(
            "Invoking stats pruning evaluation {}: {}",
            self.layout.name(),
            self.expr,
        );
        let Some(pruning_mask) = self.pruning_mask_future.clone().await? else {
            // If the expression is not prune-able, we just return the input mask.
            return Ok(mask);
        };

        let mut builder = BooleanBufferBuilder::new(mask.len());
        for (zone_idx, zone_length) in self.zone_range.clone().zip_eq(&self.zone_lengths) {
            builder.append_n(*zone_length, !pruning_mask.value(zone_idx));
        }

        let stats_mask = Mask::from(builder.finish());
        assert_eq!(stats_mask.len(), mask.len(), "Mask length mismatch");

        // Intersect the masks.
        let mut stats_mask = mask.bitand(&stats_mask);

        // Forward to data child for further pruning.
        if !stats_mask.all_false() {
            let data_mask = self.data_eval.invoke(stats_mask.clone()).await?;
            stats_mask = stats_mask.bitand(&data_mask);
        }

        log::debug!(
            "Stats evaluation approx {} - {} (mask = {}) => {}",
            self.layout.name(),
            self.expr,
            mask.density(),
            stats_mask.density(),
        );

        Ok(stats_mask)
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
    use vortex_mask::Mask;

    use crate::layouts::chunked::writer::ChunkedLayoutWriter;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::stats::writer::{StatsLayoutOptions, StatsLayoutWriter};
    use crate::segments::{SegmentSource, TestSegments};
    use crate::writer::LayoutWriterExt;
    use crate::{ExprEvaluator, Layout};

    #[fixture]
    /// Create a stats layout with three chunks of primitive arrays.
    fn stats_layout() -> (ArrayContext, Arc<dyn SegmentSource>, Layout) {
        let ctx = ArrayContext::empty();
        let mut segments = TestSegments::default();
        let layout = StatsLayoutWriter::new(
            ctx.clone(),
            &DType::Primitive(PType::I32, NonNullable),
            ChunkedLayoutWriter::new(
                ctx.clone(),
                DType::Primitive(PType::I32, NonNullable),
                Default::default(),
            )
            .boxed(),
            ArcRef::new_arc(Arc::new(FlatLayoutStrategy::default())),
            StatsLayoutOptions {
                block_size: 3,
                ..Default::default()
            },
        )
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
            Arc<dyn SegmentSource>,
            Layout,
        ),
    ) {
        block_on(async {
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

            assert_eq!(result.len(), 9);
            assert_eq!(result.as_slice::<i32>(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        })
    }

    #[rstest]
    fn test_stats_pruning_mask(
        #[from(stats_layout)] (ctx, segments, layout): (
            ArrayContext,
            Arc<dyn SegmentSource>,
            Layout,
        ),
    ) {
        block_on(async {
            let row_count = layout.row_count();
            let reader = layout.reader(&segments, &ctx).unwrap();

            // Choose a prune-able expression
            let expr = gt(Identity::new_expr(), lit(7));

            let result = reader
                .pruning_evaluation(&(0..row_count), &expr)
                .unwrap()
                .invoke(Mask::new_true(row_count.try_into().unwrap()))
                .await
                .unwrap()
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
