// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::{BitAnd, Range};
use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use vortex_array::compute::filter;
use vortex_array::pipeline::operators::MaskFuture;
use vortex_array::pipeline::{
    N, export_canonical_pipeline_expr, export_canonical_pipeline_expr_offset,
};
use vortex_array::serde::ArrayParts;
use vortex_array::stats::Precision;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_dtype::{DType, FieldMask, Nullability};
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap as _};
use vortex_expr::{ExprRef, Scope, VortexExprExt, is_root};
use vortex_mask::Mask;

use crate::layouts::SharedArrayFuture;
use crate::layouts::flat::FlatLayout;
use crate::segments::SegmentSource;
use crate::{
    ArrayEvaluation, LayoutReader, MaskEvaluation, NoOpPruningEvaluation, PruningEvaluation,
};

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
}

impl FlatReader {
    pub(crate) fn new(
        layout: FlatLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
    ) -> Self {
        Self {
            layout,
            name,
            segment_source,
        }
    }

    /// Register the segment request and return a future that would resolve into the deserialised array.
    fn array_future(&self) -> SharedArrayFuture {
        let row_count = usize::try_from(self.layout.row_count()).vortex_unwrap();

        // We create the segment_fut here to ensure we give the segment reader visibility into
        // how to prioritize this segment, even if the `array` future has already been initialized.
        // This is gross... see the function's TODO for a maybe better solution?
        let segment_fut = self.segment_source.request(self.layout.segment_id());

        let ctx = self.layout.ctx.clone();
        let dtype = self.layout.dtype().clone();
        async move {
            let segment = segment_fut.await?;
            ArrayParts::try_from(segment)?
                .decode(&ctx, &dtype, row_count)
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

    fn row_count(&self) -> Precision<u64> {
        Precision::Exact(self.layout.row_count())
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        splits.insert(row_offset + self.layout.row_count());
        Ok(())
    }

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
            name: self.name.clone(),
            array: self.array_future(),
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
            name: self.name.clone(),
            array: self.array_future(),
            row_range,
            expr: expr.clone(),
        }))
    }
}

struct FlatEvaluation {
    name: Arc<str>,
    array: SharedArrayFuture,
    row_range: Range<usize>,
    expr: ExprRef,
}

#[async_trait]
impl MaskEvaluation for FlatEvaluation {
    async fn invoke(&self, mask: MaskFuture) -> VortexResult<Mask> {
        // TODO(ngates): if the mask density is low enough, or if the mask is dense within a range
        //  (as often happens with zone map pruning), then we could slice/filter the array prior
        //  to evaluating the expression.
        let mut array = self.array.clone().await?;
        let mask = mask.await?;

        if let Some(array) =
            try_evaluate_using_operator(self.row_range.clone(), &array, &self.expr, &mask)?
        {
            let array_mask = array.try_to_mask_fill_null_false()?;
            let mask = mask.intersect_by_rank(&array_mask);
            return Ok(mask);
        }

        // Slice the array based on the row mask.
        if self.row_range.start > 0 || self.row_range.end < array.len() {
            array = array.slice(self.row_range.clone());
        }

        // TODO(ngates): the mask may actually be dense within a range, as is often the case when
        //  we have approximate mask results from a zone map. In which case we could look at
        //  the true_count between the mask's first and last true positions.
        // TODO(ngates): we could also track runtime statistics about whether it's worth selecting
        //   or not.
        let array_mask = if mask.density() < EXPR_EVAL_THRESHOLD {
            // Evaluate only the selected rows of the mask.
            array = filter(&array, &mask)?;
            // TODO(joe): fixme casting null to false is *VERY* unsound, if the expression in the filter
            // can inspect nulls (e.g. `is_null`).
            // you will need to call the array evaluation instead of the mask evaluation.
            let array_mask = self
                .expr
                .evaluate(&Scope::new(array))?
                .try_to_mask_fill_null_false()?;
            mask.intersect_by_rank(&array_mask)
        } else {
            // Evaluate all rows, avoiding the more expensive rank intersection.
            array = self.expr.evaluate(&Scope::new(array))?;
            let array_mask = array.try_to_mask_fill_null_false()?;
            mask.bitand(&array_mask)
        };

        log::debug!(
            "Flat mask evaluation {} - {} (mask = {}) => {}",
            self.name,
            self.expr,
            mask.density(),
            array_mask.density(),
        );

        Ok(array_mask)
    }
}

#[async_trait]
impl ArrayEvaluation for FlatEvaluation {
    async fn invoke(&self, mask: MaskFuture) -> VortexResult<ArrayRef> {
        log::debug!("Flat array evaluation {} - {}", self.name, self.expr);

        let mut array = self.array.clone().await?;
        let mask = mask.await?;

        if let Some(array) =
            try_evaluate_using_operator(self.row_range.clone(), &array, &self.expr, &mask)?
        {
            return Ok(array);
        }

        // Slice the array based on the row mask.
        if self.row_range.start > 0 || self.row_range.end < array.len() {
            array = array.slice(self.row_range.clone());
        }

        // Filter the array based on the row mask.
        if !mask.all_true() {
            array = filter(&array, &mask)?;
        }

        // Evaluate the projection expression.
        if !is_root(&self.expr) {
            array = self.expr.evaluate(&Scope::new(array))?;
        }

        Ok(array)
    }
}

fn try_evaluate_using_operator(
    row_range: Range<usize>,
    array: &ArrayRef,
    expr: &ExprRef,
    mask: &Mask,
) -> VortexResult<Option<ArrayRef>> {
    let Some(operator) = expr.to_operator(array)? else {
        return Ok(None);
    };

    let return_type = expr.return_dtype(array.dtype())?;
    if !matches!(
        return_type,
        DType::Primitive(_, Nullability::NonNullable) | DType::Bool(Nullability::NonNullable)
    ) {
        return Ok(None);
    }

    let result = if row_range.start % N != 0 {
        // If the start is not a multiple of PIPELINE_STEP_COUNT, then we need to slice
        // we could do mask offsets instead, but this case is rare, due to split building.
        let array = array.slice(row_range.clone());
        let operator = expr
            .to_operator(array.as_ref())?
            .vortex_expect("already converted");
        export_canonical_pipeline_expr(
            &return_type,
            row_range.end - row_range.start,
            operator.as_ref(),
            mask,
        )?
        .into_array()
    } else {
        log::trace!(
            "ArrayEvaluation: export_canonical_pipeline_expr_offset {:?}",
            operator
        );
        export_canonical_pipeline_expr_offset(
            &return_type,
            row_range.start / N,
            row_range.end - row_range.start,
            operator.as_ref(),
            mask,
        )?
        .into_array()
    };
    Ok(Some(result))
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_buffer::BooleanBuffer;
    use futures::executor::block_on;
    use futures::stream;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::pipeline::operators::MaskFuture;
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayContext, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_expr::{gt, lit, root};

    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::{SegmentSource, SequenceWriter, TestSegments};
    use crate::sequence::SequenceId;
    use crate::{LayoutStrategy as _, SequentialStreamAdapter, SequentialStreamExt};

    #[test]
    fn flat_identity() {
        block_on(async {
            let ctx = ArrayContext::empty();
            let segments = TestSegments::default();
            let sequence_writer = SequenceWriter::new(Box::new(segments.clone()));
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid).to_array();
            let array_clone = array.clone();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    &ctx,
                    sequence_writer.clone(),
                    SequentialStreamAdapter::new(
                        array.dtype().clone(),
                        stream::once(async { Ok((SequenceId::root().downgrade(), array_clone)) }),
                    )
                    .sendable(),
                )
                .await
                .unwrap();
            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

            let result = layout
                .new_reader("".into(), segments)
                .unwrap()
                .projection_evaluation(&(0..layout.row_count()), &root())
                .unwrap()
                .invoke(MaskFuture::new_true(layout.row_count().try_into().unwrap()))
                .await
                .unwrap()
                .to_primitive();

            assert_eq!(
                array.to_primitive().as_slice::<i32>(),
                result.as_slice::<i32>()
            );
        })
    }

    #[test]
    fn flat_expr() {
        block_on(async {
            let ctx = ArrayContext::empty();
            let segments = TestSegments::default();
            let sequence_writer = SequenceWriter::new(Box::new(segments.clone()));
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid).to_array();
            let array_clone = array.clone();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    &ctx,
                    sequence_writer.clone(),
                    SequentialStreamAdapter::new(
                        array.dtype().clone(),
                        stream::once(async { Ok((SequenceId::root().downgrade(), array_clone)) }),
                    )
                    .sendable(),
                )
                .await
                .unwrap();
            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

            let expr = gt(root(), lit(3i32));
            let result = layout
                .new_reader("".into(), segments)
                .unwrap()
                .projection_evaluation(&(0..layout.row_count()), &expr)
                .unwrap()
                .invoke(MaskFuture::new_true(layout.row_count().try_into().unwrap()))
                .await
                .unwrap()
                .to_bool();

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
            let segments = TestSegments::default();
            let sequence_writer = SequenceWriter::new(Box::new(segments.clone()));
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid).to_array();
            let array_clone = array.clone();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    &ctx,
                    sequence_writer.clone(),
                    SequentialStreamAdapter::new(
                        array.dtype().clone(),
                        stream::once(async { Ok((SequenceId::root().downgrade(), array_clone)) }),
                    )
                    .sendable(),
                )
                .await
                .unwrap();
            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

            let result = layout
                .new_reader("".into(), segments)
                .unwrap()
                .projection_evaluation(&(2..4), &root())
                .unwrap()
                .invoke(MaskFuture::new_true(2))
                .await
                .unwrap()
                .to_primitive();

            assert_eq!(result.as_slice::<i32>(), &[3, 4],);
        })
    }
}
