// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::{BitAnd, Range};
use std::sync::Arc;

use vortex_array::compute::filter;
use vortex_array::serde::ArrayParts;
use vortex_array::stats::Precision;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::{DType, FieldMask};
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap as _};
use vortex_expr::{ExprRef, Scope, is_root};
use vortex_mask::Mask;
use vortex_utils::aliases::hash_set::HashSet;

use crate::layouts::SharedArray;
use crate::layouts::flat::FlatLayout;
use crate::segments::{SegmentId, Segments};
use crate::{
    ArrayEvaluation, LayoutReader, LazyWithSegments, MaskEvaluation, NoOpPruningEvaluation,
    PruningEvaluation,
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
    shared_array: SharedArray,
}

impl FlatReader {
    pub(crate) fn new(layout: FlatLayout, name: Arc<str>) -> Self {
        let ctx = layout.ctx.clone();
        let dtype = layout.dtype().clone();
        let row_count = usize::try_from(layout.row_count()).vortex_unwrap();
        let segment_id = layout.segment_id();

        let shared_array = LazyWithSegments::new(move |segments| {
            let segment = segments.get(segment_id);
            ArrayParts::try_from(segment)?.decode(&ctx, &dtype, row_count)
        })
        .with_required_segments([segment_id]);

        Self {
            layout,
            name,
            shared_array,
        }
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
            array: self.shared_array.clone(),
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
            array: self.shared_array.clone(),
            row_range,
            expr: expr.clone(),
        }))
    }
}

struct FlatEvaluation {
    name: Arc<str>,
    array: SharedArray,
    row_range: Range<usize>,
    expr: ExprRef,
}

impl MaskEvaluation for FlatEvaluation {
    fn invoke(&self, mask: Mask, segments: &dyn Segments) -> VortexResult<Mask> {
        // TODO(ngates): if the mask density is low enough, or if the mask is dense within a range
        //  (as often happens with zone map pruning), then we could slice/filter the array prior
        //  to evaluating the expression.

        // Now we await the array .
        let mut array = self.array.get(segments)?.clone();

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
            let array_mask = Mask::try_from(self.expr.evaluate(&Scope::new(array))?.as_ref())?;
            mask.intersect_by_rank(&array_mask)
        } else {
            // Evaluate all rows, avoiding the more expensive rank intersection.
            array = self.expr.evaluate(&Scope::new(array))?;
            let array_mask = Mask::try_from(array.as_ref())?;
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

    fn required_segments(&self, segments: &mut HashSet<SegmentId>) {
        self.array.required_segments(segments);
    }
}

impl ArrayEvaluation for FlatEvaluation {
    fn invoke(&self, mask: Mask, segments: &dyn Segments) -> VortexResult<ArrayRef> {
        log::debug!(
            "Flat array evaluation {} - {} (mask = {})",
            self.name,
            self.expr,
            mask.density(),
        );

        // Now we await the array .
        let mut array = self.array.get(segments)?.clone();

        // Slice the array based on the row mask.
        if self.row_range.start > 0 || self.row_range.end < array.len() {
            array = array.slice(self.row_range.start, self.row_range.end)?;
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

    fn required_segments(&self, segments: &mut HashSet<SegmentId>) {
        self.array.required_segments(segments);
    }
}

#[cfg(test)]
mod test {
    use arrow_buffer::BooleanBuffer;
    use futures::executor::block_on;
    use futures::stream;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayContext, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_expr::{gt, lit, root};
    use vortex_mask::Mask;

    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::{SequenceWriter, TestSegments};
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

            let result = layout
                .new_reader("".into())
                .unwrap()
                .projection_evaluation(&(0..layout.row_count()), &root())
                .unwrap()
                .invoke(
                    Mask::new_true(layout.row_count().try_into().unwrap()),
                    &segments,
                )
                .unwrap()
                .to_primitive()
                .unwrap();

            assert_eq!(
                array.to_primitive().unwrap().as_slice::<i32>(),
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

            let expr = gt(root(), lit(3i32));
            let result = layout
                .new_reader("".into())
                .unwrap()
                .projection_evaluation(&(0..layout.row_count()), &expr)
                .unwrap()
                .invoke(
                    Mask::new_true(layout.row_count().try_into().unwrap()),
                    &segments,
                )
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

            let result = layout
                .new_reader("".into())
                .unwrap()
                .projection_evaluation(&(2..4), &root())
                .unwrap()
                .invoke(Mask::new_true(2), &segments)
                .unwrap()
                .to_primitive()
                .unwrap();

            assert_eq!(result.as_slice::<i32>(), &[3, 4],);
        })
    }
}
