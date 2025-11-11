// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::{BitAnd, Range};
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::compute::filter;
use vortex_array::expr::{Expression, is_root};
use vortex_array::serde::ArrayParts;
use vortex_array::{Array, ArrayRef, MaskFuture};
use vortex_dtype::{DType, FieldMask};
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap as _};
use vortex_mask::Mask;

use crate::LayoutReader;
use crate::layouts::SharedArrayFuture;
use crate::layouts::flat::FlatLayout;
use crate::segments::SegmentSource;

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

        let ctx = self.layout.array_ctx().clone();
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

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        splits.insert(row_range.start + self.layout.row_count);
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let row_range = usize::try_from(row_range.start)
            .vortex_expect("Row range begin must fit within FlatLayout size")
            ..usize::try_from(row_range.end)
                .vortex_expect("Row range end must fit within FlatLayout size");
        let name = self.name.clone();
        let array = self.array_future();
        let expr = expr.clone();

        Ok(MaskFuture::new(mask.len(), async move {
            // TODO(ngates): if the mask density is low enough, or if the mask is dense within a range
            //  (as often happens with zone map pruning), then we could slice/filter the array prior
            //  to evaluating the expression.
            let mut array = array.clone().await?;
            let mask = mask.await?;

            // Slice the array based on the row mask.
            if row_range.start > 0 || row_range.end < array.len() {
                array = array.slice(row_range.clone());
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
                let array_mask = expr.evaluate(&array)?.try_to_mask_fill_null_false()?;
                mask.intersect_by_rank(&array_mask)
            } else {
                // Evaluate all rows, avoiding the more expensive rank intersection.
                array = expr.evaluate(&array)?;
                let array_mask = array.try_to_mask_fill_null_false()?;
                mask.bitand(&array_mask)
            };

            log::debug!(
                "Flat mask evaluation {} - {} (mask = {}) => {}",
                name,
                expr,
                mask.density(),
                array_mask.density(),
            );

            Ok(array_mask)
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let row_range = usize::try_from(row_range.start)
            .vortex_expect("Row range begin must fit within FlatLayout size")
            ..usize::try_from(row_range.end)
                .vortex_expect("Row range end must fit within FlatLayout size");
        let name = self.name.clone();
        let array = self.array_future();
        let expr = expr.clone();

        Ok(async move {
            log::debug!("Flat array evaluation {} - {}", name, expr);

            let mut array = array.clone().await?;
            let mask = mask.await?;

            // Slice the array based on the row mask.
            if row_range.start > 0 || row_range.end < array.len() {
                array = array.slice(row_range.clone());
            }

            // Filter the array based on the row mask.
            if !mask.all_true() {
                array = filter(&array, &mask)?;
            }

            // Evaluate the projection expression.
            if !is_root(&expr) {
                array = expr.evaluate(&array)?;
            }

            Ok(array)
        }
        .boxed())
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::expr::{gt, lit, root};
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayContext, IntoArray, MaskFuture, ToCanonical, assert_arrays_eq};
    use vortex_buffer::{BitBuffer, buffer};
    use vortex_io::runtime::single::block_on;

    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::{SequenceId, SequentialArrayStreamExt};

    #[test]
    fn flat_identity() {
        block_on(|handle| async {
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid).to_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    segments.clone(),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    handle,
                )
                .await
                .unwrap();

            let result = layout
                .new_reader("".into(), segments)
                .unwrap()
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &root(),
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
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
        block_on(|handle| async {
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid).to_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    segments.clone(),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    handle,
                )
                .await
                .unwrap();

            let expr = gt(root(), lit(3i32));
            let result = layout
                .new_reader("".into(), segments)
                .unwrap()
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &expr,
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap()
                .to_bool();

            assert_eq!(
                &BitBuffer::from_iter([false, false, false, true, true]),
                result.bit_buffer()
            );
        })
    }

    #[test]
    fn flat_unaligned_row_mask() {
        block_on(|handle| async {
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid).to_array();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    segments.clone(),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    handle,
                )
                .await
                .unwrap();

            let result = layout
                .new_reader("".into(), segments)
                .unwrap()
                .projection_evaluation(&(2..4), &root(), MaskFuture::new_true(2))
                .unwrap()
                .await
                .unwrap();

            let expected = PrimitiveArray::new(buffer![3i32, 4], Validity::AllValid).into_array();
            assert_arrays_eq!(result.as_ref(), expected.as_ref());
        })
    }
}
