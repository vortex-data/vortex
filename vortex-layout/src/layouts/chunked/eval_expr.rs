use std::ops::Range;

use async_trait::async_trait;
use futures::future::{BoxFuture, try_join_all};
use futures::stream::FuturesOrdered;
use futures::{FutureExt, TryFutureExt, TryStreamExt};
use itertools::Itertools;
use vortex_array::arrays::ChunkedArray;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::layouts::chunked::reader::ChunkedReader;
use crate::reader::LayoutReader;
use crate::{ArrayEvaluation, ExprEvaluator, MaskEvaluation, MaskFuture, RowMask};

impl ExprEvaluator for ChunkedReader {
    fn evaluate_expr2(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<Option<ArrayRef>>>> {
        // Compute the result dtype of the expression.
        let dtype = expr.return_dtype(self.dtype())?;

        // Figure out which chunks intersect the RowMask
        let chunk_range = self.chunk_range(row_range);

        // Now we have to create a future for each chunk.
        let child_futures: Vec<_> = chunk_range
            .map(|chunk_idx| {
                // Figure out the chunk row range relative to the mask's row range.
                let chunk_row_range =
                    self.chunk_offset(chunk_idx)..self.chunk_offset(chunk_idx + 1);

                // Find the intersection of the mask and the chunk row ranges.
                let intersecting_row_range = row_range.start.max(chunk_row_range.start)
                    ..row_range.end.min(chunk_row_range.end);
                let intersecting_len =
                    usize::try_from(intersecting_row_range.end - intersecting_row_range.start)?;

                // Figure out the offset into the mask.
                let mask_relative_start =
                    usize::try_from(intersecting_row_range.start - row_range.start)?;

                let mask: MaskFuture = mask
                    .clone()
                    .map_ok(move |mask| mask.slice(mask_relative_start, intersecting_len))
                    .boxed()
                    .shared();

                // Figure out the row range within the chunk.
                let chunk_relative_start = intersecting_row_range.start - chunk_row_range.start;
                let chunk_relative_end = chunk_relative_start + intersecting_len as u64;

                self.child(chunk_idx)
                    .vortex_expect("out of bounds")
                    .evaluate_expr2(&(chunk_relative_start..chunk_relative_end), expr, mask)
            })
            .try_collect()?;

        Ok(Box::pin(async move {
            let mut chunks: Vec<ArrayRef> = try_join_all(child_futures)
                .await?
                .into_iter()
                .flatten()
                .collect_vec();

            if chunks.len() == 1 {
                // Avoid creating a chunked array for a single chunk
                let chunk = chunks
                    .pop()
                    .vortex_expect("Expected at least one chunk to be evaluated");
                return Ok(Some(chunk));
            }

            let chunked_array = ChunkedArray::new_unchecked(chunks, dtype);
            assert_eq!(
                chunked_array.len(),
                mask.await?.true_count(),
                "Mask length mismatch for chunked layout"
            );

            Ok(Some(chunked_array.into_array()))
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        let mut chunk_evals = vec![];
        let mut mask_ranges = vec![];

        for (chunk_idx, chunk_range, mask_range) in self.ranges(row_range) {
            let chunk_reader = self.child(chunk_idx)?;
            let chunk_eval = chunk_reader.filter_evaluation(&chunk_range, expr)?;
            chunk_evals.push(chunk_eval);
            mask_ranges.push(mask_range);
        }

        Ok(Box::new(ChunkedMaskEvaluation {
            chunk_evals,
            mask_ranges,
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        let dtype = expr.return_dtype(self.dtype())?;
        let mut chunk_evals = vec![];
        let mut mask_ranges = vec![];

        for (chunk_idx, chunk_range, mask_range) in self.ranges(row_range) {
            let chunk_reader = self.child(chunk_idx)?;
            let chunk_eval = chunk_reader.projection_evaluation(&chunk_range, expr)?;
            chunk_evals.push(chunk_eval);
            mask_ranges.push(mask_range);
        }

        Ok(Box::new(ChunkedArrayEvaluation {
            dtype,
            chunk_evals,
            mask_ranges,
        }))
    }
}

struct ChunkedMaskEvaluation {
    chunk_evals: Vec<Box<dyn MaskEvaluation>>,
    mask_ranges: Vec<Range<usize>>,
}

#[async_trait]
impl MaskEvaluation for ChunkedMaskEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        // Split the mask over each chunk.
        let masks: Vec<_> = FuturesOrdered::from_iter(
            self.mask_ranges
                .iter()
                .map(|range| mask.slice(range.start, range.end - range.start))
                .zip_eq(&self.chunk_evals)
                .map(|(mask, chunk_eval)| chunk_eval.invoke(mask)),
        )
        .try_collect()
        .await?;

        // Combine the masks.
        Ok(Mask::from_iter(masks))
    }
}

struct ChunkedArrayEvaluation {
    dtype: DType,
    chunk_evals: Vec<Box<dyn ArrayEvaluation>>,
    mask_ranges: Vec<Range<usize>>,
}

#[async_trait]
impl ArrayEvaluation for ChunkedArrayEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef> {
        // Split the mask over each chunk.
        let chunks: Vec<_> = FuturesOrdered::from_iter(
            self.mask_ranges
                .iter()
                .map(|range| mask.slice(range.start, range.end - range.start))
                .zip_eq(&self.chunk_evals)
                .map(|(mask, chunk_eval)| chunk_eval.invoke(mask)),
        )
        .try_collect()
        .await?;

        // Combine the arrays.
        Ok(ChunkedArray::try_new(chunks, self.dtype.clone())?.to_array())
    }
}

impl ChunkedReader {
    /// Adjust the row mask for the specific chunk.
    fn chunk_mask(&self, chunk_idx: usize, row_mask: &RowMask) -> VortexResult<RowMask> {
        let chunk_row_range = self.chunk_offset(chunk_idx)..self.chunk_offset(chunk_idx + 1);
        row_mask
            .slice(chunk_row_range.start, chunk_row_range.end)?
            .shift(chunk_row_range.start)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use futures::executor::block_on;
    use rstest::{fixture, rstest};
    use vortex_array::{Array, ArrayContext, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, PType};
    use vortex_expr::Identity;

    use crate::layouts::chunked::writer::ChunkedLayoutWriter;
    use crate::segments::AsyncSegmentReader;
    use crate::segments::test::TestSegments;
    use crate::writer::LayoutWriterExt;
    use crate::{Layout, RowMask};

    #[fixture]
    /// Create a chunked layout with three chunks of primitive arrays.
    fn chunked_layout() -> (ArrayContext, Arc<dyn AsyncSegmentReader>, Layout) {
        let ctx = ArrayContext::empty();
        let mut segments = TestSegments::default();
        let layout = ChunkedLayoutWriter::new(
            ctx.clone(),
            &DType::Primitive(PType::I32, NonNullable),
            Default::default(),
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
    fn test_chunked_evaluator(
        #[from(chunked_layout)] (ctx, segments, layout): (
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
}
