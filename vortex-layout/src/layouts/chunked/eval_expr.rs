use async_trait::async_trait;
use futures::future::try_join_all;
use vortex_array::arrays::ChunkedArray;
use vortex_array::{Array, ArrayRef};
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::ExprRef;

use crate::layouts::chunked::reader::ChunkedReader;
use crate::reader::LayoutReaderExt;
use crate::{ExprEvaluator, RowMask};

#[async_trait]
impl ExprEvaluator for ChunkedReader {
    async fn evaluate_expr(
        self: &Self,
        row_mask: RowMask,
        expr: ExprRef,
    ) -> VortexResult<ArrayRef> {
        // Compute the result dtype of the expression.
        let dtype = expr.return_dtype(self.dtype())?;

        // Figure out which chunks intersect the RowMask
        let chunk_range = self.chunk_range(row_mask.begin()..row_mask.end());

        // Now we set up futures to evaluate each chunk at the same time
        let mut chunks = Vec::new();

        for chunk_idx in chunk_range {
            let chunk_mask = self.chunk_mask(chunk_idx, &row_mask)?;

            if chunk_mask.true_count() == 0 {
                // If the chunk is empty skip `evaluate_expr` on child and omit chunk from array
                continue;
            }

            // Otherwise, we need to read it. So we set up a mask for the chunk range.
            let chunk_reader = self.child(chunk_idx)?;
            chunks.push(chunk_reader.evaluate_expr(chunk_mask, expr.clone()));
        }

        if chunks.len() == 1 {
            // Avoid creating a chunked array for a single chunk
            let chunk = chunks
                .pop()
                .vortex_expect("Expected at least one chunk to be evaluated")
                .await?;
            return Ok(chunk);
        }

        let chunks = try_join_all(chunks).await?;
        Ok(ChunkedArray::new_unchecked(chunks, dtype).into_array())
    }

    async fn refine_mask(&self, row_mask: RowMask, _expr: ExprRef) -> VortexResult<RowMask> {
        // TODO(ngates): we should push-down to each child
        Ok(row_mask)
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
    use crate::segments::SegmentReader;
    use crate::segments::test::TestSegments;
    use crate::writer::LayoutWriterExt;
    use crate::{Layout, RowMask};

    #[fixture]
    /// Create a chunked layout with three chunks of primitive arrays.
    fn chunked_layout() -> (ArrayContext, Arc<dyn SegmentReader>, Layout) {
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
            Arc<dyn SegmentReader>,
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
