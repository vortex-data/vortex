use std::future::ready;
use std::ops::{BitAnd, Sub};

use arrow_buffer::BooleanBufferBuilder;
use async_trait::async_trait;
use futures::future::try_join_all;
use futures::FutureExt;
use vortex_array::array::{ChunkedArray, ConstantArray};
use vortex_array::{Array, IntoArray};
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::ExprRef;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::layouts::chunked::reader::ChunkedReader;
use crate::reader::LayoutReaderExt;
use crate::{ExprEvaluator, RowMask};

#[async_trait]
impl ExprEvaluator for ChunkedReader {
    async fn evaluate_expr(self: &Self, row_mask: RowMask, expr: ExprRef) -> VortexResult<Array> {
        // Compute the result dtype of the expression.
        let dtype = expr.return_dtype(self.dtype())?;

        // If the expression is prune-able, it means we're evaluating a boolean. Even for
        // projections, we can short-circuit the evaluation of the expression and use the pruning
        // mask to return a ConstantArray.
        let pruning_mask = self.pruning_mask(&expr).await?;

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

            // If the pruning mask tells us the chunk is pruned (i.e. the expr is ALL false),
            // then we can just return a constant array.
            if let Some(pruning_mask) = &pruning_mask {
                if pruning_mask.value(chunk_idx) {
                    let false_array = ConstantArray::new(
                        Scalar::bool(false, dtype.nullability()),
                        chunk_mask.true_count(),
                    );
                    chunks.push(ready(Ok(false_array.into_array())).boxed());
                    continue;
                }
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
        Ok(ChunkedArray::try_new_unchecked(chunks, dtype).into_array())
    }

    async fn prune_mask(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<RowMask> {
        // First we need to compute the pruning mask
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

        // Figure out which chunks intersect the RowMask
        let chunk_range = self.chunk_range(row_mask.begin()..row_mask.end());

        // Extract the range mask from the RowMask
        let mut mask = row_mask.filter_mask().clone();

        for chunk_idx in chunk_range {
            if pruning_mask.value(chunk_idx) {
                // Figure out the range in the mask that corresponds to the chunk
                let start = usize::try_from(
                    self.chunk_offset(chunk_idx)
                        .saturating_sub(row_mask.begin()),
                )?;
                let end = usize::try_from(
                    self.chunk_offset(chunk_idx + 1)
                        .sub(row_mask.begin())
                        .min(mask.len() as u64),
                )?;

                // Build a mask that's *false* for the chunk range
                let mut chunk_mask = BooleanBufferBuilder::new(mask.len());
                chunk_mask.append_n(start, true);
                chunk_mask.append_n(end - start, false);
                chunk_mask.append_n(mask.len() - end, true);
                let chunk_mask = Mask::from_buffer(chunk_mask.finish());

                // Update the pruning mask.
                mask = mask.bitand(&chunk_mask);
            } else {
                // TODO(ngates): we could push-down the pruning request to the child. This
                //  would be used for chunk-of-chunk layouts.
            }
        }

        Ok(RowMask::new(mask, row_mask.begin()))
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
    use vortex_array::array::{BoolArray, ChunkedArray, ConstantArray};
    use vortex_array::{IntoArray, IntoArrayVariant};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, PType};
    use vortex_error::VortexExpect;
    use vortex_expr::{gt, lit, Identity};

    use crate::layouts::chunked::writer::ChunkedLayoutWriter;
    use crate::scan::ScanExecutor;
    use crate::segments::test::TestSegments;
    use crate::writer::LayoutWriterExt;
    use crate::{Layout, RowMask};

    /// Create a chunked layout with three chunks of primitive arrays.
    fn chunked_layout() -> (Arc<ScanExecutor>, Layout) {
        let mut segments = TestSegments::default();
        let layout = ChunkedLayoutWriter::new(
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
        (ScanExecutor::inline(Arc::new(segments)), layout)
    }

    #[test]
    fn test_chunked_evaluator() {
        block_on(async {
            let (segments, layout) = chunked_layout();

            let result = layout
                .reader(segments, Default::default())
                .unwrap()
                .evaluate_expr(
                    RowMask::new_valid_between(0, layout.row_count()),
                    Identity::new_expr(),
                )
                .await
                .unwrap()
                .into_primitive()
                .unwrap();

            assert_eq!(result.len(), 9);
            assert_eq!(result.as_slice::<i32>(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        })
    }

    #[test]
    fn test_chunked_pruning_mask() {
        block_on(async {
            let (segments, layout) = chunked_layout();
            let row_count = layout.row_count();
            let reader = layout.reader(segments, Default::default()).unwrap();

            // Choose a prune-able expression
            let expr = gt(Identity::new_expr(), lit(7));

            let result = reader
                .evaluate_expr(RowMask::new_valid_between(0, row_count), expr.clone())
                .await
                .unwrap();
            let result = ChunkedArray::try_from(result).unwrap();

            // Now we ensure that the pruned chunks are ConstantArrays, instead of having been
            // evaluated.
            assert_eq!(result.nchunks(), 3);
            ConstantArray::try_from(result.chunk(0).unwrap())
                .vortex_expect("Expected first chunk to be pruned");
            ConstantArray::try_from(result.chunk(1).unwrap())
                .vortex_expect("Expected second chunk to be pruned");
            BoolArray::try_from(result.chunk(2).unwrap())
                .vortex_expect("Expected third chunk to be evaluated");
        })
    }
}
