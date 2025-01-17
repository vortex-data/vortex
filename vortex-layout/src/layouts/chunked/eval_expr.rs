use async_trait::async_trait;
use futures::future::{ready, try_join_all};
use futures::FutureExt;
use vortex_array::array::{ChunkedArray, ConstantArray};
use vortex_array::{ArrayDType, ArrayData, Canonical, IntoArrayData};
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_scalar::Scalar;
use vortex_scan::RowMask;

use crate::layouts::chunked::reader::ChunkedReader;
use crate::reader::LayoutReaderExt;
use crate::ExprEvaluator;

#[async_trait]
impl ExprEvaluator for ChunkedReader {
    async fn evaluate_expr(
        self: &Self,
        row_mask: RowMask,
        expr: ExprRef,
    ) -> VortexResult<ArrayData> {
        // Compute the result dtype of the expression.
        let dtype = expr
            .evaluate(&Canonical::empty(self.dtype())?.into_array())?
            .dtype()
            .clone();

        // First we need to compute the pruning mask
        let pruning_mask = self.pruning_mask(&expr).await?;

        // Now we set up futures to evaluate each chunk at the same time
        let mut chunks = Vec::with_capacity(self.nchunks());

        let mut row_offset = 0;
        for chunk_idx in 0..self.nchunks() {
            let chunk_reader = self.child(chunk_idx)?;

            // Figure out the row range of the chunk
            let chunk_len = chunk_reader.layout().row_count();
            let chunk_range = row_offset..row_offset + chunk_len;
            row_offset += chunk_len;

            // Try to skip the chunk based on the row-mask
            if row_mask.is_disjoint(chunk_range.clone()) {
                continue;
            }

            // If the pruning mask tells us the chunk is pruned (i.e. the expr is ALL false),
            // then we can just return a constant array.
            if let Some(pruning_mask) = &pruning_mask {
                if pruning_mask.value(chunk_idx) {
                    let false_array = ConstantArray::new(
                        Scalar::bool(false, dtype.nullability()),
                        row_mask.true_count(),
                    );
                    chunks.push(ready(Ok(false_array.into_array())).boxed());
                    continue;
                }
            }

            // Otherwise, we need to read it. So we set up a mask for the chunk range.
            let chunk_mask = row_mask
                .slice(chunk_range.start, chunk_range.end)?
                .shift(chunk_range.start)?;

            let expr = expr.clone();
            chunks.push(chunk_reader.evaluate_expr(chunk_mask, expr).boxed());
        }

        // Wait for all chunks to be evaluated
        let chunks = try_join_all(chunks).await?;

        Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use futures::executor::block_on;
    use vortex_array::array::{BoolArray, ChunkedArray, ConstantArray};
    use vortex_array::{ArrayLen, IntoArrayData, IntoArrayVariant};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, PType};
    use vortex_error::VortexExpect;
    use vortex_expr::{gt, lit, Identity};
    use vortex_scan::RowMask;

    use crate::layouts::chunked::writer::ChunkedLayoutWriter;
    use crate::segments::test::TestSegments;
    use crate::strategies::LayoutWriterExt;
    use crate::LayoutData;

    /// Create a chunked layout with three chunks of primitive arrays.
    fn chunked_layout() -> (Arc<TestSegments>, LayoutData) {
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
        (Arc::new(segments), layout)
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
