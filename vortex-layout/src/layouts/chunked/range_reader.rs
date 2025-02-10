use std::future::ready;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::FuturesOrdered;
use futures::{FutureExt, TryStreamExt};
use itertools::Itertools;
use vortex_array::array::{ChunkedArray, ConstantArray};
use vortex_array::{Array, IntoArray};
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::layouts::chunked::reader::SharedState;
use crate::{Layout, LayoutRangeReader};

pub struct ChunkedRangeReader {
    /// The layout of this reader.
    pub(super) layout: Layout,
    /// The row range of this reader within the global file.
    pub(super) row_range: Range<u64>,
    /// The range of chunks that this reader covers.
    pub(super) chunk_range: Range<usize>,
    /// The chunk readers for this range, already pruned for the row range.
    pub(super) chunks: Vec<Arc<dyn LayoutRangeReader>>,
    /// Shared state across all chunk readers, e.g. pruning cache.
    pub(super) shared_state: Arc<SharedState>,
}

#[async_trait]
impl LayoutRangeReader for ChunkedRangeReader {
    fn row_range(&self) -> &Range<u64> {
        &self.row_range
    }

    async fn evaluate_expr(&self, mask: Mask, expr: ExprRef) -> VortexResult<Array> {
        // First we need to compute the pruning mask
        let pruning_mask = self.shared_state.pruning_mask(&expr).await?;

        // Now we set up futures to evaluate each chunk at the same time
        let mut chunks = FuturesOrdered::new();

        // Compute the result dtype of the expression.
        let dtype = expr.return_dtype(self.layout.dtype())?;

        let mut mask_offset = 0;
        for (chunk_idx, chunk_reader) in self.chunk_range.clone().zip_eq(self.chunks.iter()) {
            let chunk_mask = mask.slice(mask_offset, chunk_reader.len());
            mask_offset += chunk_reader.len();

            // If the chunk is empty skip `evaluate_expr` on child and omit chunk from array
            if chunk_mask.true_count() == 0 {
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
                    chunks.push_back(ready(Ok(false_array.into_array())).boxed());
                    continue;
                }
            }

            // Otherwise, we need to read it. So we set up a mask for the chunk range.
            chunks.push_back(chunk_reader.evaluate_expr(chunk_mask, expr.clone()));
        }

        let chunks = chunks.try_collect::<Vec<_>>().await?;

        Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
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
    use vortex_dtype::{DType, FieldMask, PType};
    use vortex_error::VortexExpect;
    use vortex_expr::{gt, lit, Identity};
    use vortex_mask::Mask;

    use crate::layouts::chunked::writer::ChunkedLayoutWriter;
    use crate::segments::test::TestSegments;
    use crate::writer::LayoutWriterExt;
    use crate::{Layout, LayoutReader};

    /// Create a chunked layout with three chunks of primitive arrays.
    fn chunked_layout() -> (Arc<TestSegments>, Layout) {
        let mut segments = TestSegments::default();
        let layout = ChunkedLayoutWriter::new(
            &DType::Primitive(PType::I32, NonNullable),
            0,
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
                .reader(segments, Default::default(), &[FieldMask::All])
                .unwrap()
                .range_reader(0..layout.row_count())
                .evaluate_expr(
                    Mask::new_true(usize::try_from(layout.row_count()).unwrap()),
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
            let reader = layout
                .reader(segments, Default::default(), &[FieldMask::All])
                .unwrap();

            // Choose a prune-able expression
            let expr = gt(Identity::new_expr(), lit(7));

            let result = reader
                .range_reader(0..row_count)
                .evaluate_expr(
                    Mask::new_true(usize::try_from(row_count).unwrap()),
                    expr.clone(),
                )
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
