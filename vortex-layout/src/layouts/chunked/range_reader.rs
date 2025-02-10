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
