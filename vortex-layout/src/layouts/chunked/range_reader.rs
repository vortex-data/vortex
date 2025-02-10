use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::Array;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_mask::Mask;
use vortex_scan::RowMask;

use crate::{ExprEvaluator, LayoutRangeReader, LayoutReader};

pub struct ChunkedRangeReader {
    pub(super) row_range: Range<u64>,
    pub(super) chunks: Vec<Arc<dyn LayoutReader>>,
    pub(super) start_chunk_offset: usize,
    pub(super) end_chunk_length: usize,
}

#[async_trait]
impl LayoutRangeReader for ChunkedRangeReader {
    async fn evaluate_expr(&self, mask: Mask, expr: ExprRef) -> VortexResult<Array> {
        todo!()
    }
}
