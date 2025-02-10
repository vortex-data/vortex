use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::Array;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_mask::Mask;
use vortex_scan::RowMask;

use crate::layouts::chunked::reader::SharedState;
use crate::{ExprEvaluator, LayoutRangeReader, LayoutReader};

pub struct ChunkedRangeReader {
    pub(super) chunk_range: Range<usize>,
    pub(super) chunks: Vec<Arc<dyn LayoutRangeReader>>,
    pub(super) shared_state: Arc<SharedState>,
}

#[async_trait]
impl LayoutRangeReader for ChunkedRangeReader {
    async fn evaluate_expr(&self, mask: Mask, expr: ExprRef) -> VortexResult<Array> {
        todo!()
    }
}
