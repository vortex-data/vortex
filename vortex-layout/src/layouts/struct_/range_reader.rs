use std::ops::Range;

use async_trait::async_trait;
use vortex_array::Array;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::{LayoutRangeReader, LayoutReader};

pub struct StructRangeReader {
    row_range: Range<u64>,
}

#[async_trait]
impl LayoutRangeReader for StructRangeReader {
    async fn evaluate_expr(&self, mask: Mask, expr: ExprRef) -> VortexResult<Array> {
        todo!()
    }
}
