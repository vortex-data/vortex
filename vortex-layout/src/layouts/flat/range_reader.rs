use std::ops::{Deref, Range};

use async_trait::async_trait;
use vortex_array::compute::{filter, slice};
use vortex_array::Array;
use vortex_error::{vortex_bail, VortexResult};
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::layouts::flat::reader::LazyArray;
use crate::LayoutRangeReader;

pub struct FlatRangeReader {
    pub(super) row_range: Range<u64>,
    pub(super) self_range: Range<usize>,
    pub(super) array: LazyArray,
}

#[async_trait]
impl LayoutRangeReader for FlatRangeReader {
    fn row_range(&self) -> &Range<u64> {
        &self.row_range
    }

    async fn evaluate_expr(&self, mask: Mask, expr: ExprRef) -> VortexResult<Array> {
        let array = match self.array.deref().await {
            Ok(array) => array.clone(),
            // TODO(ngates): Improve error handling, we lose backtrace since VortexError isn't clone
            Err(e) => vortex_bail!("Failed to load array: {}", e),
        };

        let array = slice(array, self.self_range.start, self.self_range.end)?;
        let array = filter(&array, &mask)?;
        // Then apply the expression
        expr.evaluate(&array)
    }
}
