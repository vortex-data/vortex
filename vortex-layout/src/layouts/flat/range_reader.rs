use std::future::Future;
use std::ops::{Deref, Range};
use std::sync::Arc;

use async_once_cell::{Lazy, OnceCell};
use async_trait::async_trait;
use futures::future::BoxFuture;
use vortex_array::compute::{filter, slice};
use vortex_array::Array;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::layouts::flat::reader::LazyArray;
use crate::segments::{AsyncSegmentReader, SegmentId};
use crate::{Layout, LayoutRangeReader, LayoutReaderExt};

pub struct FlatRangeReader {
    pub(super) row_range: Range<usize>,
    pub(super) array: LazyArray,
}

#[async_trait]
impl LayoutRangeReader for FlatRangeReader {
    async fn evaluate_expr(&self, mask: Mask, expr: ExprRef) -> VortexResult<Array> {
        let array = match self.array.deref().await {
            Ok(array) => array.clone(),
            // TODO(ngates): Improve error handling, we lose backtrace since VortexError isn't clone
            Err(e) => vortex_bail!("Failed to load array: {}", e),
        };
        let array = slice(array, self.row_range.start, self.row_range.end)?;
        let array = filter(&array, &mask)?;
        // Then apply the expression
        expr.evaluate(&array)
    }
}
