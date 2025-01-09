use std::sync::Arc;

// NOTE(ngates): we have chosen a general "run this expression" API instead of  separate
//  `filter(row_mask, expr) -> row_mask` + `project(row_mask, field_mask)` APIs.
//  The reason for this is so we can eventually support cell-level push-down.
//  If we only projected using a field mask, then it means we need to download all the data
//  for the rows of field present in the row mask. When I say cell-level push-down, I mean
//  we can slice the cell directly out of storage using an API like
//  `SegmentReader::read(segment_id, byte_range: Range<usize>)`. This is a highly advanced
//  use-case, but can prove invaluable for large cell values such as images and video.
//  If instead we make the projection API `project(row_mask, expr)`, then identical to the
//  filter API and there's now no point having two. Hence: `evaluate(row_mask, expr)`.
use async_trait::async_trait;
use vortex_array::ArrayData;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;

use crate::RowMask;

pub trait Evaluator {
    fn evaluate(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<ArrayData>;
}

#[async_trait]
pub trait AsyncEvaluator {
    async fn evaluate(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<ArrayData>;
}

#[async_trait]
impl<E: AsyncEvaluator + Send + Sync> AsyncEvaluator for Arc<E> {
    async fn evaluate(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<ArrayData> {
        (**self).evaluate(row_mask, expr).await
    }
}
