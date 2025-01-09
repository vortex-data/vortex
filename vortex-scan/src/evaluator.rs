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

/// An async evaluator that can evaluate expressions against a row mask.
///
/// This trait is explicitly non-send to force users to think about how to isolate this async
/// CPU-heavy operation from the rest of their likely I/O-heavy async code.
#[async_trait(?Send)]
pub trait AsyncEvaluator {
    async fn evaluate(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<ArrayData>;
}
