use std::future::Future;
use std::ops::{BitAnd, Range};
use std::sync::Arc;

use vortex_array::compute::FilterMask;
use vortex_array::{ArrayData, IntoArrayVariant};
use vortex_error::VortexResult;
use vortex_expr::ExprRef;

use crate::{RowMask, Scan};

pub struct RangeScan {
    scan: Arc<Scan>,
    row_range: Range<u64>,
    mask: FilterMask,
    state: State,
}

enum State {
    // First we run the filter expression over the full row range.
    FilterEval((FilterMask, ExprRef)),
    // Then we project the selected rows.
    Project((FilterMask, ExprRef)),
    // And then we're done.
    Ready(ArrayData),
}

pub enum NextOp {
    /// The finished result of the scan.
    Ready(ArrayData),
    /// The next expression to evaluate.
    /// The caller **must** first apply the mask before evaluating the expression.
    Eval((Range<u64>, FilterMask, ExprRef)),
}

/// We implement the range scan via polling for the next operation to perform, and then posting
/// the result back to the range scan to make progress.
///
/// This allows us to minimize the amount of logic we duplicate in order to support both
/// synchronous and asynchronous evaluation.
///
/// ## A note on the API of evaluator functions
//
// We have chosen a general "run this expression" API instead of separate
// `filter(row_mask, expr) -> row_mask` + `project(row_mask, field_mask)` APIs. The reason for
// this is so we can eventually support cell-level push-down.
//
// If we only projected using a field mask, then it means we need to download all the data
// for the rows of field present in the row mask. When I say cell-level push-down, I mean
// we can slice the cell directly out of storage using an API like
// `SegmentReader::read(segment_id, byte_range: Range<usize>)`.
//
// Admittedly, this is a highly advanced use-case, but can prove invaluable for large cell values
// such as images and video.
//
// If instead we make the projection API `project(row_mask, expr)`, then the project API is
// identical to the filter API and there's no point having both. Hence, a single
// `evaluate(row_mask, expr)` API.
impl RangeScan {
    pub(crate) fn new(scan: Arc<Scan>, row_offset: u64, mask: FilterMask) -> Self {
        let state = scan
            .filter()
            .map(|filter| {
                // If we have a filter expression, then for now we evaluate it against all rows
                // of the range.
                // TODO(ngates): we should decide based on mask.true_count() whether to include the
                //  current mask or not. But note that the resulting expression would need to be
                //  aligned and intersected with the given mask.
                State::FilterEval((FilterMask::new_true(mask.len()), filter.clone()))
            })
            .unwrap_or_else(|| {
                // If there is no filter expression, then we immediately perform a mask + project.
                State::Project((mask.clone(), scan.projection().clone()))
            });

        Self {
            scan,
            row_range: row_offset..row_offset + mask.len() as u64,
            mask,
            state,
        }
    }

    /// Check for the next operation to perform.
    /// Returns `Poll::Ready` when the scan is complete.
    fn next(&self) -> NextOp {
        match &self.state {
            State::FilterEval((mask, expr)) => {
                NextOp::Eval((self.row_range.clone(), mask.clone(), expr.clone()))
            }
            State::Project((mask, expr)) => {
                NextOp::Eval((self.row_range.clone(), mask.clone(), expr.clone()))
            }
            State::Ready(array) => NextOp::Ready(array.clone()),
        }
    }

    /// Post the result of the last expression evaluation back to the range scan.
    fn post(&mut self, result: ArrayData) -> VortexResult<()> {
        match &self.state {
            State::FilterEval(_) => {
                // Intersect the result of the filter expression with our initial row mask.
                let mask = result.into_bool()?.boolean_buffer();
                let mask = self.mask.to_boolean_buffer()?.bitand(&mask);
                // Then move onto the projection
                self.state =
                    State::Project((FilterMask::from(mask), self.scan.projection().clone()))
            }
            State::Project(_) => {
                // We're done.
                self.state = State::Ready(result);
            }
            State::Ready(_) => {}
        }
        Ok(())
    }

    /// Evaluate the [`RangeScan`] operation using a synchronous expression evaluator.
    pub fn evaluate<E>(mut self, evaluator: E) -> VortexResult<ArrayData>
    where
        E: Fn(RowMask, ExprRef) -> VortexResult<ArrayData>,
    {
        loop {
            match self.next() {
                NextOp::Ready(array) => return Ok(array),
                NextOp::Eval((row_range, mask, expr)) => {
                    self.post(evaluator(RowMask::new(mask, row_range.start), expr)?)?;
                }
            }
        }
    }

    /// Evaluate the [`RangeScan`] operation using an async expression evaluator.
    pub async fn evaluate_async<E, F>(mut self, evaluator: E) -> VortexResult<ArrayData>
    where
        E: Fn(RowMask, ExprRef) -> F,
        F: Future<Output = VortexResult<ArrayData>>,
    {
        loop {
            match self.next() {
                NextOp::Ready(array) => return Ok(array),
                NextOp::Eval((row_range, mask, expr)) => {
                    self.post(evaluator(RowMask::new(mask, row_range.start), expr).await?)?;
                }
            }
        }
    }
}
