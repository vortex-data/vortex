use std::ops::{BitAnd, Range};
use std::sync::Arc;

use vortex_array::compute::FilterMask;
use vortex_array::{ArrayData, IntoArrayVariant};
use vortex_error::VortexResult;
use vortex_expr::ExprRef;

use crate::Scan;

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

    /// The caller polls for the next expression they need to evaluate.
    /// Once they have evaluated the expression, they must post the result back to the range scan
    /// to make progress.
    ///
    /// Returns `Poll::Ready` when the scan is complete.
    pub fn next(&self) -> NextOp {
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
    pub fn post(&mut self, result: ArrayData) -> VortexResult<()> {
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
}
