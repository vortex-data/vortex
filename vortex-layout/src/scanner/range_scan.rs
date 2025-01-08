use std::ops::BitAnd;
use std::sync::Arc;

use vortex_array::compute::FilterMask;
use vortex_array::{ArrayData, IntoArrayVariant};
use vortex_error::VortexResult;
use vortex_expr::ExprRef;

use crate::scanner::Scan;
use crate::RowMask;

pub struct RangeScan {
    scan: Arc<Scan>,
    row_mask: RowMask,
    state: State,
}

enum State {
    // First we run the filter expression over the full row range.
    FilterEval((RowMask, ExprRef)),
    // Then we project the selected rows.
    Project((RowMask, ExprRef)),
    // And then we're done.
    Ready(ArrayData),
}

pub enum NextOp {
    Ready(ArrayData),
    Eval((RowMask, ExprRef)),
}

impl RangeScan {
    pub(crate) fn new(scan: Arc<Scan>, row_mask: RowMask) -> Self {
        let state = scan
            .filter()
            .map(|filter| {
                State::FilterEval((
                    RowMask::new_valid_between(row_mask.begin(), row_mask.end()),
                    filter.clone(),
                ))
            })
            .unwrap_or(State::Project((
                row_mask.clone(),
                scan.projection().clone(),
            )));

        Self {
            scan,
            row_mask,
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
            State::FilterEval((mask, expr)) => NextOp::Eval((mask.clone(), expr.clone())),
            State::Project((mask, expr)) => NextOp::Eval((mask.clone(), expr.clone())),
            State::Ready(array) => NextOp::Ready(array.clone()),
        }
    }

    /// Post the result of the last expression evaluation back to the range scan.
    pub fn post(&mut self, result: ArrayData) -> VortexResult<()> {
        match &self.state {
            State::FilterEval(_) => {
                // Intersect the result of the filter expression with our initial row mask.
                let mask = self
                    .row_mask
                    .clone()
                    .into_filter_mask()?
                    .to_boolean_buffer()?;
                let result_mask = result.into_bool()?.boolean_buffer();
                let mask = mask.bitand(&result_mask);

                // Then move onto the projection
                self.state = State::Project((
                    RowMask::try_new(
                        FilterMask::from(mask),
                        self.row_mask.begin(),
                        self.row_mask.end(),
                    )?,
                    self.scan.projection().clone(),
                ))
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
