use std::future::Future;
use std::ops::{BitAnd, Range};
use std::sync::Arc;

use vortex_array::Array;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::{RowMask, Scanner};

/// A scan operation defined for a single row-range of the columnar data.
pub struct RangeScanner {
    row_range: Range<u64>,
    mask: Mask,
    state: State,
}

enum State {
    // We run the filters.
    FilterEval((Mask, Vec<ExprRef>)),
    // And then we return the projection mask.
    Ready(Mask),
}

/// The next operation that should be performed. Either an expression to run, or the result
/// of the [`RangeScanner`].
pub enum NextOp {
    /// The finished result of the scan is the projection mask.
    Ready(Mask),
    /// The next expression to evaluate.
    /// The caller **must** first apply the mask before evaluating the expression.
    Eval((Range<u64>, Mask, ExprRef)),
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
impl RangeScanner {
    pub(crate) fn new(scan: Arc<Scanner>, row_offset: u64, mask: Mask) -> Self {
        let state = if !scan.rev_filter.is_empty() {
            // If we have a filter expression, then for now we evaluate it against all rows
            // of the range.
            // TODO(ngates): we should decide based on mask.true_count() whether to include the
            //  current mask or not. But note that the resulting expression would need to be
            //  aligned and intersected with the given mask.
            State::FilterEval((Mask::new_true(mask.len()), scan.rev_filter.to_vec()))
        } else {
            State::Ready(mask.clone())
        };

        Self {
            row_range: row_offset..row_offset + mask.len() as u64,
            mask,
            state,
        }
    }

    /// Check for the next operation to perform.
    /// Returns `Poll::Ready` when the scan is complete.
    ///
    // FIXME(ngates): currently we have to clone the Mask to return it. Doing this
    //  forces the eager evaluation of the iterators.
    fn next(&self) -> NextOp {
        match &self.state {
            State::FilterEval((mask, conjuncts)) => NextOp::Eval((
                self.row_range.clone(),
                mask.clone(),
                conjuncts
                    .last()
                    .vortex_expect("conjuncts is always non-empty")
                    .clone(),
            )),
            State::Ready(mask) => NextOp::Ready(mask.clone()),
        }
    }

    /// Post the result of the last expression evaluation back to the range scan.
    fn transition(mut self, result: Array) -> VortexResult<Self> {
        const APPLY_FILTER_SELECTIVITY_THRESHOLD: f64 = 0.2;
        match self.state {
            State::FilterEval((eval_mask, mut conjuncts_rev)) => {
                // conjuncts are non-empty here
                conjuncts_rev.pop();

                // Intersect the result of the filter expression with our initial row mask.
                let mask = Mask::try_from(result)?;

                // We passed a full mask to the eval function so we must bit intersect instead
                // of set-bit intersection if we massed a non-full mask to the evaluator.
                let mask = if self.mask.len() == eval_mask.true_count() {
                    self.mask.bitand(&mask)
                } else {
                    self.mask.intersect_by_rank(&mask)
                };

                if mask.true_count() > 0 && !conjuncts_rev.is_empty() {
                    self.mask = mask;
                    let mask = if self.mask.density() < APPLY_FILTER_SELECTIVITY_THRESHOLD {
                        self.mask.clone()
                    } else {
                        Mask::new_true(self.mask.len())
                    };
                    // conjuncts_rev is again non-empty, so we can put it into FilterEval
                    self.state = State::FilterEval((mask, conjuncts_rev))
                } else {
                    // Otherwise, we're done
                    self.state = State::Ready(mask);
                }
            }
            State::Ready(_) => {
                vortex_bail!("unexpected state transition")
            }
        }
        Ok(self)
    }

    /// Evaluate the [`RangeScanner`] operation using a synchronous expression evaluator.
    pub fn evaluate<E>(mut self, evaluator: E) -> VortexResult<Mask>
    where
        E: Fn(RowMask, ExprRef) -> VortexResult<Array>,
    {
        loop {
            match self.next() {
                NextOp::Ready(mask) => return Ok(mask),
                NextOp::Eval((row_range, mask, expr)) => {
                    self =
                        self.transition(evaluator(RowMask::new(mask, row_range.start), expr)?)?;
                }
            }
        }
    }

    /// Evaluate the [`RangeScanner`] operation using an async expression evaluator.
    pub async fn evaluate_async<E, F>(mut self, evaluator: E) -> VortexResult<Mask>
    where
        E: Fn(RowMask, ExprRef) -> F,
        F: Future<Output = VortexResult<Array>>,
    {
        loop {
            match self.next() {
                NextOp::Ready(mask) => return Ok(mask),
                NextOp::Eval((row_range, mask, expr)) => {
                    log::debug!("RangeScan {:?} evaluating expr {}", row_range, expr);
                    self = self
                        .transition(evaluator(RowMask::new(mask, row_range.start), expr).await?)?;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::array::{BoolArray, PrimitiveArray, StructArray};
    use vortex_array::compute::filter;
    use vortex_array::IntoArray;
    use vortex_expr::{and, get_item, gt, ident, lit};
    use vortex_mask::Mask;

    use crate::{RangeScanner, Scanner};

    #[test]
    fn range_scan_few_conj_filter_low_selectivity() {
        let expr_a = gt(get_item("a", ident()), lit(1));
        let expr_b = gt(get_item("b", ident()), lit(2));
        let expr_c = gt(get_item("c", ident()), lit(3));
        let scan = Arc::new(
            Scanner::new(Some(and(
                expr_a.clone(),
                and(expr_b.clone(), expr_c.clone()),
            )))
            .unwrap(),
        );
        let len = 1000;
        let range = RangeScanner::new(scan, 0, Mask::new_true(len));

        let mask = range
            .evaluate(|mask, expr| {
                let arr = if expr == expr_a.clone() {
                    BoolArray::from_iter((0..mask.len()).map(|i| !(i > 10 && i < 30))).into_array()
                } else if expr == expr_b.clone() {
                    BoolArray::from_iter((0..mask.len()).map(|i| !(i > 100 && i < 130)))
                        .into_array()
                } else if expr == expr_c.clone() {
                    BoolArray::from_iter((0..mask.len()).map(|i| !(i > 510 && i < 530)))
                        .into_array()
                } else if expr == ident() {
                    let arr = PrimitiveArray::from_iter(0..mask.len() as u64).into_array();
                    StructArray::from_fields(
                        [("a", arr.clone()), ("b", arr.clone()), ("c", arr)].as_slice(),
                    )
                    .unwrap()
                    .into_array()
                } else {
                    unreachable!("unexpected expression {:?}", expr)
                };

                filter(&arr, mask.filter_mask())
            })
            .unwrap();

        assert_eq!(
            mask.to_boolean_buffer()
                .iter()
                .collect::<Vec<_>>()
                .as_slice(),
            (0..len as u64)
                .map(|i| {
                    (i <= 10 || i >= 30) && (i <= 100 || i >= 130) && (i <= 510 || i >= 530)
                })
                .collect::<Vec<_>>()
                .as_slice()
        );
    }

    #[test]
    fn range_scan_few_conj_filter_high_overlapping_selectivity() {
        let expr_a = gt(get_item("a", ident()), lit(1));
        let expr_b = gt(get_item("b", ident()), lit(2));
        let expr_c = gt(get_item("c", ident()), lit(3));
        let scan = Arc::new(
            Scanner::new(Some(and(
                expr_a.clone(),
                and(expr_b.clone(), expr_c.clone()),
            )))
            .unwrap(),
        );
        let len = 1000;
        let range = RangeScanner::new(scan, 0, Mask::new_true(len));

        let mask = range
            .evaluate(|mask, expr| {
                let arr = if expr == expr_a.clone() {
                    BoolArray::from_iter((0..mask.len()).map(|i| !(i > 10 && i < 900))).into_array()
                } else if expr == expr_b.clone() {
                    BoolArray::from_iter((0..mask.len()).map(|i| !(i > 100 && i < 950)))
                        .into_array()
                } else if expr == expr_c.clone() {
                    BoolArray::from_iter((0..mask.len()).map(|i| !(i > 210 && i < 990)))
                        .into_array()
                } else if expr == ident() {
                    let arr = PrimitiveArray::from_iter(0..mask.len() as u64).into_array();
                    StructArray::from_fields(
                        [("a", arr.clone()), ("b", arr.clone()), ("c", arr)].as_slice(),
                    )
                    .unwrap()
                    .into_array()
                } else {
                    unreachable!("unexpected expression {:?}", expr)
                };

                filter(&arr, mask.filter_mask())
            })
            .unwrap();

        assert_eq!(
            mask.to_boolean_buffer()
                .iter()
                .collect::<Vec<_>>()
                .as_slice(),
            (0..len as u64)
                .map(|i| !(i > 10 && i < 990))
                .collect::<Vec<_>>()
                .as_slice()
        );
    }
}
