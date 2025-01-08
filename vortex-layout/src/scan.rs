use std::sync::Arc;

use vortex_array::ArrayData;
use vortex_error::VortexResult;
use vortex_scan::{NextOp, RangeScan};

use crate::operations::{Operation, Poll};
use crate::segments::SegmentReader;
use crate::{EvalOp, LayoutReader, RowMask};

/// A layout operation that executes a [`Scan`].
pub(crate) struct LayoutRangeScan {
    reader: Arc<dyn LayoutReader>,
    range_scan: RangeScan,
    evaluator: Option<EvalOp>,
}

impl LayoutRangeScan {
    pub(crate) fn new(reader: Arc<dyn LayoutReader>, range_scan: RangeScan) -> Self {
        Self {
            reader,
            range_scan,
            evaluator: None,
        }
    }
}

impl Operation for LayoutRangeScan {
    type Output = ArrayData;

    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll<Self::Output>> {
        loop {
            match self.evaluator.take() {
                None => match self.range_scan.next() {
                    // The scan has finished, return the result.
                    NextOp::Ready(array) => return Ok(Poll::Some(array)),
                    NextOp::Eval((row_range, mask, expr)) => {
                        let evaluator = self.reader.clone().create_evaluator(
                            RowMask::try_new(mask, row_range.start, row_range.end)?,
                            expr,
                        )?;
                        self.evaluator = Some(evaluator);
                    }
                },
                Some(mut evaluator) => {
                    match evaluator.poll(segments)? {
                        Poll::Some(array) => self.range_scan.post(array)?,
                        Poll::NeedMore(segments) => {
                            // Replace the evaluator to continue using it
                            self.evaluator = Some(evaluator);
                            return Ok(Poll::NeedMore(segments));
                        }
                    }
                }
            }
        }
    }
}
