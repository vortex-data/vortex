// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

use vortex_array::MaskFuture;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_scan::planning::morsel::Morsel;
use vortex_scan::planning::morsel::MorselOutput;
use vortex_scan::planning::planner::State;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::scan::planning::segments::PollingSegmentSource;

/// Evaluates the filter and projection for one split by polling the reader's future, one poll
/// per compute, and reporting the segments each poll missed as `NeedsIO`. Only rows selected by
/// the initial mask are considered. A split with no matching rows finishes without a batch.
pub struct SplitMorsel {
    source: Arc<PollingSegmentSource>,
    reader: Arc<dyn LayoutReader>,
    range: Range<u64>,
    /// Rows of `range` to consider before the filter runs.
    mask: Mask,
    filter: Option<BoundExpression>,
    projection: BoundExpression,
    /// The reader's future, built on the first compute and polled until ready.
    pending: Option<ArrayFuture>,
    /// Requests published by the last poll and not yet delivered.
    outstanding: IoBatch,
    /// How many `NeedsIO` rounds the split has taken so far.
    rounds: usize,
    done: bool,
}

impl SplitMorsel {
    /// Creates a morsel for the rows of `range` selected by `mask`, read through `reader`, whose
    /// segment source must be `source`.
    pub fn new(
        source: Arc<PollingSegmentSource>,
        reader: Arc<dyn LayoutReader>,
        range: Range<u64>,
        mask: Mask,
        filter: Option<BoundExpression>,
        projection: BoundExpression,
    ) -> Self {
        Self {
            source,
            reader,
            range,
            mask,
            filter,
            projection,
            pending: None,
            outstanding: Vec::new(),
            rounds: 0,
            done: false,
        }
    }

    /// Number of `NeedsIO` rounds so far: one per poll that missed a segment.
    pub fn rounds(&self) -> usize {
        self.rounds
    }

    fn build(&self) -> VortexResult<ArrayFuture> {
        let mut mask = MaskFuture::ready(self.mask.clone());
        if let Some(filter) = &self.filter {
            mask = self.reader.filter_evaluation(&self.range, filter, mask)?;
        }
        self.reader
            .projection_evaluation(&self.range, &self.projection, mask)
    }
}

impl IoConsumer for SplitMorsel {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        let IoResult::Bytes(bytes) = result else {
            vortex_panic!("SplitMorsel: segment {request:?} answered with a size");
        };
        let before = self.outstanding.len();
        self.outstanding
            .retain(|pending| pending.request != request);
        if self.outstanding.len() == before {
            vortex_panic!("SplitMorsel: delivery of {request:?}, which is not outstanding");
        }
        if let Err(error) = self.source.deliver(request, bytes) {
            vortex_panic!("SplitMorsel: delivery of {request:?} failed: {error}");
        }
    }
}

impl Morsel for SplitMorsel {
    fn state(&self) -> State {
        if self.done {
            State::Done
        } else if self.outstanding.is_empty() {
            State::NeedsCompute
        } else {
            State::NeedsIO(self.outstanding.clone())
        }
    }

    fn compute(&mut self) -> VortexResult<MorselOutput> {
        if self.done {
            vortex_bail!("SplitMorsel: compute called after Done");
        }
        if self.pending.is_none() {
            self.pending = Some(self.build()?);
        }
        let Some(future) = self.pending.as_mut() else {
            vortex_bail!("SplitMorsel: no future to poll");
        };
        let mut cx = Context::from_waker(Waker::noop());
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(array) => {
                self.done = true;
                self.pending = None;
                let array = array?;
                Ok(if array.is_empty() {
                    MorselOutput::Done
                } else {
                    MorselOutput::Batch(array)
                })
            }
            Poll::Pending => {
                let batch = self.source.take_batch()?;
                if batch.is_empty() {
                    vortex_bail!(
                        "SplitMorsel for rows {:?}: the reader is pending on something that is \
                         not a segment",
                        self.range
                    );
                }
                self.rounds += 1;
                self.outstanding = batch.clone();
                Ok(MorselOutput::NeedsIO(batch))
            }
        }
    }
}

#[cfg(test)]
mod tests;
