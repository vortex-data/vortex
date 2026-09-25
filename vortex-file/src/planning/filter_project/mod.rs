// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filter and project over natural splits, one morsel per split.
//!
//! Data reads go through the protocol. Each morsel holds the layout reader's future and polls it
//! once per compute against a [`PollingSegmentSource`]; the segments the reader could not get
//! become the morsel's `NeedsIO` batch, and delivered bytes let the next poll get further. The
//! reader itself is unchanged and no runtime is involved.

use std::ops::Range;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

use vortex_array::MaskFuture;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_layout::ArrayFuture;
use vortex_layout::LayoutReader;
use vortex_layout::segments::SegmentSource;
use vortex_mask::Mask;
use vortex_scan::planning::morsel::Morsel;
use vortex_scan::planning::morsel::MorselOutput;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;
use vortex_session::VortexSession;

use crate::VortexFile;
use crate::planning::OpenedFile;
use crate::planning::segments::PollingSegmentSource;

/// Opens the file's layout on the first compute, then emits one [`SplitMorsel`] per natural
/// split in file order and finishes.
pub struct FilterProject {
    opened: Option<OpenedFile>,
    filter: Option<Expression>,
    projection: Expression,
    session: VortexSession,
    prepared: Option<Prepared>,
    next_split: usize,
}

struct Prepared {
    /// Shared by every morsel of the file, so a segment two splits both need is read once.
    source: Arc<PollingSegmentSource>,
    reader: Arc<dyn LayoutReader>,
    filter: Option<BoundExpression>,
    projection: BoundExpression,
    splits: Vec<Range<u64>>,
}

impl FilterProject {
    /// Creates the stage; the layout is opened and expressions bound on the first compute.
    pub fn new(
        opened: OpenedFile,
        filter: Option<Expression>,
        projection: Expression,
        session: VortexSession,
    ) -> Self {
        Self {
            opened: Some(opened),
            filter,
            projection,
            session,
            prepared: None,
            next_split: 0,
        }
    }

    fn prepare(&mut self, opened: OpenedFile) -> VortexResult<()> {
        let dtype = opened.footer.dtype().clone();
        let source = Arc::new(PollingSegmentSource::new(
            opened.footer.segment_specs_with_metadata(),
        ));
        let file = VortexFile::new(
            opened.footer,
            Arc::clone(&source) as Arc<dyn SegmentSource>,
            self.session.clone(),
        );
        self.prepared = Some(Prepared {
            source,
            reader: file.layout_reader()?,
            filter: self
                .filter
                .as_ref()
                .map(|filter| filter.bind(&dtype))
                .transpose()?,
            projection: self.projection.bind(&dtype)?,
            splits: file.splits()?,
        });
        Ok(())
    }
}

impl IoConsumer for FilterProject {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for FilterProject {
    fn state(&self) -> State {
        match &self.prepared {
            None if self.opened.is_some() => State::NeedsCompute,
            None => State::Done,
            Some(prepared) if self.next_split < prepared.splits.len() => State::NeedsCompute,
            Some(_) => State::Done,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if let Some(opened) = self.opened.take() {
            self.prepare(opened)?;
            return Ok(PlannerOutput::Continue);
        }
        let Some(prepared) = &self.prepared else {
            vortex_bail!("FilterProject: compute called after Done");
        };
        let Some(range) = prepared.splits.get(self.next_split).cloned() else {
            return Ok(PlannerOutput::Done);
        };
        self.next_split += 1;
        let scope = WorkScope {
            file_ordinal: 0,
            rows: range.clone(),
        };
        Ok(PlannerOutput::Morsel(
            scope,
            Box::new(SplitMorsel {
                source: Arc::clone(&prepared.source),
                reader: Arc::clone(&prepared.reader),
                range,
                filter: prepared.filter.clone(),
                projection: prepared.projection.clone(),
                pending: None,
                outstanding: Vec::new(),
                rounds: 0,
                done: false,
            }),
        ))
    }
}

/// Evaluates the filter and projection for one split by polling the reader's future, one poll
/// per compute, and reporting the segments each poll missed as `NeedsIO`. A split with no
/// matching rows finishes without a batch.
pub struct SplitMorsel {
    source: Arc<PollingSegmentSource>,
    reader: Arc<dyn LayoutReader>,
    range: Range<u64>,
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
    /// Number of `NeedsIO` rounds so far: one per poll that missed a segment.
    pub fn rounds(&self) -> usize {
        self.rounds
    }

    fn build(&self) -> VortexResult<ArrayFuture> {
        let len = usize::try_from(self.range.end - self.range.start)?;
        let mut mask = MaskFuture::ready(Mask::new_true(len));
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
