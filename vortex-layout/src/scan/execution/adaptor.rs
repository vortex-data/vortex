// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Adapts pipeline execution to the planner I/O and compute protocol.

use std::sync::Arc;

use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_io::request;
use vortex_io::request::IoBatch;
use vortex_io::request::IoIntent;
use vortex_io::request::IoResult;
use vortex_scan::planning::planner::State;

use crate::plan::physical::pipeline::ReadId;
use crate::plan::physical::pipeline::Scan;
use crate::plan::physical::pipeline::Turn;
use crate::scan::execution::SegmentLocation;
use crate::segments::SegmentId;

/// Rows a running plan produced, in row order within its split.
#[derive(Clone)]
pub(super) struct Piece {
    /// The index of the split the rows belong to.
    pub(super) split: usize,
    /// The selected rows the piece holds.
    pub(super) array: ArrayRef,
}

/// What one [`ScanAdaptor::compute`] produced.
pub(super) enum AdaptorStep {
    /// Control returns to the owner with CPU work still ready.
    Yield,
    /// Requests to publish, each returned once.
    NeedsIO(IoBatch),
    /// Rows the plan produced.
    Piece(Piece),
}

/// Adapts a pipeline [`Scan`] to the planner I/O and compute protocol.
///
/// The scan names segments by id; the protocol names byte ranges. Each scan read becomes a
/// `Fetch` of the segment's range at the segment's alignment, and each delivery is handed back to
/// the scan under its own id.
pub(super) struct ScanAdaptor {
    scan: Scan,
    /// A turn already taken while batching reads, retained for the next compute.
    pending: Option<Turn>,
    done: bool,
    locations: Arc<[SegmentLocation]>,
    /// Published and undelivered requests, with the scan read and segment each one answers.
    outstanding: FxHashMap<u32, (ReadId, SegmentId)>,
    /// A delivery the scan could not take, reported by the next compute.
    failed: Option<VortexError>,
    /// The protocol id the next published request gets.
    next_id: u32,
}

impl ScanAdaptor {
    /// Wraps a scan, numbering requests uniquely within this adaptor's lifetime.
    pub(super) fn new(scan: Scan, locations: Arc<[SegmentLocation]>) -> Self {
        Self {
            scan,
            pending: None,
            done: false,
            locations,
            outstanding: FxHashMap::default(),
            failed: None,
            next_id: 0,
        }
    }

    /// The owner's protocol state while the scan is running.
    pub(super) fn state(&self) -> State {
        match &self.pending {
            _ if self.failed.is_some() => State::NeedsCompute,
            _ if self.done => State::Done,
            Some(Turn::Waiting) => State::Waiting,
            _ => State::NeedsCompute,
        }
    }

    pub(super) fn compute(&mut self) -> VortexResult<AdaptorStep> {
        if let Some(error) = self.failed.take() {
            return Err(error);
        }
        Ok(match self.step()? {
            Turn::Waiting | Turn::Done => AdaptorStep::Yield,
            Turn::Output(split, array) => AdaptorStep::Piece(Piece { split, array }),
            Turn::Read(read) => {
                // Publish every read the scan has ready together, so the owner can coalesce them.
                let mut batch = vec![self.request(read.id, read.segment_id)?];
                loop {
                    match self.step()? {
                        Turn::Read(read) => batch.push(self.request(read.id, read.segment_id)?),
                        turn => {
                            if !matches!(turn, Turn::Waiting) {
                                self.pending = Some(turn);
                                self.done = false;
                            }
                            break;
                        }
                    }
                }
                AdaptorStep::NeedsIO(batch)
            }
        })
    }

    fn step(&mut self) -> VortexResult<Turn> {
        let turn = match self.pending.take() {
            Some(Turn::Waiting) | None => self.scan.step()?,
            Some(turn) => turn,
        };
        match &turn {
            Turn::Done => self.done = true,
            Turn::Waiting => self.pending = Some(Turn::Waiting),
            Turn::Read(_) | Turn::Output(..) => {}
        }
        Ok(turn)
    }

    fn request(&mut self, read: ReadId, segment_id: SegmentId) -> VortexResult<request::IoRequest> {
        let location = self.location(segment_id)?;
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| vortex_err!("ScanAdaptor ran out of request ids"))?;
        self.outstanding.insert(id, (read, segment_id));
        Ok(request::IoRequest {
            intent: IoIntent::Fetch,
            request: request::IoRequestId(id),
            target: location.target(),
        })
    }

    /// Hands a delivered read to the scan. The driver only delivers requests this adaptor
    /// published, so anything else is a driver bug. Bytes the scan cannot take fail the owner's
    /// next compute.
    pub(super) fn set_io_result(&mut self, id: request::IoRequestId, result: IoResult) {
        let Some((read, segment_id)) = self.outstanding.remove(&id.0) else {
            vortex_panic!("ScanAdaptor: delivery of {id:?}, which is not outstanding");
        };
        let IoResult::Bytes(bytes) = result else {
            vortex_panic!("ScanAdaptor: segment {segment_id} answered with a size");
        };
        if let Err(error) = self.deliver(read, segment_id, bytes)
            && self.failed.is_none()
        {
            self.failed = Some(error.with_context(format!("delivering segment {segment_id}")));
        }
    }

    fn deliver(
        &mut self,
        read: ReadId,
        segment_id: SegmentId,
        bytes: BufferHandle,
    ) -> VortexResult<()> {
        // A source that honours the target's alignment makes this a check, not a copy.
        let alignment = self.location(segment_id)?.alignment;
        let bytes = BufferHandle::new_host(bytes.try_into_host_sync()?.aligned(alignment));
        self.scan.deliver(read, bytes)?;
        if matches!(self.pending, Some(Turn::Waiting)) {
            self.pending = None;
        }
        Ok(())
    }

    fn location(&self, id: SegmentId) -> VortexResult<SegmentLocation> {
        self.locations
            .get(*id as usize)
            .copied()
            .ok_or_else(|| vortex_err!("segment {id} has no known location"))
    }
}
