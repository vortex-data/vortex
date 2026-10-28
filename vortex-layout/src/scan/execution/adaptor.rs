// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Adapts pipeline execution to the planner I/O and compute protocol.

use std::ops::Range;
use std::sync::Arc;

use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_io::request;
use vortex_io::request::IoBatch;
use vortex_io::request::IoIntent;
use vortex_io::request::IoResult;
use vortex_io::request::IoTarget;
use vortex_scan::planning::planner::State;

use crate::plan::physical::pipeline::ReadId;
use crate::plan::physical::pipeline::ReadRequest;
use crate::plan::physical::pipeline::Scan;
use crate::plan::physical::pipeline::Turn;
use crate::scan::execution::SegmentLocation;
use crate::scan::execution::primitive::value_offset;
use crate::segments::SegmentId;

/// Rows a running plan produced, in row order.
#[derive(Clone)]
pub(super) struct Piece {
    /// Orders the piece among the outputs of one scan.
    pub(super) rows: Range<u64>,
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
    /// Orders the pieces handed out.
    next_piece: u64,
    locations: Arc<[SegmentLocation]>,
    /// Published and undelivered requests, with the scan read and segment each one answers.
    outstanding: FxHashMap<u32, Delivery>,
    queued: IoBatch,
    sparse: FxHashMap<ReadId, SparseRead>,
    /// A delivery the scan could not take, reported by the next compute.
    failed: Option<VortexError>,
    /// The protocol id the next published request gets.
    next_id: u32,
    subsegment_reads: bool,
}

enum Delivery {
    Whole(ReadId, SegmentId),
    Probe(ReadRequest, usize, Vec<Range<usize>>),
    Part(ReadId, Range<usize>),
}

struct SparseRead {
    segment: SegmentId,
    buffer: ByteBufferMut,
    remaining: usize,
}

impl ScanAdaptor {
    /// Wraps a scan, numbering requests uniquely within this adaptor's lifetime.
    pub(super) fn new(
        scan: Scan,
        locations: Arc<[SegmentLocation]>,
        subsegment_reads: bool,
    ) -> Self {
        Self {
            scan,
            pending: None,
            done: false,
            next_piece: 0,
            locations,
            outstanding: FxHashMap::default(),
            queued: Vec::new(),
            sparse: FxHashMap::default(),
            failed: None,
            next_id: 0,
            subsegment_reads,
        }
    }

    /// The owner's protocol state while the scan is running.
    pub(super) fn state(&self) -> State {
        match &self.pending {
            _ if self.failed.is_some() || !self.queued.is_empty() => State::NeedsCompute,
            _ if self.done => State::Done,
            Some(Turn::Waiting) => State::Waiting,
            _ => State::NeedsCompute,
        }
    }

    pub(super) fn compute(&mut self) -> VortexResult<AdaptorStep> {
        if let Some(error) = self.failed.take() {
            return Err(error);
        }
        if !self.queued.is_empty() {
            return Ok(AdaptorStep::NeedsIO(std::mem::take(&mut self.queued)));
        }
        Ok(match self.step()? {
            Turn::Waiting | Turn::Done => AdaptorStep::Yield,
            Turn::Output(_, array) => AdaptorStep::Piece(self.piece(array)),
            Turn::Read(read) => {
                // Publish every read the scan has ready together, so the owner can coalesce them.
                let mut batch = vec![self.request(read)?];
                loop {
                    match self.step()? {
                        Turn::Read(read) => batch.push(self.request(read)?),
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

    fn piece(&mut self, array: ArrayRef) -> Piece {
        let start = self.next_piece;
        self.next_piece += 1;
        Piece {
            rows: start..start + 1,
            array,
        }
    }

    fn request(&mut self, read: ReadRequest) -> VortexResult<request::IoRequest> {
        let location = self.location(read.segment_id)?;
        let len = location.length as usize;
        if self.subsegment_reads
            && let Some(demand) = &read.primitive
            && len >= demand.buffer_len
            && len - demand.buffer_len <= 4096
            && len > 4096
            && let Some(ranges) = demand.ranges()
        {
            let start = len - 4096;
            return self.publish(
                IoTarget::Range {
                    offset: location.offset + start as u64,
                    len: 4096,
                    alignment: Alignment::none(),
                },
                Delivery::Probe(read, start, ranges),
            );
        }
        self.publish(location.target(), Delivery::Whole(read.id, read.segment_id))
    }

    fn publish(
        &mut self,
        target: IoTarget,
        delivery: Delivery,
    ) -> VortexResult<request::IoRequest> {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| vortex_err!("ScanAdaptor ran out of request ids"))?;
        self.outstanding.insert(id, delivery);
        Ok(request::IoRequest {
            intent: IoIntent::Fetch,
            request: request::IoRequestId(id),
            target,
        })
    }

    pub(super) fn set_io_result(&mut self, id: request::IoRequestId, result: IoResult) {
        if let Err(error) = self.receive(id, result)
            && self.failed.is_none()
        {
            self.failed = Some(error);
        }
    }

    fn receive(&mut self, id: request::IoRequestId, result: IoResult) -> VortexResult<()> {
        let delivery = self.outstanding.remove(&id.0).ok_or_else(|| {
            vortex_err!("ScanAdaptor: delivery of {id:?}, which is not outstanding")
        })?;
        let IoResult::Bytes(bytes) = result else {
            vortex_panic!("ScanAdaptor: range answered with a size");
        };
        match delivery {
            Delivery::Whole(read, segment) => self.deliver(read, segment, bytes),
            Delivery::Probe(read, start, ranges) => {
                let location = self.location(read.segment_id)?;
                let tail = bytes.try_into_host_sync()?;
                vortex_ensure_eq!(tail.len(), location.length as usize - start);
                let offset = read.primitive.as_ref().and_then(|demand| {
                    value_offset(&tail, location.length as usize, demand, &ranges)
                });
                let Some(offset) = offset else {
                    let request =
                        self.publish(location.target(), Delivery::Whole(read.id, read.segment_id))?;
                    self.queued.push(request);
                    return Ok(());
                };
                // The scan source filters these bytes before any row leaves it. Missing values
                // remain initialized, and this private reconstruction is never put in a cache.
                let mut buffer =
                    ByteBufferMut::zeroed_aligned(location.length as usize, location.alignment);
                buffer.as_mut_slice()[start..].copy_from_slice(&tail);
                let ranges: Vec<_> = ranges
                    .into_iter()
                    .filter_map(|range| {
                        let begin = offset + range.start;
                        let end = (offset + range.end).min(start);
                        (begin < end).then_some(begin..end)
                    })
                    .collect();
                if ranges.is_empty() {
                    return self.deliver(
                        read.id,
                        read.segment_id,
                        BufferHandle::new_host(buffer.freeze()),
                    );
                }
                self.sparse.insert(
                    read.id,
                    SparseRead {
                        segment: read.segment_id,
                        buffer,
                        remaining: ranges.len(),
                    },
                );
                for range in ranges {
                    let target = IoTarget::Range {
                        offset: location.offset + range.start as u64,
                        len: range.len(),
                        alignment: Alignment::none(),
                    };
                    let request = self.publish(target, Delivery::Part(read.id, range))?;
                    self.queued.push(request);
                }
                Ok(())
            }
            Delivery::Part(read, range) => {
                let bytes = bytes.try_into_host_sync()?;
                vortex_ensure_eq!(bytes.len(), range.len());
                let pending = self
                    .sparse
                    .get_mut(&read)
                    .ok_or_else(|| vortex_err!("Unknown sparse read"))?;
                pending.buffer.as_mut_slice()[range].copy_from_slice(&bytes);
                pending.remaining -= 1;
                if pending.remaining == 0 {
                    let pending = self
                        .sparse
                        .remove(&read)
                        .ok_or_else(|| vortex_err!("Unknown sparse read"))?;
                    self.deliver(
                        read,
                        pending.segment,
                        BufferHandle::new_host(pending.buffer.freeze()),
                    )?;
                }
                Ok(())
            }
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
