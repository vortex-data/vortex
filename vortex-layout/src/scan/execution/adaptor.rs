// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Adapts pipeline execution to the planner I/O and compute protocol.

use std::ops::Range;
use std::sync::Arc;

use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_io::segments::IoIntent;
use vortex_io::segments::IoKey;
use vortex_io::segments::IoRequest;
use vortex_io::segments::SegmentSourceId;
use vortex_io::segments::SourceRequest;
use vortex_io::segments::SourceResult;
use vortex_scan::planning::planner::State;

use crate::plan::physical::pipeline::ReadId;
use crate::plan::physical::pipeline::Scan;
use crate::plan::physical::pipeline::Turn;
use crate::scan::execution::SegmentLocation;
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
    NeedsIO(Vec<IoRequest>),
    /// Rows the plan produced.
    Piece(Piece),
}

/// Adapts a pipeline [`Scan`] to the planner I/O and compute protocol.
///
/// Each scan read becomes a `Fetch` of its segment in the file's source, and each delivery is
/// handed back to every scan read of that segment. A segment is fetched once while any read of it
/// is outstanding.
pub(super) struct ScanAdaptor {
    scan: Scan,
    /// A turn already taken while batching reads, retained for the next compute.
    pending: Option<Turn>,
    done: bool,
    /// Orders the pieces handed out.
    next_piece: u64,
    locations: Arc<[SegmentLocation]>,
    source: SegmentSourceId,
    /// Segments fetched and not yet delivered, with the scan reads waiting for each.
    outstanding: FxHashMap<u32, Vec<ReadId>>,
    /// A delivery the scan could not take, reported by the next compute.
    failed: Option<VortexError>,
}

impl ScanAdaptor {
    /// Wraps a scan of the file registered as `source`.
    pub(super) fn new(
        scan: Scan,
        locations: Arc<[SegmentLocation]>,
        source: SegmentSourceId,
    ) -> Self {
        Self {
            scan,
            pending: None,
            done: false,
            next_piece: 0,
            locations,
            source,
            outstanding: FxHashMap::default(),
            failed: None,
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
            Turn::Output(_, array) => AdaptorStep::Piece(self.piece(array)),
            Turn::Read(read) => {
                // Publish every read the scan has ready together, so the owner can coalesce them.
                let mut batch = Vec::new();
                self.request(read.id, read.segment_id, &mut batch)?;
                loop {
                    match self.step()? {
                        Turn::Read(read) => self.request(read.id, read.segment_id, &mut batch)?,
                        turn => {
                            if !matches!(turn, Turn::Waiting) {
                                self.pending = Some(turn);
                                self.done = false;
                            }
                            break;
                        }
                    }
                }
                if batch.is_empty() {
                    // Every read rode on a fetch already outstanding.
                    AdaptorStep::Yield
                } else {
                    AdaptorStep::NeedsIO(batch)
                }
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

    /// Records a scan read of `segment_id`, adding a fetch to `batch` unless one of the segment
    /// is already outstanding.
    fn request(
        &mut self,
        read: ReadId,
        segment_id: SegmentId,
        batch: &mut Vec<IoRequest>,
    ) -> VortexResult<()> {
        self.location(segment_id)?;
        let reads = self.outstanding.entry(*segment_id).or_default();
        if reads.is_empty() {
            batch.push(IoRequest::new(
                IoKey::segment(self.source, *segment_id),
                IoIntent::Fetch,
            ));
        }
        reads.push(read);
        Ok(())
    }

    /// Hands a delivered segment to every scan read waiting for it. The driver only delivers
    /// fetches this adaptor published, so anything else is a driver bug. Bytes the scan cannot
    /// take fail the owner's next compute.
    pub(super) fn set_result(&mut self, key: IoKey, result: SourceResult) {
        let SourceRequest::Segment(segment) = key.request else {
            vortex_panic!("ScanAdaptor: delivery of {key:?}, though it only fetches segments");
        };
        let segment_id = SegmentId::from(segment);
        let Some(reads) = self.outstanding.remove(&segment) else {
            vortex_panic!("ScanAdaptor: delivery of {key:?}, which is not outstanding");
        };
        let delivered = result.into_bytes().and_then(|bytes| {
            reads
                .into_iter()
                .try_for_each(|read| self.deliver(read, segment_id, bytes.clone()))
        });
        if let Err(error) = delivered
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
