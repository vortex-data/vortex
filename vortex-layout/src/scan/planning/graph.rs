// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use smallvec::SmallVec;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_io::request;
use vortex_io::request::IoBatch;
use vortex_io::request::IoIntent;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_scan::planning::planner::State;
use vortex_session::VortexSession;

use crate::plan::PlanRef;
use crate::plan::exec;
use crate::plan::exec::DecodeCache;
use crate::plan::exec::ExecGraph;
use crate::plan::exec::ExecOutput;
use crate::plan::exec::ExecState;
use crate::plan::exec::Piece;
use crate::plan::pipeline::PipelineGraph;
use crate::scan::planning::SegmentLocation;
use crate::segments::SegmentId;

/// The environment variable that runs plans on the experimental [`PipelineGraph`] when set to
/// `pipeline`, in place of [`ExecGraph`]. It exists to compare the two executors.
pub const EXEC_ENV_VAR: &str = "VORTEX_SCAN_EXEC";

static PIPELINES: LazyLock<bool> =
    LazyLock::new(|| env::var(EXEC_ENV_VAR).is_ok_and(|v| v == "pipeline"));

/// A running plan on the executor [`EXEC_ENV_VAR`] selects. Both are driven the same way.
pub(crate) enum ScanGraph {
    Nodes(ExecGraph),
    Pipelines(PipelineGraph),
}

impl ScanGraph {
    /// Builds the graph for `plan` as [`ExecGraph::try_new`] does.
    pub(crate) fn try_new(
        session: VortexSession,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: Mask,
        row_offset: u64,
        decoded: DecodeCache,
    ) -> VortexResult<Self> {
        Ok(if *PIPELINES {
            Self::Pipelines(PipelineGraph::try_new(
                session, plan, rows, mask, row_offset, decoded,
            )?)
        } else {
            Self::Nodes(ExecGraph::try_new(
                session, plan, rows, mask, row_offset, decoded,
            )?)
        })
    }

    fn state(&self) -> ExecState {
        match self {
            Self::Nodes(graph) => graph.state(),
            Self::Pipelines(graph) => graph.state(),
        }
    }

    fn compute(&mut self) -> VortexResult<ExecOutput> {
        match self {
            Self::Nodes(graph) => graph.compute(),
            Self::Pipelines(graph) => graph.compute(),
        }
    }

    fn set_io_result(&mut self, id: exec::IoRequestId, result: BufferHandle) -> VortexResult<()> {
        match self {
            Self::Nodes(graph) => graph.set_io_result(id, result),
            Self::Pipelines(graph) => graph.set_io_result(id, result),
        }
    }
}

/// What one [`ProtocolGraph::compute`] produced.
pub(crate) enum GraphStep {
    /// Control returns to the owner with CPU work still ready.
    Yield,
    /// Requests to publish, each returned once.
    NeedsIO(IoBatch),
    /// A piece that reached the graph's root.
    Piece(Piece),
}

/// A [`ScanGraph`] whose segment reads are published as protocol requests.
///
/// The graph names segments by id; the protocol names byte ranges. Each graph request becomes a
/// `Fetch` of the segment's range at the segment's alignment, and each delivery is handed back to
/// the graph under its own id.
pub(crate) struct ProtocolGraph {
    graph: ScanGraph,
    locations: Arc<[SegmentLocation]>,
    /// Published and undelivered requests, with the graph request and segment each one answers.
    outstanding: SmallVec<[Option<(exec::IoRequestId, SegmentId)>; 2]>,
    first_id: u32,
    /// A delivery the graph could not take, reported by the next compute.
    failed: Option<VortexError>,
    /// The protocol id the next published request gets.
    next_id: u32,
}

impl ProtocolGraph {
    /// Wraps `graph`, numbering its requests from `first_id`.
    ///
    /// Request ids must be unique over an owner's lifetime, so an owner that runs several graphs
    /// starts each where the previous one's [`next_id`](Self::next_id) left off.
    pub(crate) fn new(graph: ScanGraph, locations: Arc<[SegmentLocation]>, first_id: u32) -> Self {
        Self {
            graph,
            locations,
            outstanding: SmallVec::new(),
            first_id,
            failed: None,
            next_id: first_id,
        }
    }

    /// The protocol id the next request would get: the first id a following graph of the same
    /// owner may use.
    pub(crate) fn next_id(&self) -> u32 {
        self.next_id
    }

    /// The owner's protocol state while the graph is running.
    pub(crate) fn state(&self) -> State {
        if self.failed.is_some() {
            return State::NeedsCompute;
        }
        match self.graph.state() {
            ExecState::Done => State::Done,
            ExecState::NeedsCompute => State::NeedsCompute,
            ExecState::Waiting => State::Waiting,
        }
    }

    pub(crate) fn compute(&mut self) -> VortexResult<GraphStep> {
        if let Some(error) = self.failed.take() {
            return Err(error);
        }
        Ok(match self.graph.compute()? {
            ExecOutput::Yield => GraphStep::Yield,
            ExecOutput::Piece(piece) => GraphStep::Piece(piece),
            ExecOutput::NeedsIO(requests) => {
                let mut batch = Vec::with_capacity(requests.len());
                for graph_request in requests {
                    let location = self.location(graph_request.segment_id)?;
                    let id = request::IoRequestId(self.next_id);
                    self.next_id = self
                        .next_id
                        .checked_add(1)
                        .ok_or_else(|| vortex_err!("ProtocolGraph ran out of request ids"))?;
                    self.outstanding
                        .push(Some((graph_request.id, graph_request.segment_id)));
                    batch.push(request::IoRequest {
                        intent: IoIntent::Fetch,
                        request: id,
                        target: location.target(),
                    });
                }
                GraphStep::NeedsIO(batch)
            }
        })
    }

    /// Hands a delivered read to the graph. The driver only delivers requests this graph
    /// published, so anything else is a driver bug. Bytes the graph cannot take fail the owner's
    /// next compute.
    pub(crate) fn set_io_result(&mut self, id: request::IoRequestId, result: IoResult) {
        let pending = id
            .0
            .checked_sub(self.first_id)
            .and_then(|index| self.outstanding.get_mut(index as usize))
            .and_then(Option::take);
        let Some((graph_id, segment_id)) = pending else {
            vortex_panic!("ProtocolGraph: delivery of {id:?}, which is not outstanding");
        };
        let IoResult::Bytes(bytes) = result else {
            vortex_panic!("ProtocolGraph: segment {segment_id} answered with a size");
        };
        if let Err(error) = self.deliver(graph_id, segment_id, bytes)
            && self.failed.is_none()
        {
            self.failed = Some(error.with_context(format!("delivering segment {segment_id}")));
        }
    }

    fn deliver(
        &mut self,
        graph_id: exec::IoRequestId,
        segment_id: SegmentId,
        bytes: BufferHandle,
    ) -> VortexResult<()> {
        // A source that honours the target's alignment makes this a check, not a copy.
        let alignment = self.location(segment_id)?.alignment;
        let bytes = BufferHandle::new_host(bytes.try_into_host_sync()?.aligned(alignment));
        self.graph.set_io_result(graph_id, bytes)
    }

    fn location(&self, id: SegmentId) -> VortexResult<SegmentLocation> {
        self.locations
            .get(*id as usize)
            .copied()
            .ok_or_else(|| vortex_err!("segment {id} has no known location"))
    }
}
