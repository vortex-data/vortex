// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_io::request;
use vortex_io::request::IoBatch;
use vortex_io::request::IoIntent;
use vortex_io::request::IoResult;
use vortex_io::request::IoTarget;
use vortex_scan::planning::planner::State;
use vortex_utils::aliases::hash_map::HashMap;

use crate::plan::exec;
use crate::plan::exec::ExecGraph;
use crate::plan::exec::ExecOutput;
use crate::plan::exec::ExecState;
use crate::plan::exec::Piece;
use crate::scan::planning::SegmentLocation;
use crate::segments::SegmentId;

/// What one [`ProtocolGraph::compute`] produced.
pub(crate) enum GraphStep {
    /// Control returns to the owner with CPU work still ready.
    Yield,
    /// Requests to publish, each returned once.
    NeedsIO(IoBatch),
    /// A piece that reached the graph's root.
    Piece(Piece),
}

/// An [`ExecGraph`] whose segment reads are published as protocol requests.
///
/// The graph names segments by id; the protocol names byte ranges. Each graph request becomes a
/// `Fetch` of the segment's range, and each delivery is realigned to the segment's alignment and
/// handed back to the graph under its own id.
pub(crate) struct ProtocolGraph {
    graph: ExecGraph,
    locations: Arc<[SegmentLocation]>,
    /// Published and undelivered requests, with the graph request and segment each one answers.
    outstanding: HashMap<request::IoRequestId, (exec::IoRequestId, SegmentId)>,
    batch: IoBatch,
}

impl ProtocolGraph {
    pub(crate) fn new(graph: ExecGraph, locations: Arc<[SegmentLocation]>) -> Self {
        Self {
            graph,
            locations,
            outstanding: HashMap::default(),
            batch: Vec::new(),
        }
    }

    /// The owner's protocol state while the graph is running.
    pub(crate) fn state(&self) -> State {
        match self.graph.state() {
            ExecState::Done => State::Done,
            ExecState::NeedsCompute => State::NeedsCompute,
            ExecState::Waiting => State::NeedsIO(self.batch.clone()),
        }
    }

    pub(crate) fn compute(&mut self) -> VortexResult<GraphStep> {
        Ok(match self.graph.compute()? {
            ExecOutput::Yield => GraphStep::Yield,
            ExecOutput::Piece(piece) => GraphStep::Piece(piece),
            ExecOutput::NeedsIO(requests) => {
                let mut batch = Vec::with_capacity(requests.len());
                for graph_request in requests {
                    let location = self.location(graph_request.segment_id)?;
                    let id = request::IoRequestId(u32::try_from(graph_request.id.0)?);
                    self.outstanding
                        .insert(id, (graph_request.id, graph_request.segment_id));
                    batch.push(request::IoRequest {
                        intent: IoIntent::Fetch,
                        request: id,
                        target: IoTarget::Range {
                            offset: location.offset,
                            len: location.length as usize,
                        },
                    });
                }
                self.batch.extend(batch.iter().cloned());
                GraphStep::NeedsIO(batch)
            }
        })
    }

    /// Hands a delivered read to the graph. The driver only delivers requests this graph
    /// published, so anything else is a driver bug.
    pub(crate) fn set_io_result(&mut self, id: request::IoRequestId, result: IoResult) {
        let Some((graph_id, segment_id)) = self.outstanding.remove(&id) else {
            vortex_panic!("ProtocolGraph: delivery of {id:?}, which is not outstanding");
        };
        self.batch.retain(|pending| pending.request != id);
        let IoResult::Bytes(bytes) = result else {
            vortex_panic!("ProtocolGraph: segment {segment_id} answered with a size");
        };
        if let Err(error) = self.deliver(graph_id, segment_id, bytes) {
            vortex_panic!("ProtocolGraph: delivery of segment {segment_id} failed: {error}");
        }
    }

    fn deliver(
        &mut self,
        graph_id: exec::IoRequestId,
        segment_id: SegmentId,
        bytes: BufferHandle,
    ) -> VortexResult<()> {
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
