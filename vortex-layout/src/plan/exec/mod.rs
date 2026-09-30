// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Push-based execution of physical plans.
//!
//! A [`PlanRef`] is turned into an [`ExecGraph`] for one row range and selection. The graph is a
//! flat arena of [`ExecNode`]s. IO completions mark leaves ready; running a ready node emits
//! row-tagged [`Piece`]s that are pushed into the parent's input port, which may in turn become
//! ready. Pieces that reach the root are handed to the owner one per [`ExecGraph::compute`].
//!
//! Nodes never talk to the IO service. They publish requests through [`StepCx::request`], the
//! graph returns them from [`ExecGraph::compute`], and the owner delivers results through
//! [`ExecGraph::set_io_result`].
//!
//! A node follows the same model as the planners that own its graph: it is built from
//! construction-time data, it is handed what arrived for it, and it reports what happens next.
//!
//! | Exec node              | Planner                         |
//! |------------------------|---------------------------------|
//! | [`ExecContext`]        | construction-time data          |
//! | [`Event`]s             | [`set_io_result`] deliveries    |
//! | [`NodeState::Wait`]    | `NeedsIO`: waits on deliveries  |
//! | [`NodeState::Yield`]   | `Continue`: requeued            |
//! | [`NodeState::Done`]    | `Done`                          |
//!
//! [`set_io_result`]: vortex_io::request::IoConsumer::set_io_result

mod concat;
mod eval;
mod filter;
mod pack;
mod piece;
mod row_idx;
mod segment_scan;
mod share;
mod take;
mod zoned;

use std::collections::VecDeque;
use std::fmt;
use std::mem;
use std::ops::Range;
use std::sync::Arc;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::plan::PlanRef;
use crate::segments::SegmentId;

/// Identifies one IO request within an [`ExecGraph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IoRequestId(pub(crate) u64);

/// A read published by a leaf node.
#[derive(Clone, Debug)]
pub struct IoRequest {
    /// Identifies the request when its result is delivered.
    pub id: IoRequestId,
    /// The segment to read.
    pub segment_id: SegmentId,
}

/// Requests discovered together.
pub type IoBatch = Vec<IoRequest>;

/// A row-tagged partial result.
///
/// `rows` is in the producing plan's row domain, and `array` holds exactly the selected rows in
/// that range, in row order.
#[derive(Clone)]
pub struct Piece {
    /// The rows this piece covers, including unselected rows.
    pub rows: Range<u64>,
    /// The selected rows in `rows`.
    pub array: ArrayRef,
}

impl fmt::Debug for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Piece")
            .field("rows", &self.rows)
            .field("len", &self.array.len())
            .finish()
    }
}

/// Index of an input port on a node, equal to the child's plan index.
pub type Port = usize;

/// Something that arrived for a node since its previous [`ExecNode::compute`].
#[derive(Debug)]
pub enum Event {
    /// A piece produced by the child on this port.
    Piece(Port, Piece),
    /// The child on this port will produce no more pieces.
    Closed(Port),
    /// The bytes of a segment this node requested.
    Delivered(IoRequestId, BufferHandle),
}

impl Event {
    /// The error for an event `node` cannot receive, such as a delivery to a node that never
    /// requests IO.
    pub(crate) fn unexpected(&self, node: &str) -> VortexError {
        vortex_err!("{node} did not expect {self:?}")
    }
}

/// One running plan operator inside an [`ExecGraph`].
///
/// A node has a single entry point. The graph calls [`compute`](Self::compute) once when the node
/// is spawned, again whenever an [`Event`] has arrived for it, and again when it last returned
/// [`NodeState::Yield`]. The events are taken from the [`StepCx`]; everything the node produces
/// is recorded there too.
pub trait ExecNode: Send {
    /// Consumes what arrived, does work, and reports what happens to the node next.
    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState>;
}

/// What happens to a node after a [`compute`](ExecNode::compute).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeState {
    /// Run again once an event arrives: a child's piece or close, or a requested delivery.
    Wait,
    /// Run again without waiting for an event, after other work: CPU work is left.
    Yield,
    /// Finished: the node's output closes and it never runs again.
    Done,
}

/// What every node of a graph is built with.
#[derive(Clone)]
pub struct ExecContext {
    session: VortexSession,
    row_offset: u64,
    decoded: DecodeCache,
}

impl ExecContext {
    /// The session used for decoding and expression evaluation.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// The global row index of the graph's first plan row.
    pub fn row_offset(&self) -> u64 {
        self.row_offset
    }

    /// Decoded segments shared by the graphs reading the same rows.
    pub fn decoded(&self) -> &DecodeCache {
        &self.decoded
    }
}

/// Decoded segments shared by the graphs that read one row range, so a segment several of them
/// read is fetched and decoded once.
///
/// A segment decodes the same way wherever it appears, so entries are keyed by segment id alone.
#[derive(Clone, Default)]
pub struct DecodeCache(Arc<Mutex<FxHashMap<SegmentId, ArrayRef>>>);

impl DecodeCache {
    /// The whole decoded array of `id`, if a graph sharing this cache decoded it.
    pub(crate) fn get(&self, id: SegmentId) -> Option<ArrayRef> {
        self.0.lock().get(&id).cloned()
    }

    /// Shares the whole decoded array of `id` with the graphs sharing this cache.
    pub(crate) fn insert(&self, id: SegmentId, array: ArrayRef) {
        self.0.lock().insert(id, array);
    }
}

/// What arrived for a node, and the effects of running it, applied by the graph afterwards.
pub struct StepCx<'a> {
    next_io_id: &'a mut u64,
    events: Vec<Event>,
    spawned: Vec<(Port, PlanRef, Range<u64>, Mask)>,
    requests: IoBatch,
    emitted: Vec<Piece>,
}

impl<'a> StepCx<'a> {
    fn new(next_io_id: &'a mut u64, events: Vec<Event>) -> Self {
        Self {
            next_io_id,
            events,
            spawned: Vec::new(),
            requests: Vec::new(),
            emitted: Vec::new(),
        }
    }

    /// Takes what arrived since the previous call, in arrival order.
    pub fn events(&mut self) -> Vec<Event> {
        mem::take(&mut self.events)
    }

    /// Spawns `plan` over `rows` of its own domain, restricted to `mask`, feeding `port`.
    pub fn spawn(&mut self, port: Port, plan: PlanRef, rows: Range<u64>, mask: Mask) {
        self.spawned.push((port, plan, rows, mask));
    }

    /// Publishes a read of `segment_id` and returns the id its bytes are delivered under.
    pub fn request(&mut self, segment_id: SegmentId) -> IoRequestId {
        let id = IoRequestId(*self.next_io_id);
        *self.next_io_id += 1;
        self.requests.push(IoRequest { id, segment_id });
        id
    }

    /// Pushes a piece to the parent.
    pub fn emit(&mut self, piece: Piece) {
        self.emitted.push(piece);
    }
}

/// What an [`ExecGraph`] needs next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecState {
    /// A node is ready, or a piece or requests are waiting to be returned.
    NeedsCompute,
    /// Every remaining node waits on a delivered request.
    Waiting,
    /// The root has closed and every piece has been returned.
    Done,
}

/// The result of one [`ExecGraph::compute`] call.
#[derive(Debug)]
pub enum ExecOutput {
    /// Control returns to the owner. Check [`ExecGraph::state`] for what is needed next.
    Yield,
    /// Requests published since the last flush, each returned exactly once.
    NeedsIO(IoBatch),
    /// A piece that reached the root.
    Piece(Piece),
}

type NodeId = usize;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    /// Waiting for input; not in the ready queue.
    Idle,
    /// In the ready queue.
    Queued,
    /// Closed; never run again.
    Done,
}

/// A running plan: a flat arena of nodes wired child-to-parent.
pub struct ExecGraph {
    ctx: ExecContext,
    nodes: Vec<Box<dyn ExecNode>>,
    parents: Vec<Option<(NodeId, Port)>>,
    /// Events that arrived for each node since its last compute.
    inboxes: Vec<Vec<Event>>,
    status: Vec<Status>,
    /// Ready nodes, run last-in first-out so a woken parent runs before its child's next call.
    ready: Vec<NodeId>,
    io_routes: FxHashMap<IoRequestId, NodeId>,
    new_io: IoBatch,
    outputs: VecDeque<Piece>,
    root_closed: bool,
    next_io_id: u64,
}

impl ExecGraph {
    /// Builds the graph for `plan` over `rows`, restricted to `mask`, and runs each node's first
    /// [`ExecNode::compute`]. `row_offset` is the global row index of the plan's first row, used
    /// by row-index plans. Segments found in `decoded` are neither read nor decoded again, and
    /// segments the graph decodes are added to it.
    ///
    /// Construction does no IO. Every leaf's first request is returned by the first
    /// [`compute`](Self::compute).
    pub fn try_new(
        session: VortexSession,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: Mask,
        row_offset: u64,
        decoded: DecodeCache,
    ) -> VortexResult<Self> {
        let mut graph = Self {
            ctx: ExecContext {
                session,
                row_offset,
                decoded,
            },
            nodes: Vec::new(),
            parents: Vec::new(),
            inboxes: Vec::new(),
            status: Vec::new(),
            ready: Vec::new(),
            io_routes: FxHashMap::default(),
            new_io: Vec::new(),
            outputs: VecDeque::new(),
            root_closed: false,
            next_io_id: 0,
        };
        graph.spawn(None, plan, rows, mask)?;
        Ok(graph)
    }

    /// Reports what the graph needs next. Cheap and side-effect free.
    pub fn state(&self) -> ExecState {
        if !self.outputs.is_empty() || !self.ready.is_empty() || !self.new_io.is_empty() {
            ExecState::NeedsCompute
        } else if self.root_closed {
            ExecState::Done
        } else {
            ExecState::Waiting
        }
    }

    /// Ids of requests returned by [`compute`](Self::compute) whose results are not yet delivered.
    pub fn outstanding(&self) -> impl Iterator<Item = IoRequestId> + '_ {
        let flushed = |id: &&IoRequestId| !self.new_io.iter().any(|request| request.id == **id);
        self.io_routes.keys().filter(flushed).copied()
    }

    /// Runs ready nodes until a piece reaches the root, requests should be flushed, a node
    /// yields, or nothing is ready.
    pub fn compute(&mut self) -> VortexResult<ExecOutput> {
        loop {
            if let Some(piece) = self.outputs.pop_front() {
                return Ok(ExecOutput::Piece(piece));
            }
            let Some(node) = self.ready.pop() else {
                return Ok(self.flush_or_yield());
            };
            let yielded = self.run(node)?;

            // Get reads in flight before spending more CPU on other nodes.
            if !self.new_io.is_empty() && self.outputs.is_empty() {
                return Ok(ExecOutput::NeedsIO(mem::take(&mut self.new_io)));
            }
            if yielded && self.outputs.is_empty() {
                return Ok(ExecOutput::Yield);
            }
        }
    }

    /// Delivers the result of a request returned by [`compute`](Self::compute).
    pub fn set_io_result(&mut self, id: IoRequestId, result: BufferHandle) -> VortexResult<()> {
        let node = self
            .io_routes
            .remove(&id)
            .ok_or_else(|| vortex_err!("Unknown IO request {id:?}"))?;
        self.inboxes[node].push(Event::Delivered(id, result));
        self.enqueue(node);
        Ok(())
    }

    fn flush_or_yield(&mut self) -> ExecOutput {
        if self.new_io.is_empty() {
            ExecOutput::Yield
        } else {
            ExecOutput::NeedsIO(mem::take(&mut self.new_io))
        }
    }

    fn spawn(
        &mut self,
        parent: Option<(NodeId, Port)>,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: Mask,
    ) -> VortexResult<()> {
        let id = self.nodes.len();
        self.nodes.push(plan.exec(rows, mask, &self.ctx)?);
        self.parents.push(parent);
        self.inboxes.push(Vec::new());
        self.status.push(Status::Queued);
        self.run(id)?;
        Ok(())
    }

    /// Runs one node's compute with the events that arrived for it and applies the effects.
    /// Returns whether control goes back to the owner: when the node yields, and after a node
    /// handles a delivery, since decoding and evaluating newly read data is the costly work
    /// between reads that other owners' work should interleave with.
    fn run(&mut self, node: NodeId) -> VortexResult<bool> {
        let events = mem::take(&mut self.inboxes[node]);
        let delivered = events
            .iter()
            .any(|event| matches!(event, Event::Delivered(..)));
        let mut next_io_id = self.next_io_id;
        let mut cx = StepCx::new(&mut next_io_id, events);
        let state = self.nodes[node].compute(&mut cx)?;
        let effects = Effects::from(cx);
        self.next_io_id = next_io_id;

        // Settle the node before waking its parent, so a re-queued node runs after the parent.
        self.status[node] = match state {
            NodeState::Done => Status::Done,
            NodeState::Wait | NodeState::Yield => Status::Idle,
        };
        if state == NodeState::Yield || !self.inboxes[node].is_empty() {
            self.enqueue(node);
        }
        self.apply(node, effects, state == NodeState::Done)?;
        Ok(state == NodeState::Yield || delivered)
    }

    fn apply(&mut self, node: NodeId, effects: Effects, done: bool) -> VortexResult<()> {
        for (port, plan, rows, mask) in effects.spawned {
            self.spawn(Some((node, port)), &plan, rows, mask)?;
        }
        for request in effects.requests {
            self.io_routes.insert(request.id, node);
            self.new_io.push(request);
        }
        for piece in effects.emitted {
            self.deliver(node, Some(piece));
        }
        if done {
            self.deliver(node, None);
        }
        Ok(())
    }

    /// Passes a piece, or the close when `piece` is `None`, from `from` to its parent.
    fn deliver(&mut self, from: NodeId, piece: Option<Piece>) {
        match self.parents[from] {
            None => match piece {
                Some(piece) => self.outputs.push_back(piece),
                None => self.root_closed = true,
            },
            Some((parent, port)) => {
                self.inboxes[parent].push(match piece {
                    Some(piece) => Event::Piece(port, piece),
                    None => Event::Closed(port),
                });
                self.enqueue(parent);
            }
        }
    }

    fn enqueue(&mut self, node: NodeId) {
        if self.status[node] == Status::Idle {
            self.status[node] = Status::Queued;
            self.ready.push(node);
        }
    }
}

/// The owned effects of a [`StepCx`], detached from its borrows.
struct Effects {
    spawned: Vec<(Port, PlanRef, Range<u64>, Mask)>,
    requests: IoBatch,
    emitted: Vec<Piece>,
}

impl From<StepCx<'_>> for Effects {
    fn from(cx: StepCx<'_>) -> Self {
        Self {
            spawned: cx.spawned,
            requests: cx.requests,
            emitted: cx.emitted,
        }
    }
}

pub(crate) use concat::ConcatNode;
pub(crate) use eval::EvalNode;
pub(crate) use filter::FilterNode;
pub(crate) use pack::PackNode;
pub(crate) use piece::Selection;
pub(crate) use row_idx::RowIdxNode;
pub(crate) use segment_scan::SegmentScanNode;
pub(crate) use share::ShareNode;
pub(crate) use take::TakeNode;
pub(crate) use zoned::ZonePruneNode;

#[cfg(test)]
mod tests;
