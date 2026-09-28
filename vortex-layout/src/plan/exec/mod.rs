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

mod concat;
mod eval;
mod pack;
mod piece;
mod row_idx;
mod segment_scan;

use std::collections::VecDeque;
use std::fmt;
use std::mem;
use std::ops::Range;

use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::plan::PlanRef;
use crate::segments::SegmentId;

/// Identifies one IO request within an [`ExecGraph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IoRequestId(u64);

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

/// Input pushed into a node's port.
#[derive(Debug)]
pub enum Input {
    /// A piece produced by the child on this port.
    Piece(Piece),
    /// The child on this port will produce no more pieces.
    Closed,
}

/// One running plan operator inside an [`ExecGraph`].
///
/// A node has a single entry point. The graph calls [`compute`](Self::compute) once when the node
/// is spawned, again whenever input has arrived for it, and again when it last returned
/// [`NodeState::Ready`]. Input that arrived since the previous call is taken from the
/// [`StepCx`]; everything the node produces is recorded there too.
pub trait ExecNode {
    /// Consumes pending input, does work, and reports when the node needs to run again.
    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState>;
}

/// When a node needs to run again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeState {
    /// Run again only once new input or an IO result arrives.
    Waiting,
    /// Run again even without new input, for example after yielding with work left.
    Ready,
    /// The node has closed and will not be run again.
    Done,
}

/// Input that arrived for a node, and the effects of running it, applied by the graph afterwards.
pub struct StepCx<'a> {
    session: &'a VortexSession,
    row_offset: u64,
    next_io_id: &'a mut u64,
    inputs: Vec<(Port, Input)>,
    io: Vec<(IoRequestId, BufferHandle)>,
    spawned: Vec<(Port, PlanRef, Range<u64>, Mask)>,
    requests: IoBatch,
    emitted: Vec<Piece>,
    closed: bool,
    yielded: bool,
}

impl<'a> StepCx<'a> {
    fn new(
        session: &'a VortexSession,
        row_offset: u64,
        next_io_id: &'a mut u64,
        inbox: Inbox,
    ) -> Self {
        Self {
            session,
            row_offset,
            next_io_id,
            inputs: inbox.inputs,
            io: inbox.io,
            spawned: Vec::new(),
            requests: Vec::new(),
            emitted: Vec::new(),
            closed: false,
            yielded: false,
        }
    }

    /// The session used for decoding and expression evaluation.
    pub fn session(&self) -> &VortexSession {
        self.session
    }

    /// The global row index of the graph's first plan row.
    pub fn row_offset(&self) -> u64 {
        self.row_offset
    }

    /// Takes the child input that arrived since the previous call, in arrival order.
    pub fn take_inputs(&mut self) -> Vec<(Port, Input)> {
        mem::take(&mut self.inputs)
    }

    /// Takes the results of this node's requests that arrived since the previous call.
    pub fn take_io(&mut self) -> Vec<(IoRequestId, BufferHandle)> {
        mem::take(&mut self.io)
    }

    /// Spawns `plan` over `rows` of its own domain, restricted to `mask`, feeding `port`.
    pub fn spawn(&mut self, port: Port, plan: PlanRef, rows: Range<u64>, mask: Mask) {
        self.spawned.push((port, plan, rows, mask));
    }

    /// Publishes a read of `segment_id` and returns the id its result is delivered under.
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

    /// Tells the parent this node will emit nothing more.
    pub fn close(&mut self) {
        self.closed = true;
    }

    /// Gives the worker back after this call.
    pub fn yield_now(&mut self) {
        self.yielded = true;
    }
}

/// Input waiting for a node's next [`ExecNode::compute`].
#[derive(Default)]
struct Inbox {
    inputs: Vec<(Port, Input)>,
    io: Vec<(IoRequestId, BufferHandle)>,
}

impl Inbox {
    fn is_empty(&self) -> bool {
        self.inputs.is_empty() && self.io.is_empty()
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
    session: VortexSession,
    row_offset: u64,
    nodes: Vec<Box<dyn ExecNode>>,
    parents: Vec<Option<(NodeId, Port)>>,
    inboxes: Vec<Inbox>,
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
    /// by row-index plans.
    ///
    /// Construction does no IO. Every leaf's first request is returned by the first
    /// [`compute`](Self::compute).
    pub fn try_new(
        session: VortexSession,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: Mask,
        row_offset: u64,
    ) -> VortexResult<Self> {
        let mut graph = Self {
            session,
            row_offset,
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
        self.inboxes[node].io.push((id, result));
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
        self.nodes.push(plan.exec(rows, mask)?);
        self.parents.push(parent);
        self.inboxes.push(Inbox::default());
        self.status.push(Status::Queued);
        self.run(id)?;
        Ok(())
    }

    /// Runs one node's compute with its pending input and applies the effects. Returns whether
    /// the node yielded.
    fn run(&mut self, node: NodeId) -> VortexResult<bool> {
        let inbox = mem::take(&mut self.inboxes[node]);
        let mut next_io_id = self.next_io_id;
        let mut cx = StepCx::new(&self.session, self.row_offset, &mut next_io_id, inbox);
        let state = self.nodes[node].compute(&mut cx)?;
        let effects = Effects::from(cx);
        self.next_io_id = next_io_id;

        // Settle the node before waking its parent, so a re-queued node runs after the parent.
        self.status[node] = match state {
            NodeState::Done => Status::Done,
            NodeState::Waiting | NodeState::Ready => Status::Idle,
        };
        if state == NodeState::Ready || !self.inboxes[node].is_empty() {
            self.enqueue(node);
        }
        let yielded = effects.yielded;
        self.apply(node, effects)?;
        Ok(yielded)
    }

    fn apply(&mut self, node: NodeId, effects: Effects) -> VortexResult<()> {
        for (port, plan, rows, mask) in effects.spawned {
            self.spawn(Some((node, port)), &plan, rows, mask)?;
        }
        for request in effects.requests {
            self.io_routes.insert(request.id, node);
            self.new_io.push(request);
        }
        for piece in effects.emitted {
            self.deliver(node, Input::Piece(piece));
        }
        if effects.closed {
            self.deliver(node, Input::Closed);
        }
        Ok(())
    }

    fn deliver(&mut self, from: NodeId, input: Input) {
        match self.parents[from] {
            None => match input {
                Input::Piece(piece) => self.outputs.push_back(piece),
                Input::Closed => self.root_closed = true,
            },
            Some((parent, port)) => {
                self.inboxes[parent].inputs.push((port, input));
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
    closed: bool,
    yielded: bool,
}

impl From<StepCx<'_>> for Effects {
    fn from(cx: StepCx<'_>) -> Self {
        Self {
            spawned: cx.spawned,
            requests: cx.requests,
            emitted: cx.emitted,
            closed: cx.closed,
            yielded: cx.yielded,
        }
    }
}

pub(crate) use concat::ConcatNode;
pub(crate) use eval::EvalNode;
pub(crate) use pack::PackNode;
pub(crate) use piece::Selection;
pub(crate) use row_idx::RowIdxNode;
pub(crate) use segment_scan::SegmentScanNode;

#[cfg(test)]
mod tests;
