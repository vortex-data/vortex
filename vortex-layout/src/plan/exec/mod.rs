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
mod pack;
mod piece;
mod segment_scan;

use std::collections::VecDeque;
use std::fmt;
use std::mem;
use std::ops::Range;

use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
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
/// Nodes are built by [`PlanRef::exec`] and started once by the graph. They receive input through
/// [`on_input`](Self::on_input) and [`on_io`](Self::on_io), which only buffer, and do their work
/// in [`step`](Self::step).
pub trait ExecNode {
    /// Spawns children and publishes the node's first requests.
    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<()>;

    /// Whether [`step`](Self::step) can make progress.
    fn is_ready(&self) -> bool;

    /// Performs one unit of work. Only called when [`is_ready`](Self::is_ready) is true.
    fn step(&mut self, cx: &mut StepCx<'_>) -> VortexResult<()>;

    /// Buffers input from the child on `port`.
    fn on_input(&mut self, port: Port, input: Input) -> VortexResult<()> {
        drop(input);
        vortex_bail!("Exec node does not accept input on port {port}")
    }

    /// Buffers the result of a request this node published.
    fn on_io(&mut self, id: IoRequestId, result: BufferHandle) -> VortexResult<()> {
        drop(result);
        vortex_bail!("Exec node did not request {id:?}")
    }
}

/// Effects of starting or stepping a node, applied by the graph afterwards.
pub struct StepCx<'a> {
    session: &'a VortexSession,
    next_io_id: &'a mut u64,
    spawned: Vec<(Port, PlanRef, Range<u64>, Mask)>,
    requests: IoBatch,
    emitted: Vec<Piece>,
    closed: bool,
    yielded: bool,
}

impl<'a> StepCx<'a> {
    fn new(session: &'a VortexSession, next_io_id: &'a mut u64) -> Self {
        Self {
            session,
            next_io_id,
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

    /// Gives the worker back after this step.
    pub fn yield_now(&mut self) {
        self.yielded = true;
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

/// A running plan: a flat arena of nodes wired child-to-parent.
pub struct ExecGraph {
    session: VortexSession,
    nodes: Vec<Box<dyn ExecNode>>,
    parents: Vec<Option<(NodeId, Port)>>,
    queued: Vec<bool>,
    /// Ready nodes, run last-in first-out so a woken parent runs before its child's next step.
    ready: Vec<NodeId>,
    io_routes: FxHashMap<IoRequestId, NodeId>,
    new_io: IoBatch,
    outputs: VecDeque<Piece>,
    root_closed: bool,
    next_io_id: u64,
}

impl ExecGraph {
    /// Builds and starts the graph for `plan` over `rows`, restricted to `mask`.
    ///
    /// Construction does no IO. Every leaf's first request is returned by the first
    /// [`compute`](Self::compute).
    pub fn try_new(
        session: VortexSession,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: Mask,
    ) -> VortexResult<Self> {
        let mut graph = Self {
            session,
            nodes: Vec::new(),
            parents: Vec::new(),
            queued: Vec::new(),
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
            self.queued[node] = false;

            let mut next_io_id = self.next_io_id;
            let mut cx = StepCx::new(&self.session, &mut next_io_id);
            self.nodes[node].step(&mut cx)?;
            let effects = Effects::from(cx);
            self.next_io_id = next_io_id;
            let yielded = effects.yielded;
            self.apply(node, effects)?;

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
        self.nodes[node].on_io(id, result)?;
        self.enqueue_if_ready(node);
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
        let mut node = plan.exec(rows, mask)?;
        let id = self.nodes.len();

        let mut next_io_id = self.next_io_id;
        let mut cx = StepCx::new(&self.session, &mut next_io_id);
        node.start(&mut cx)?;
        let effects = Effects::from(cx);
        self.next_io_id = next_io_id;

        self.nodes.push(node);
        self.parents.push(parent);
        self.queued.push(false);
        self.apply(id, effects)
    }

    fn apply(&mut self, node: NodeId, effects: Effects) -> VortexResult<()> {
        for (port, plan, rows, mask) in effects.spawned {
            self.spawn(Some((node, port)), &plan, rows, mask)?;
        }
        for request in effects.requests {
            self.io_routes.insert(request.id, node);
            self.new_io.push(request);
        }
        // Re-queue the node before waking its parent, so the parent runs first.
        self.enqueue_if_ready(node);
        for piece in effects.emitted {
            self.deliver(node, Input::Piece(piece))?;
        }
        if effects.closed {
            self.deliver(node, Input::Closed)?;
        }
        Ok(())
    }

    fn deliver(&mut self, from: NodeId, input: Input) -> VortexResult<()> {
        match self.parents[from] {
            None => match input {
                Input::Piece(piece) => self.outputs.push_back(piece),
                Input::Closed => self.root_closed = true,
            },
            Some((parent, port)) => {
                self.nodes[parent].on_input(port, input)?;
                self.enqueue_if_ready(parent);
            }
        }
        Ok(())
    }

    fn enqueue_if_ready(&mut self, node: NodeId) {
        if !self.queued[node] && self.nodes[node].is_ready() {
            self.queued[node] = true;
            self.ready.push(node);
        }
    }
}

/// The owned part of a [`StepCx`], detached from its borrows.
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
pub(crate) use pack::PackNode;
pub(crate) use piece::Selection;
pub(crate) use segment_scan::SegmentScanNode;

#[cfg(test)]
mod tests;
