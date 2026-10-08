// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Push-based execution of physical plans.
//!
//! A [`PlanRef`] is turned into an [`ExecGraph`] for one row range and selection. The graph is a
//! flat arena of [`ExecNode`]s wired child-to-parent through [`Input`] ports. A node's output is
//! an ordered sequence of arrays: each node emits its rows in row order, and its parent reads
//! them from the port the child feeds. The root's arrays are handed to the owner one per
//! [`ExecGraph::compute`], in row order. [`execute`] drives a graph over a [`SegmentSource`]
//! and yields them as a stream.
//!
//! IO may complete in any order. A node whose children are backed by independent reads, such as
//! a concatenation of chunks, holds the early ones in their ports and emits only the ordered
//! prefix, so nothing above it sees the disorder. Nodes never talk to the IO service: they
//! publish requests through [`StepCx::request`], the graph returns them from
//! [`ExecGraph::compute`], and the owner delivers results through [`ExecGraph::set_io_result`].
//!
//! # Interfaces
//!
//! The module has one interface for each side of a graph:
//!
//! - A plan operator runs as an [`ExecNode`], built by
//!   [`PlanVTable::exec`](crate::plan::PlanVTable::exec). A node sees the graph only through its
//!   [`StepCx`]: it reads its ports, and spawns children, requests segments, and emits arrays
//!   through it. [`ExecNode`] says what a node must do, and [`Ready`] says when it runs.
//! - An owner drives an [`ExecGraph`]: it calls [`ExecGraph::compute`], answers the
//!   [`IoRequest`]s that returns, and collects the root's arrays.
//!
//! How a graph stores, wires, and schedules its nodes belongs to neither interface, and nor do
//! the nodes of this crate's own operators.
//!
//! # Pipelines
//!
//! A chain of nodes that are [`Ready::Any`] over one port is a pipeline: an array emitted at
//! its bottom can be pushed through every node in the chain in turn. Nodes with several ports
//! merge pipelines, and nodes that wait for a port to close, [`Ready::Closed`] and
//! [`Ready::AllClosed`], are barriers between them, as a join build is in a query engine.
//!
//! [`SegmentSource`]: crate::segments::SegmentSource

mod concat;
mod eval;
mod filter;
mod list_pack;
mod pack;
mod query;
mod row_idx;
mod segment_scan;
mod selection;
mod stream;
pub mod synthetic;
mod take;

use std::collections::VecDeque;
use std::mem;
use std::ops::Range;
use std::sync::Arc;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::plan::Concat;
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

/// An input of a node.
///
/// A node names the port when it spawns the child that feeds it, and reads the child's output
/// from it through [`StepCx::input`].
pub type Port = usize;

/// One input port of a node: the arrays its child has produced and not yet been taken, in row
/// order, and whether the child has closed.
///
/// A port that was never spawned reads as closed with nothing available, so a node can treat an
/// optional child it did not spawn like one that finished at once.
#[derive(Default)]
pub struct Input {
    spawned: bool,
    closed: bool,
    pieces: VecDeque<ArrayRef>,
    available: usize,
}

impl Input {
    /// Rows the child has produced that the node has not taken yet.
    pub fn available(&self) -> usize {
        self.available
    }

    /// Whether the child will produce nothing more. True for a port that was never spawned.
    pub fn closed(&self) -> bool {
        !self.spawned || self.closed
    }

    /// Whether the child has closed and every row it produced has been taken.
    pub fn finished(&self) -> bool {
        self.closed() && self.pieces.is_empty()
    }

    /// Takes every available array, in row order.
    pub fn take_all(&mut self) -> Vec<ArrayRef> {
        self.available = 0;
        self.pieces.drain(..).collect()
    }

    /// Takes the next available array, if any.
    pub fn pop(&mut self) -> Option<ArrayRef> {
        let array = self.pieces.pop_front()?;
        self.available -= array.len();
        Some(array)
    }

    /// Takes the next `n` rows, in row order, as the arrays that hold them. The array that
    /// spans the boundary is sliced, and its remainder stays available.
    ///
    /// `n` must not exceed [`available`](Self::available).
    pub fn take(&mut self, n: usize) -> VortexResult<Vec<ArrayRef>> {
        if n > self.available {
            vortex_bail!(
                "Taking {n} rows from a port with {} available",
                self.available
            );
        }
        let mut taken = Vec::new();
        let mut remaining = n;
        while remaining > 0 {
            let Some(front) = self.pieces.pop_front() else {
                vortex_bail!("Port ran out of arrays with {remaining} rows still to take");
            };
            if front.len() <= remaining {
                remaining -= front.len();
                taken.push(front);
            } else {
                taken.push(front.slice(0..remaining)?);
                self.pieces.push_front(front.slice(remaining..front.len())?);
                remaining = 0;
            }
        }
        self.available -= n;
        Ok(taken)
    }

    fn push(&mut self, array: ArrayRef) {
        self.available += array.len();
        self.pieces.push_back(array);
    }
}

/// When a node's [`compute`](ExecNode::compute) runs.
///
/// A node is considered whenever something changes for it: a child produced rows or closed, or
/// a requested segment was delivered. The rule then decides whether the node runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ready {
    /// On every change. For nodes that stream their input through.
    Any,
    /// Only once each of these ports has closed. For nodes that need one input whole before
    /// they can use the others, as a dictionary needs its values before its codes.
    Closed(&'static [Port]),
    /// Only once every spawned port has closed. For nodes that need all their input at once.
    AllClosed,
}

/// One running plan operator inside an [`ExecGraph`].
///
/// A node is built by [`PlanVTable::exec`](crate::plan::PlanVTable::exec) for a range of its
/// plan's rows and a selection over them, and produces the selected rows as arrays, in row
/// order.
///
/// # Lifecycle
///
/// The graph calls [`start`](Self::start) once, right after the node is built. It then calls
/// [`compute`](Self::compute) whenever something changed for the node and the node's
/// [`ready`](Self::ready) rule holds, and again after `compute` returned [`NodeState::Yield`].
/// A node is not called again once it has returned [`NodeState::Done`] or an error.
///
/// Everything a node does goes through its [`StepCx`] and takes effect once the call returns,
/// in this order: its children are spawned and started, its requests are published, its arrays
/// are pushed to its parent's port, and, if it is done, its port closes.
///
/// # Output
///
/// A node emits the selected rows of the range it was built for, in row order, as any number of
/// non-empty arrays. It returns [`NodeState::Done`] once it has emitted them all. Its children
/// are not cancelled, so it returns `Done` only once every child it spawned has closed, and it
/// returns [`NodeState::Wait`] only while a child is open or a request is outstanding, since
/// nothing else runs it again.
///
/// The graph checks none of this.
///
/// # Errors
///
/// An error from `start` or `compute` fails the whole graph.
pub trait ExecNode: Send {
    /// When [`compute`](Self::compute) runs. May change as the node progresses.
    fn ready(&self) -> Ready {
        Ready::Any
    }

    /// Spawns children, publishes requests, and emits whatever needs no input.
    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState>;

    /// Takes what its ports hold, does work, and reports what happens to the node next.
    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState>;
}

/// What happens to a node after a [`start`](ExecNode::start) or [`compute`](ExecNode::compute).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeState {
    /// Run again once something changes: a child produces rows or closes, or a delivery lands.
    Wait,
    /// Run again without waiting for a change, after other work: CPU work is left.
    Yield,
    /// Finished: the node's port closes and it never runs again.
    Done,
}

/// What a graph gives every node it builds.
#[derive(Clone)]
pub struct ExecContext {
    session: VortexSession,
    row_offset: u64,
    decoded: DecodeCache,
}

impl ExecContext {
    pub(crate) fn new(session: VortexSession, row_offset: u64, decoded: DecodeCache) -> Self {
        Self {
            session,
            row_offset,
            decoded,
        }
    }

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
pub struct DecodeCache {
    decoded: Arc<Mutex<FxHashMap<SegmentId, ArrayRef>>>,
}

impl DecodeCache {
    /// The whole decoded array of `id`, if a graph sharing this cache decoded it.
    pub fn get(&self, id: SegmentId) -> Option<ArrayRef> {
        self.decoded.lock().get(&id).cloned()
    }

    /// Shares the whole decoded array of `id` with the graphs sharing this cache.
    pub fn insert(&self, id: SegmentId, array: ArrayRef) {
        self.decoded.lock().insert(id, array);
    }
}

/// A child a node spawned: its port, plan, rows, and selection.
type Spawn = (Port, PlanRef, Range<u64>, Mask);

/// A node's view of its graph during one [`start`](ExecNode::start) or
/// [`compute`](ExecNode::compute): its ports, a delivered segment, and what the node does about
/// them.
///
/// Nothing recorded here happens until the call returns. [`ExecNode`] gives the order.
pub struct StepCx<'a> {
    inputs: &'a mut SmallVec<[Input; 2]>,
    delivered: Option<BufferHandle>,
    next_io_id: &'a mut u64,
    spawned: SmallVec<[Spawn; 1]>,
    requests: SmallVec<[IoRequest; 1]>,
    emitted: SmallVec<[ArrayRef; 1]>,
}

impl<'a> StepCx<'a> {
    fn new(
        inputs: &'a mut SmallVec<[Input; 2]>,
        delivered: Option<BufferHandle>,
        next_io_id: &'a mut u64,
    ) -> Self {
        Self {
            inputs,
            delivered,
            next_io_id,
            spawned: SmallVec::new(),
            requests: SmallVec::new(),
            emitted: SmallVec::new(),
        }
    }

    /// The port fed by the child spawned on `port`.
    pub fn input(&mut self, port: Port) -> &mut Input {
        if port >= self.inputs.len() {
            self.inputs.resize_with(port + 1, Input::default);
        }
        &mut self.inputs[port]
    }

    /// Every port, in port order, including ports never spawned.
    pub fn inputs(&mut self) -> &mut [Input] {
        self.inputs
    }

    /// Whether every spawned port has closed and been drained.
    pub fn all_finished(&self) -> bool {
        self.inputs.iter().all(Input::finished)
    }

    /// The bytes of the segment this node requested, if they arrived since it last ran.
    pub fn take_delivery(&mut self) -> Option<BufferHandle> {
        self.delivered.take()
    }

    /// Spawns `plan` as a child feeding `port`, over `rows` of the child's row domain restricted
    /// to `mask`, which is as long as `rows`.
    pub fn spawn(&mut self, port: Port, plan: PlanRef, rows: Range<u64>, mask: Mask) {
        self.spawned.push((port, plan, rows, mask));
    }

    /// Publishes a read of `segment_id`. Its bytes arrive through
    /// [`take_delivery`](Self::take_delivery). A node has at most one read outstanding.
    pub fn request(&mut self, segment_id: SegmentId) -> IoRequestId {
        let id = IoRequestId(*self.next_io_id);
        *self.next_io_id += 1;
        self.requests.push(IoRequest { id, segment_id });
        id
    }

    /// Pushes the node's next rows to its parent, or from the root to the graph's owner. Empty
    /// arrays are dropped.
    pub fn emit(&mut self, array: ArrayRef) {
        if !array.is_empty() {
            self.emitted.push(array);
        }
    }
}

/// What an [`ExecGraph`] needs next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecState {
    /// A node is ready, or an array or requests are waiting to be returned.
    NeedsCompute,
    /// Every remaining node waits on a delivered request.
    Waiting,
    /// The root has closed and every array has been returned.
    Done,
}

/// The result of one [`ExecGraph::compute`] call.
#[derive(Debug)]
pub enum ExecOutput {
    /// Control returns to the owner. Check [`ExecGraph::state`] for what is needed next.
    Yield,
    /// Requests published since the last flush, each returned exactly once.
    NeedsIO(IoBatch),
    /// The root's next rows.
    Piece(ArrayRef),
}

type NodeId = usize;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    /// Not in the ready queue.
    Idle,
    /// In the ready queue.
    Queued,
    /// Closed; never run again.
    Done,
}

struct Node {
    exec: Box<dyn ExecNode>,
    parent: Option<(NodeId, Port)>,
    inputs: SmallVec<[Input; 2]>,
    delivered: Option<BufferHandle>,
    status: Status,
}

/// A running plan: a flat arena of nodes wired child-to-parent.
pub struct ExecGraph {
    ctx: ExecContext,
    nodes: Vec<Node>,
    /// Ready nodes, run last-in first-out so a woken parent runs before its child's next call.
    ready: Vec<NodeId>,
    io_routes: SmallVec<[Option<NodeId>; 2]>,
    new_io: IoBatch,
    outputs: VecDeque<ArrayRef>,
    root_closed: bool,
    next_io_id: u64,
}

impl ExecGraph {
    /// Builds the graph for `plan` over `rows`, restricted to `mask`, and starts each node.
    /// `row_offset` is the global row index of the plan's first row, used by row-index plans.
    /// Segments found in `decoded` are neither read nor decoded again, and segments the graph
    /// decodes are added to it.
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
            ctx: ExecContext::new(session, row_offset, decoded),
            // A split of a few columns runs about this many nodes; a split of one column
            // inside one chunk runs two.
            nodes: Vec::with_capacity(8),
            ready: Vec::with_capacity(4),
            io_routes: SmallVec::new(),
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
        self.io_routes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                let id = IoRequestId(index as u64);
                (node.is_some() && !self.new_io.iter().any(|request| request.id == id))
                    .then_some(id)
            })
    }

    /// Runs ready nodes until an array reaches the root, requests should be flushed, a node
    /// yields, or nothing is ready.
    ///
    /// # Errors
    ///
    /// Returns the error of a node that failed. The graph cannot be run further.
    pub fn compute(&mut self) -> VortexResult<ExecOutput> {
        loop {
            if let Some(array) = self.outputs.pop_front() {
                return Ok(ExecOutput::Piece(array));
            }
            let Some(node) = self.ready.pop() else {
                return Ok(self.flush_or_yield());
            };
            let yielded = self.run(node, false)?;

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
            .get_mut(usize::try_from(id.0)?)
            .and_then(Option::take)
            .ok_or_else(|| vortex_err!("Unknown IO request {id:?}"))?;
        self.nodes[node].delivered = Some(result);
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

    /// Builds the node for `plan` under `parent` and starts it.
    ///
    /// A concatenation whose requested rows fall inside one chunk is skipped: the chunk is
    /// wired to the parent directly, since its rows are the concatenation's rows in order.
    fn spawn(
        &mut self,
        parent: Option<(NodeId, Port)>,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: Mask,
    ) -> VortexResult<()> {
        if let Some(concat) = plan.as_opt::<Concat>()
            && rows.start < rows.end
        {
            let offsets = concat.row_offsets();
            let index = offsets
                .partition_point(|&offset| offset <= rows.start)
                .saturating_sub(1);
            let end = offsets
                .get(index + 1)
                .copied()
                .unwrap_or_else(|| concat.row_count());
            if let Some(&start) = offsets.get(index)
                && rows.end <= end
            {
                return self.spawn(
                    parent,
                    &concat.child_required(index)?,
                    rows.start - start..rows.end - start,
                    mask,
                );
            }
        }
        if let Some((parent, port)) = parent {
            let inputs = &mut self.nodes[parent].inputs;
            if port >= inputs.len() {
                inputs.resize_with(port + 1, Input::default);
            }
            inputs[port].spawned = true;
        }
        let id = self.nodes.len();
        self.nodes.push(Node {
            exec: plan.exec(rows, mask, &self.ctx)?,
            parent,
            inputs: SmallVec::new(),
            delivered: None,
            status: Status::Queued,
        });
        self.run(id, true)?;
        Ok(())
    }

    /// Runs one node, its `start` or its `compute`, and applies the effects. Returns whether
    /// control goes back to the owner: when the node yields, and after a node handles a
    /// delivery, since decoding and evaluating newly read data is the costly work between reads
    /// that other owners' work should interleave with.
    fn run(&mut self, id: NodeId, start: bool) -> VortexResult<bool> {
        let mut next_io_id = self.next_io_id;
        let node = &mut self.nodes[id];
        let delivered = node.delivered.take();
        let had_delivery = delivered.is_some();
        let mut cx = StepCx::new(&mut node.inputs, delivered, &mut next_io_id);
        let state = if start {
            node.exec.start(&mut cx)?
        } else {
            node.exec.compute(&mut cx)?
        };
        let effects = Effects::from(cx);
        self.next_io_id = next_io_id;

        // Settle the node before waking its parent, so a re-queued node runs after the parent.
        self.nodes[id].status = match state {
            NodeState::Done => Status::Done,
            NodeState::Wait | NodeState::Yield => Status::Idle,
        };
        if state == NodeState::Yield {
            self.enqueue(id);
        }
        self.apply(id, effects, state == NodeState::Done)?;
        Ok(state == NodeState::Yield || had_delivery)
    }

    fn apply(&mut self, id: NodeId, effects: Effects, done: bool) -> VortexResult<()> {
        for (port, plan, rows, mask) in effects.spawned {
            self.spawn(Some((id, port)), &plan, rows, mask)?;
        }
        for request in effects.requests {
            let index = usize::try_from(request.id.0)?;
            if index >= self.io_routes.len() {
                self.io_routes.resize(index + 1, None);
            }
            self.io_routes[index] = Some(id);
            self.new_io.push(request);
        }
        for array in effects.emitted {
            self.deliver(id, Some(array));
        }
        if done {
            self.deliver(id, None);
        }
        Ok(())
    }

    /// Passes an array, or the close when `array` is `None`, from `from` to its parent's port,
    /// and wakes the parent if its readiness rule holds.
    fn deliver(&mut self, from: NodeId, array: Option<ArrayRef>) {
        match self.nodes[from].parent {
            None => match array {
                Some(array) => self.outputs.push_back(array),
                None => self.root_closed = true,
            },
            Some((parent, port)) => {
                let input = &mut self.nodes[parent].inputs[port];
                match array {
                    Some(array) => input.push(array),
                    None => input.closed = true,
                }
                if self.is_ready(parent) {
                    self.enqueue(parent);
                }
            }
        }
    }

    fn is_ready(&self, id: NodeId) -> bool {
        let node = &self.nodes[id];
        match node.exec.ready() {
            Ready::Any => true,
            Ready::Closed(ports) => ports
                .iter()
                .all(|&port| node.inputs.get(port).is_none_or(Input::closed)),
            Ready::AllClosed => node.inputs.iter().all(Input::closed),
        }
    }

    fn enqueue(&mut self, id: NodeId) {
        if self.nodes[id].status == Status::Idle {
            self.nodes[id].status = Status::Queued;
            self.ready.push(id);
        }
    }
}

/// The owned effects of a [`StepCx`], detached from its borrows.
struct Effects {
    spawned: SmallVec<[Spawn; 1]>,
    requests: SmallVec<[IoRequest; 1]>,
    emitted: SmallVec<[ArrayRef; 1]>,
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
pub(crate) use list_pack::ListPackNode;
pub(crate) use pack::PackNode;
pub(crate) use query::QueryNode;
pub(crate) use row_idx::RowIdxNode;
pub(crate) use segment_scan::SegmentScanNode;
pub(crate) use selection::Selection;
pub use stream::ExecStream;
pub use stream::execute;
pub(crate) use take::TakeNode;

#[cfg(test)]
mod scheduling_tests;
#[cfg(test)]
mod tests;
