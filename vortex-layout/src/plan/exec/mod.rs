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
//! # Interfaces
//!
//! The module has one interface for each side of a graph:
//!
//! - A plan operator runs as an [`ExecNode`], built by
//!   [`PlanVTable::exec`](crate::plan::PlanVTable::exec). A node sees the graph only through its
//!   [`StepCx`]: it takes [`Event`]s from it, and spawns children, requests segments, and emits
//!   [`Piece`]s through it. [`ExecNode`] says what a node must do.
//! - An owner drives an [`ExecGraph`]: it calls [`ExecGraph::compute`], answers the
//!   [`IoRequest`]s that returns, and collects the root's pieces.
//!
//! How a graph stores, wires, and schedules its nodes belongs to neither interface, and nor do
//! the nodes of this crate's own operators.
//!
//! # Planner model
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
mod list_pack;
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
use smallvec::SmallVec;
use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexError;
use vortex_error::VortexResult;
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

/// A run of a plan's rows, as one array.
///
/// `rows` is a range of the producing plan's row domain. `array` holds the rows of that range
/// that the node's selection kept, in row order, so it is shorter than `rows` where rows were
/// unselected and empty where none were selected.
///
/// A plan may instead be specified to produce every row of the range, taking the selection as a
/// hint and leaving the values of unselected rows unspecified. The child of a
/// [`Filter`](crate::plan::Filter) must be such a plan, and
/// [`SegmentScan`](crate::plan::SegmentScan) is one.
#[derive(Clone)]
pub struct Piece {
    /// The rows this piece covers, including unselected rows.
    pub rows: Range<u64>,
    /// The values of the rows in `rows` that the piece holds.
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

/// An input of a node.
///
/// A node names the port when it spawns the child that feeds it, and the child's pieces and close
/// arrive tagged with it.
pub type Port = usize;

/// Something that arrived for a node since its previous [`ExecNode::compute`].
#[derive(Debug)]
pub enum Event {
    /// A piece produced by the child on this port, in the child's row domain.
    Piece(Port, Piece),
    /// The child on this port will produce no more pieces. It arrives once for each child.
    Closed(Port),
    /// The bytes of a segment this node requested. It arrives once for each request.
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
/// A node is built by [`PlanVTable::exec`](crate::plan::PlanVTable::exec) for a range of its
/// plan's rows and a selection over them, and produces those rows as [`Piece`]s.
///
/// # Lifecycle
///
/// A node has a single entry point. The graph calls [`compute`](Self::compute) once when the node
/// is spawned, with no events; again whenever an [`Event`] has arrived for it; and again after it
/// returned [`NodeState::Yield`]. A node is not called again once it has returned
/// [`NodeState::Done`] or an error.
///
/// Everything a node does goes through its [`StepCx`] and takes effect once `compute` returns,
/// in this order: its children are spawned and run their first `compute`, its requests are
/// published, its pieces are pushed to its parent, and, if it is done, its output closes.
///
/// # Output
///
/// By the time it returns [`NodeState::Done`], a node must have emitted pieces whose row ranges
/// cover the rows it was built for exactly once. The pieces may be of any size and emitted in any
/// order. [`Piece`] says what each one holds.
///
/// A node's children are not cancelled when it is done, so it returns `Done` only once every
/// child it spawned has closed. It returns [`NodeState::Wait`] only while a child is open or a
/// request is outstanding, since nothing else runs it again.
///
/// The graph checks none of this.
///
/// # Errors
///
/// An error from `compute` fails the whole graph.
pub trait ExecNode: Send {
    /// Takes what arrived, does work, and reports what happens to the node next.
    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState>;
}

/// What happens to a node after an [`ExecNode::compute`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeState {
    /// Run again once an event arrives: a child's piece or close, or a requested delivery.
    Wait,
    /// Run again without waiting for an event, after other work: CPU work is left.
    Yield,
    /// Finished: the node's output closes and it never runs again.
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
    pub(crate) fn decoded(&self) -> &DecodeCache {
        &self.decoded
    }
}

/// Decoded segments shared by the graphs that read one row range, so a segment several of them
/// read is fetched and decoded once.
///
/// A segment decodes the same way wherever it appears, so entries are keyed by segment id alone.
#[derive(Clone)]
pub struct DecodeCache(Option<Arc<Mutex<FxHashMap<SegmentId, ArrayRef>>>>);

impl Default for DecodeCache {
    fn default() -> Self {
        Self(Some(Arc::default()))
    }
}

impl DecodeCache {
    /// Reads and decodes ordinary segments for each evaluation, matching V1's flat reader.
    pub(crate) fn disabled() -> Self {
        Self(None)
    }

    /// The whole decoded array of `id`, if a graph sharing this cache decoded it.
    pub(crate) fn get(&self, id: SegmentId) -> Option<ArrayRef> {
        self.0.as_ref()?.lock().get(&id).cloned()
    }

    /// Shares the whole decoded array of `id` with the graphs sharing this cache.
    pub(crate) fn insert(&self, id: SegmentId, array: ArrayRef) {
        if let Some(cache) = &self.0 {
            cache.lock().insert(id, array);
        }
    }
}

/// A node's view of its graph during one [`ExecNode::compute`]: what arrived for the node, and
/// what the node does about it.
///
/// Nothing recorded here happens until `compute` returns. [`ExecNode`] gives the order.
pub struct StepCx<'a> {
    next_io_id: &'a mut u64,
    events: SmallVec<[Event; 2]>,
    spawned: SmallVec<[(Port, PlanRef, Range<u64>, Mask); 1]>,
    requests: SmallVec<[IoRequest; 1]>,
    emitted: SmallVec<[Piece; 1]>,
}

impl<'a> StepCx<'a> {
    fn new(next_io_id: &'a mut u64, events: SmallVec<[Event; 2]>) -> Self {
        Self {
            next_io_id,
            events,
            spawned: SmallVec::new(),
            requests: SmallVec::new(),
            emitted: SmallVec::new(),
        }
    }

    /// Takes the events that arrived since the node's previous `compute`, in arrival order.
    ///
    /// Events still here when `compute` returns are discarded.
    pub fn events(&mut self) -> impl ExactSizeIterator<Item = Event> + DoubleEndedIterator + use<> {
        mem::take(&mut self.events).into_iter()
    }

    /// Spawns `plan` as a child feeding `port`, over `rows` of the child's row domain restricted
    /// to `mask`, which is as long as `rows`.
    ///
    /// The child's pieces arrive as [`Event::Piece`] and its close as [`Event::Closed`].
    pub fn spawn(&mut self, port: Port, plan: PlanRef, rows: Range<u64>, mask: Mask) {
        self.spawned.push((port, plan, rows, mask));
    }

    /// Publishes a read of `segment_id`. Its bytes arrive as [`Event::Delivered`] under the
    /// returned id.
    pub fn request(&mut self, segment_id: SegmentId) -> IoRequestId {
        let id = IoRequestId(*self.next_io_id);
        *self.next_io_id += 1;
        self.requests.push(IoRequest { id, segment_id });
        id
    }

    /// Pushes a piece to the node's parent, or from the root to the graph's owner.
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

struct Node {
    exec: Box<dyn ExecNode>,
    parent: Option<(NodeId, Port)>,
    inbox: SmallVec<[Event; 2]>,
    status: Status,
    output_offset: u64,
}

/// A running plan: a flat arena of nodes wired child-to-parent.
pub struct ExecGraph {
    ctx: ExecContext,
    nodes: Vec<Node>,
    /// Ready nodes, run last-in first-out so a woken parent runs before its child's next call.
    ready: Vec<NodeId>,
    io_routes: SmallVec<[Option<NodeId>; 2]>,
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
            ctx: ExecContext::new(session, row_offset, decoded),
            nodes: Vec::new(),
            ready: Vec::new(),
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

    /// Runs ready nodes until a piece reaches the root, requests should be flushed, a node
    /// yields, or nothing is ready.
    ///
    /// # Errors
    ///
    /// Returns the error of a node that failed. The graph cannot be run further.
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
            .get_mut(usize::try_from(id.0)?)
            .and_then(Option::take)
            .ok_or_else(|| vortex_err!("Unknown IO request {id:?}"))?;
        self.nodes[node].inbox.push(Event::Delivered(id, result));
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
        self.spawn_rebased(parent, plan, rows, mask, 0)
    }

    fn spawn_rebased(
        &mut self,
        parent: Option<(NodeId, Port)>,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: Mask,
        output_offset: u64,
    ) -> VortexResult<()> {
        // A split usually touches one chunk. Rebase that child's output at the edge instead
        // of allocating and scheduling a concatenation node solely to pass its pieces through.
        if let Some(concat) = plan.as_opt::<Concat>()
            && rows.start < rows.end
            && !mask.all_false()
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
                return self.spawn_rebased(
                    parent,
                    &concat.child_required(index)?,
                    rows.start - start..rows.end - start,
                    mask,
                    output_offset + start,
                );
            }
        }
        let id = self.nodes.len();
        self.nodes.push(Node {
            exec: plan.exec(rows, mask, &self.ctx)?,
            parent,
            inbox: SmallVec::new(),
            status: Status::Queued,
            output_offset,
        });
        self.run(id)?;
        Ok(())
    }

    /// Runs one node's compute with the events that arrived for it and applies the effects.
    /// Returns whether control goes back to the owner: when the node yields, and after a node
    /// handles a delivery, since decoding and evaluating newly read data is the costly work
    /// between reads that other owners' work should interleave with.
    fn run(&mut self, node: NodeId) -> VortexResult<bool> {
        let events = mem::take(&mut self.nodes[node].inbox);
        let delivered = events
            .iter()
            .any(|event| matches!(event, Event::Delivered(..)));
        let mut next_io_id = self.next_io_id;
        let mut cx = StepCx::new(&mut next_io_id, events);
        let state = self.nodes[node].exec.compute(&mut cx)?;
        let effects = Effects::from(cx);
        self.next_io_id = next_io_id;

        // Settle the node before waking its parent, so a re-queued node runs after the parent.
        self.nodes[node].status = match state {
            NodeState::Done => Status::Done,
            NodeState::Wait | NodeState::Yield => Status::Idle,
        };
        if state == NodeState::Yield || !self.nodes[node].inbox.is_empty() {
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
            let index = usize::try_from(request.id.0)?;
            if index >= self.io_routes.len() {
                self.io_routes.resize(index + 1, None);
            }
            self.io_routes[index] = Some(node);
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
        let offset = self.nodes[from].output_offset;
        let piece = piece.map(|piece| Piece {
            rows: piece.rows.start + offset..piece.rows.end + offset,
            array: piece.array,
        });
        match self.nodes[from].parent {
            None => match piece {
                Some(piece) => self.outputs.push_back(piece),
                None => self.root_closed = true,
            },
            Some((parent, port)) => {
                self.nodes[parent].inbox.push(match piece {
                    Some(piece) => Event::Piece(port, piece),
                    None => Event::Closed(port),
                });
                self.enqueue(parent);
            }
        }
    }

    fn enqueue(&mut self, node: NodeId) {
        if self.nodes[node].status == Status::Idle {
            self.nodes[node].status = Status::Queued;
            self.ready.push(node);
        }
    }
}

/// The owned effects of a [`StepCx`], detached from its borrows.
struct Effects {
    spawned: SmallVec<[(Port, PlanRef, Range<u64>, Mask); 1]>,
    requests: SmallVec<[IoRequest; 1]>,
    emitted: SmallVec<[Piece; 1]>,
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
pub(crate) use eval::fuse_dictionary_predicate;
pub(crate) use filter::FilterNode;
pub(crate) use filter::keep_selected;
pub(crate) use list_pack::ListPackNode;
pub(crate) use pack::PackNode;
pub(crate) use pack::assemble;
pub(crate) use piece::Selection;
pub(crate) use piece::empty_piece;
pub(crate) use piece::join;
pub(crate) use row_idx::RowIdxNode;
pub(crate) use row_idx::row_indices;
pub(crate) use segment_scan::SegmentScanNode;
pub(crate) use segment_scan::decode_segment;
pub(crate) use segment_scan::slice_rows;
pub(crate) use share::ShareNode;
pub(crate) use take::TakeNode;
pub(crate) use zoned::ZonePruneNode;
pub(crate) use zoned::expand_zones;
pub(crate) use zoned::prune_zones;
pub(crate) use zoned::pruned_zones;

#[cfg(test)]
mod tests;
