// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`ExecNode`] contract and the arena that drives it.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::cells::SharedCells;
use crate::io::IoBatch;
use crate::io::IoKey;
use crate::io::IoPlane;
use crate::io::IoPriority;
use crate::io::IoTicket;
use crate::stats::ScanStats;

/// Index of a node within an [`Arena`].
pub type NodeId = u32;

/// A value produced by a node for its parent.
#[derive(Clone)]
pub enum Value {
    /// Dense rows: length equals the true count of the demand mask the node was executed under.
    Array(ArrayRef),
    /// A refinement of the demand mask the node was executed under; same length as that mask.
    Mask(Mask),
}

impl Value {
    /// Unwrap an array value, or fail if this is a mask.
    pub fn into_array(self) -> VortexResult<ArrayRef> {
        match self {
            Value::Array(array) => Ok(array),
            Value::Mask(_) => Err(vortex_err!("expected an array value, got a mask")),
        }
    }

    /// Unwrap a mask value, or fail if this is an array.
    pub fn into_mask(self) -> VortexResult<Mask> {
        match self {
            Value::Mask(mask) => Ok(mask),
            Value::Array(_) => Err(vortex_err!("expected a mask value, got an array")),
        }
    }
}

/// A value plus the dense range of *input* rows it accounts for.
pub struct ValueBatch {
    /// The root-coordinate row range this batch accounts for.
    pub coverage: Range<u64>,
    /// The value itself.
    pub value: Value,
}

/// What a node's planning stream produced.
pub enum PlanItem {
    /// A batch of named IO uses, already registered with the IO plane.
    Io(IoBatch),
    /// The node yielded before refining further; call `next_plan` again to resume.
    Plan,
}

/// The result of polling a node's planning stream.
pub enum PlanPoll {
    /// An item was produced.
    Item(PlanItem),
    /// Planning is suspended on the given waits; no worker thread is parked.
    Blocked(WaitSet),
    /// Planning has finished. This forfeits any further refinement of this node's IO.
    Complete,
}

/// The result of polling a node's execution.
pub enum ExecPoll {
    /// A value covering a dense input row range.
    Value(ValueBatch),
    /// Execution is suspended on the given waits; no worker thread is parked.
    Blocked(WaitSet),
    /// The node made progress but has not produced a value yet.
    Yield(Progress),
    /// The node has produced everything it will produce.
    Done,
}

/// Result of advancing a child from inside its parent node.
pub enum ChildPoll<T> {
    /// The child produced the requested value.
    Value(T),
    /// The child is suspended on exact external dependencies.
    Blocked(WaitSet),
    /// The child has no more values.
    Done,
}

/// A coarse progress marker returned with [`ExecPoll::Yield`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Progress {
    /// Rows of input consumed since the last poll.
    pub rows: u64,
}

/// Something a node can park on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Wait {
    /// An IO ticket the node's own planning stream emitted.
    Io(IoTicket),
}

/// A set of [`Wait`]s. Small by construction — a node parks on the handful of cells it named.
#[derive(Clone, Debug, Default)]
pub struct WaitSet(Vec<Wait>);

impl WaitSet {
    /// An empty wait set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Park on one more thing.
    pub fn push(&mut self, wait: Wait) {
        self.0.push(wait);
    }

    /// The waits in this set.
    pub fn waits(&self) -> &[Wait] {
        &self.0
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl FromIterator<Wait> for WaitSet {
    fn from_iter<T: IntoIterator<Item = Wait>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// A stateful, per-morsel execution node.
///
/// Nodes are arena-allocated once per worker and reset when that worker's arena is recycled to
/// another morsel. `&mut self` state survives suspension and always resumes on its owning worker.
pub trait ExecNode: Send {
    /// Reset this node for a new morsel covering `range` (in this node's local coordinates).
    fn reset(&mut self, range: Range<u64>);

    /// Advance this node's planning stream.
    ///
    /// Planning only names IO; it never reads. A node that has more planning to do than its
    /// budget allows returns [`PlanItem::Plan`] and resumes from its own cursor.
    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll>;

    /// Advance this node's execution, producing values under the demand in `cx`.
    ///
    /// This method may use [`ExecCx::ready`] to attempt an inline read that the source guarantees
    /// will not wait on storage. It must not perform blocking IO, poll background futures,
    /// synchronously transfer device data, or wait for an external resource. A missing dependency
    /// must return [`ExecPoll::Blocked`] so the scheduler can resume the continuation later.
    fn execute(&mut self, cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll>;

    /// Release anything this node holds for the finished morsel.
    fn retire(&mut self, cx: &mut RetireCx<'_>);

    /// This node's children, in edge order.
    fn children(&self) -> &[NodeId];
}

/// An arena of nodes, owned by one worker and recycled across its morsels.
pub struct Arena {
    nodes: Vec<Option<Box<dyn ExecNode>>>,
}

impl Arena {
    /// Build an arena from a list of nodes.
    pub fn new(nodes: Vec<Box<dyn ExecNode>>) -> Self {
        Self {
            nodes: nodes.into_iter().map(Some).collect(),
        }
    }

    /// The number of nodes in the arena.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the arena is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Take a node out of the arena so its children can be driven through the remaining slots.
    ///
    /// The node must be put back with [`Arena::put`]. The take/put pair is what lets a node hold
    /// `&mut self` while recursively driving its children: the tree shape guarantees a node is
    /// never reachable from its own subtree, so a taken slot is never observed as empty.
    fn take(&mut self, id: NodeId) -> Box<dyn ExecNode> {
        self.nodes[id as usize].take().unwrap_or_else(|| {
            vortex_panic!("node {id} is already being driven: the exec graph is not a tree")
        })
    }

    fn put(&mut self, id: NodeId, node: Box<dyn ExecNode>) {
        self.nodes[id as usize] = Some(node);
    }

    /// Reset the subtree rooted at `id` for a morsel covering `range`.
    pub fn reset_subtree(&mut self, id: NodeId, range: Range<u64>) {
        let mut node = self.take(id);
        node.reset(range);
        self.put(id, node);
    }
}

/// Context handed to [`ExecNode::next_plan`].
pub struct PlanCx<'a> {
    arena: &'a mut Arena,
    io: &'a IoPlane,
    cells: &'a SharedCells,
    stats: &'a mut ScanStats,
    /// Remaining IO uses this planning quantum may emit before the node should yield.
    budget: u32,
    priority: IoPriority,
}

impl<'a> PlanCx<'a> {
    /// The remaining planning budget, in IO uses.
    pub fn budget(&self) -> u32 {
        self.budget
    }

    /// Whether the planning quantum is exhausted.
    pub fn out_of_budget(&self) -> bool {
        self.budget == 0
    }

    /// Whether a shared cell already holds the decoded value for a unit.
    ///
    /// A hit lets the node skip issuing the read entirely: the caller's own lease (counted into
    /// the cell before the scan started) keeps the value alive until this morsel retires.
    pub fn decoded_available(&self, key: IoKey) -> bool {
        self.cells.decoded(key).is_some()
    }

    /// Register a batch of IO uses, spending budget and returning tickets.
    pub fn register(&mut self, batch: IoBatch) -> VortexResult<Vec<IoTicket>> {
        self.budget = self
            .budget
            .saturating_sub(u32::try_from(batch.uses().len()).unwrap_or(u32::MAX));
        self.stats.io_uses += batch.uses().len() as u64;
        self.io.register(batch, self.priority, self.stats)
    }

    /// Drive one child with an explicit scheduler priority for reads it registers.
    pub(crate) fn plan_child_with_priority(
        &mut self,
        id: NodeId,
        range: Range<u64>,
        fresh: bool,
        priority: IoPriority,
    ) -> VortexResult<bool> {
        let previous = std::mem::replace(&mut self.priority, priority);
        let result = self.plan_child(id, range, fresh);
        self.priority = previous;
        result
    }

    /// Drive a child's planning stream to completion, cutting it to `range` first.
    ///
    /// Returns `true` when the child completed, `false` when the shared budget ran out and the
    /// caller should yield and resume at this child.
    pub fn plan_child(&mut self, id: NodeId, range: Range<u64>, fresh: bool) -> VortexResult<bool> {
        let mut node = self.arena.take(id);
        let result = (|| {
            if fresh {
                node.reset(range);
            }
            loop {
                match node.next_plan(self)? {
                    PlanPoll::Item(PlanItem::Io(_)) => continue,
                    PlanPoll::Item(PlanItem::Plan) => return Ok(false),
                    PlanPoll::Blocked(_) => {
                        // P1 has no gated planning: nothing can park a planning stream.
                        return Ok(false);
                    }
                    PlanPoll::Complete => return Ok(true),
                }
            }
        })();
        self.arena.put(id, node);
        result
    }
}

/// Context handed to [`ExecNode::execute`].
pub struct ExecCx<'a> {
    arena: &'a mut Arena,
    io: &'a IoPlane,
    cells: &'a SharedCells,
    session: &'a VortexSession,
    stats: &'a mut ScanStats,
    demand: Mask,
}

impl<'a> ExecCx<'a> {
    /// The demand mask this node is executing under.
    ///
    /// Its length equals the number of rows in the node's local range; the node must produce
    /// exactly `demand().true_count()` rows.
    pub fn demand(&self) -> &Mask {
        &self.demand
    }

    /// The session, for creating expression execution contexts.
    pub fn session(&self) -> &VortexSession {
        self.session
    }

    /// Clone ready bytes, first attempting a source-provided non-blocking inline read if unissued.
    pub fn ready(&mut self, ticket: IoTicket) -> VortexResult<Option<BufferHandle>> {
        self.io.ready(ticket, self.stats)
    }

    /// Take a decoded value from the shared cell for a unit, if a morsel already published one.
    pub fn shared_decoded(&mut self, key: IoKey) -> Option<ArrayRef> {
        let hit = self.cells.decoded(key);
        if hit.is_some() {
            self.stats.decode_reuses += 1;
        }
        hit
    }

    /// Publish a decoded value into the shared cell for a unit.
    pub fn publish_decoded(&self, key: IoKey, array: &ArrayRef) {
        self.cells.publish(key, array);
    }

    /// Mutable access to the run's counters.
    pub fn stats(&mut self) -> &mut ScanStats {
        self.stats
    }

    /// Drive a child to a value under `demand`.
    ///
    /// The child is polled until it yields a value, blocks on exact tickets, or reports `Done`.
    pub fn child_value(&mut self, id: NodeId, demand: Mask) -> VortexResult<ChildPoll<ValueBatch>> {
        let mut node = self.arena.take(id);
        let saved = std::mem::replace(&mut self.demand, demand);
        let result = (|| {
            loop {
                match node.execute(self)? {
                    ExecPoll::Value(batch) => return Ok(ChildPoll::Value(batch)),
                    ExecPoll::Yield(_) => continue,
                    ExecPoll::Blocked(waits) => return Ok(ChildPoll::Blocked(waits)),
                    ExecPoll::Done => return Ok(ChildPoll::Done),
                }
            }
        })();
        self.demand = saved;
        self.arena.put(id, node);
        result
    }

    /// Drive a child to an array value, failing if it produced nothing.
    pub fn child_array(&mut self, id: NodeId, demand: Mask) -> VortexResult<ChildPoll<ArrayRef>> {
        match self.child_value(id, demand)? {
            ChildPoll::Value(batch) => Ok(ChildPoll::Value(batch.value.into_array()?)),
            ChildPoll::Blocked(waits) => Ok(ChildPoll::Blocked(waits)),
            ChildPoll::Done => Ok(ChildPoll::Done),
        }
    }

    /// Drive a child to a mask value.
    pub fn child_mask(&mut self, id: NodeId, demand: Mask) -> VortexResult<ChildPoll<Mask>> {
        match self.child_value(id, demand)? {
            ChildPoll::Value(batch) => Ok(ChildPoll::Value(batch.value.into_mask()?)),
            ChildPoll::Blocked(waits) => Ok(ChildPoll::Blocked(waits)),
            ChildPoll::Done => Ok(ChildPoll::Done),
        }
    }
}

/// Context handed to [`ExecNode::retire`].
pub struct RetireCx<'a> {
    arena: &'a mut Arena,
    cells: &'a SharedCells,
    stats: &'a mut ScanStats,
}

impl<'a> RetireCx<'a> {
    /// Retire a child subtree.
    pub fn retire_child(&mut self, id: NodeId) {
        let mut node = self.arena.take(id);
        node.retire(self);
        self.arena.put(id, node);
    }

    /// Mutable access to the run's counters.
    pub fn stats(&mut self) -> &mut ScanStats {
        self.stats
    }

    /// Release this morsel's lease on a unit, dropping the shared cell at the last release.
    pub fn release_use(&mut self, key: IoKey) {
        self.cells.release(key);
    }
}

/// The number of IO uses one planning quantum may emit before a node should yield.
pub const PLAN_BUDGET: u32 = 64;

/// Reset an arena for one morsel before its planning continuation is queued.
pub(crate) fn begin_morsel(arena: &mut Arena, root: NodeId, range: Range<u64>) {
    arena.reset_subtree(root, range);
}

/// Advance one planning quantum for a morsel.
pub(crate) fn poll_plan_morsel(
    arena: &mut Arena,
    root: NodeId,
    io: &IoPlane,
    cells: &SharedCells,
    stats: &mut ScanStats,
) -> VortexResult<PlanPoll> {
    let mut cx = PlanCx {
        arena,
        io,
        cells,
        stats,
        budget: PLAN_BUDGET,
        priority: IoPriority::Required,
    };
    let mut node = cx.arena.take(root);
    let poll = node.next_plan(&mut cx);
    cx.arena.put(root, node);
    poll
}

/// Advance one execution quantum for a morsel.
pub(crate) fn poll_execute_morsel(
    arena: &mut Arena,
    root: NodeId,
    range: &Range<u64>,
    io: &IoPlane,
    cells: &SharedCells,
    session: &VortexSession,
    stats: &mut ScanStats,
) -> VortexResult<ExecPoll> {
    let rows = usize::try_from(range.end - range.start)
        .map_err(|_| vortex_err!("morsel row count exceeds usize"))?;
    let mut cx = ExecCx {
        arena,
        io,
        cells,
        session,
        stats,
        demand: Mask::new_true(rows),
    };
    let mut node = cx.arena.take(root);
    let poll = node.execute(&mut cx);
    cx.arena.put(root, node);
    poll
}

/// Retire a completed morsel and release its decoded-cell leases.
pub(crate) fn retire_morsel(
    arena: &mut Arena,
    root: NodeId,
    cells: &SharedCells,
    stats: &mut ScanStats,
) {
    let mut cx = RetireCx {
        arena,
        cells,
        stats,
    };
    cx.retire_child(root);
}
