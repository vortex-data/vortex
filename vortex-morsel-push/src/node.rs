// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`ExecNode`] contract and the arena that drives it.

use std::collections::VecDeque;
use std::mem::size_of;
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

/// Index of an input edge within a push node.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InputPort(u32);

impl InputPort {
    /// Build a port from its zero-based edge index.
    pub fn new(index: usize) -> VortexResult<Self> {
        Ok(Self(u32::try_from(index).map_err(|_| {
            vortex_err!("push input port index exceeds u32")
        })?))
    }

    /// The zero-based edge index.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// The destination of values pushed across one child edge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Route {
    /// The parent node receiving the value.
    pub parent: NodeId,
    /// The parent's input port for this edge.
    pub port: InputPort,
}

/// Which value execution model a scan uses.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExecutionMode {
    /// Recursively pull values from the plan root.
    #[default]
    Pull,
    /// Activate sources and push values toward the plan root.
    Push,
}

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

/// Authoritative logical rows and the (possibly wider) physical evaluation domain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActivationRows {
    selected: Mask,
    materialized: Mask,
}

impl ActivationRows {
    /// Validate a logical selection and its physical materialization domain.
    pub fn try_new(selected: Mask, materialized: Mask) -> VortexResult<Self> {
        if selected.len() != materialized.len() {
            return Err(vortex_err!(
                "activation selection length {} differs from materialization length {}",
                selected.len(),
                materialized.len()
            ));
        }
        if !selected.is_subset_of(&materialized) {
            return Err(vortex_err!(
                "activation selects rows outside its materialization domain"
            ));
        }
        Ok(Self {
            selected,
            materialized,
        })
    }

    /// Materialize exactly the authoritative selected rows.
    pub fn selected(selected: Mask) -> Self {
        Self {
            materialized: selected.clone(),
            selected,
        }
    }

    /// The authoritative logical selection.
    pub fn logical(&self) -> &Mask {
        &self.selected
    }

    /// The physical rows supplied to the operator expression.
    pub fn materialized(&self) -> &Mask {
        &self.materialized
    }

    pub(crate) fn into_parts(self) -> (Mask, Mask) {
        (self.selected, self.materialized)
    }

    pub(crate) fn slice(&self, range: Range<usize>) -> Self {
        // `ActivationRows` guarantees selected ⊆ materialized, so equal cardinality proves the
        // masks are equal. Slice once in the common exact-materialization regime: `Mask::slice`
        // computes the sliced true count, and doing that twice repeats a row-proportional scan.
        if self.selected.true_count() == self.materialized.true_count() {
            return Self::selected(self.selected.slice(range));
        }
        Self {
            selected: self.selected.slice(range.clone()),
            materialized: self.materialized.slice(range),
        }
    }
}

/// A pushed value together with its root-coordinate coverage and authoritative row selection.
///
/// Array values are dense over `materialized`, which may be wider than the logical `selection`
/// used by a predicate cascade. Mask values stay in the coverage domain and refine `selection`.
#[derive(Clone)]
pub struct PushBatch {
    /// The root-coordinate input rows accounted for by this batch.
    pub coverage: Range<u64>,
    /// The authoritative selection over `coverage`.
    pub selection: Mask,
    /// The physical evaluation domain over `coverage`.
    pub materialized: Mask,
    /// A dense array or a coverage-domain mask, as described on [`PushBatch`].
    pub value: Value,
}

impl PushBatch {
    /// Build and validate a pushed batch.
    pub fn try_new(coverage: Range<u64>, selection: Mask, value: Value) -> VortexResult<Self> {
        Self::try_new_materialized(coverage, ActivationRows::selected(selection), value)
    }

    pub(crate) fn try_new_materialized(
        coverage: Range<u64>,
        rows: ActivationRows,
        value: Value,
    ) -> VortexResult<Self> {
        let coverage_len = usize::try_from(coverage.end.saturating_sub(coverage.start))
            .map_err(|_| vortex_err!("push batch coverage length exceeds usize"))?;
        if coverage.end < coverage.start {
            return Err(vortex_err!("push batch coverage is reversed: {coverage:?}"));
        }
        if rows.logical().len() != coverage_len {
            return Err(vortex_err!(
                "push selection length {} does not match coverage length {coverage_len}",
                rows.logical().len()
            ));
        }
        match &value {
            Value::Array(array) if array.len() != rows.materialized().true_count() => {
                return Err(vortex_err!(
                    "dense push array length {} does not match materialized true count {}",
                    array.len(),
                    rows.materialized().true_count()
                ));
            }
            Value::Mask(mask) => {
                if mask.len() != coverage_len {
                    return Err(vortex_err!(
                        "pushed mask length {} does not match coverage length {coverage_len}",
                        mask.len()
                    ));
                }
                if !mask.is_subset_of(rows.logical()) {
                    return Err(vortex_err!(
                        "pushed mask selects rows outside the authoritative selection"
                    ));
                }
            }
            Value::Array(_) => {}
        }
        let (selection, materialized) = rows.into_parts();
        Ok(Self {
            coverage,
            selection,
            materialized,
            value,
        })
    }

    /// Construct a batch whose invariants follow from an earlier validated batch and a
    /// preserving operation such as slicing or mask intersection.
    ///
    /// Unlike [`Self::try_new`], this does not rescan a mask to prove subset containment. Keep
    /// this crate-private: callers must establish that mask values refine `selection`.
    pub(crate) fn from_validated_parts(
        coverage: Range<u64>,
        selection: Mask,
        materialized: Mask,
        value: Value,
    ) -> Self {
        debug_assert!(coverage.start <= coverage.end);
        debug_assert_eq!(
            usize::try_from(coverage.end.saturating_sub(coverage.start)).ok(),
            Some(selection.len())
        );
        debug_assert_eq!(selection.len(), materialized.len());
        match &value {
            Value::Array(array) => debug_assert_eq!(array.len(), materialized.true_count()),
            Value::Mask(mask) => debug_assert_eq!(mask.len(), selection.len()),
        }
        Self {
            coverage,
            selection,
            materialized,
            value,
        }
    }

    /// Slice this batch to a root-coordinate subrange.
    ///
    /// Dense array offsets are computed from the rank of the subrange boundaries in
    /// `materialized`;
    /// mask values are sliced directly because they remain in the coverage domain.
    pub fn slice(self, coverage: Range<u64>) -> VortexResult<Self> {
        if coverage.start < self.coverage.start || coverage.end > self.coverage.end {
            return Err(vortex_err!(
                "push slice {coverage:?} is outside batch coverage {:?}",
                self.coverage
            ));
        }
        if coverage.end < coverage.start {
            return Err(vortex_err!("push slice coverage is reversed: {coverage:?}"));
        }
        if coverage == self.coverage {
            return Ok(self);
        }
        let start = usize::try_from(coverage.start - self.coverage.start)
            .map_err(|_| vortex_err!("push slice start exceeds usize"))?;
        let end = usize::try_from(coverage.end - self.coverage.start)
            .map_err(|_| vortex_err!("push slice end exceeds usize"))?;
        let dense_start = self.materialized.count_range(0, start);
        let selection = self.selection.slice(start..end);
        let materialized = self.materialized.slice(start..end);
        let dense_end = dense_start + materialized.true_count();
        let value = match self.value {
            Value::Array(array) => Value::Array(array.slice(dense_start..dense_end)?),
            Value::Mask(mask) => Value::Mask(mask.slice(start..end)),
        };
        // Bounds/rank slicing preserves every invariant already established by `try_new`.
        Ok(Self {
            coverage,
            selection,
            materialized,
            value,
        })
    }
}

/// A logical source group that must receive a span-exact activation decision.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ActivationTarget {
    /// One predicate conjunct, indexed in expression order.
    PredicateSlot(usize),
    /// The projection subtree.
    Projection,
}

/// A logical source group receiving an optional scheduling hint.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DemandTarget {
    /// One predicate conjunct, indexed in expression order.
    PredicateSlot(usize),
    /// The projection subtree.
    Projection,
}

/// Allocation-free result slot for one typed physical-stage call.
///
/// Data and terminal values have dedicated fields so fused stages hand a [`PushBatch`] directly
/// to the next stage. Ordered credit and scheduler sidebands share reusable node-indexed storage,
/// so one logical call can return its complete control prefix together with the data batch.
#[derive(Default)]
pub struct StageOutput {
    batch: Option<(PushBatch, bool)>,
    end: bool,
    sidebands: VecDeque<StageSideband>,
}

pub(crate) enum StageSideband {
    Consumed(InputPort),
    Gate {
        target: ActivationTarget,
        coverage: Range<u64>,
        rows: ActivationRows,
    },
    #[cfg(test)]
    Demand {
        target: DemandTarget,
        coverage: Range<u64>,
        selection: Mask,
    },
}

impl StageOutput {
    pub(crate) fn with_sidebands(sidebands: VecDeque<StageSideband>) -> Self {
        debug_assert!(sidebands.is_empty());
        Self {
            batch: None,
            end: false,
            sidebands,
        }
    }

    pub(crate) fn replace_sidebands(
        &mut self,
        sidebands: VecDeque<StageSideband>,
    ) -> VecDeque<StageSideband> {
        debug_assert!(self.is_empty());
        debug_assert!(sidebands.is_empty());
        std::mem::replace(&mut self.sidebands, sidebands)
    }

    /// Clear the slot before reusing it for another stage call.
    pub fn clear(&mut self) {
        self.batch = None;
        self.end = false;
        self.sidebands.clear();
    }

    /// Emit one batch directly to the next physical stage.
    pub fn set_batch(&mut self, batch: PushBatch, last_for_input: bool) {
        debug_assert!(self.batch.is_none() && !self.end);
        self.batch = Some((batch, last_for_input));
    }

    /// Emit a terminal marker from a source or boundary drain.
    pub fn set_end(&mut self) {
        debug_assert!(self.batch.is_none() && !self.end);
        self.end = true;
    }

    pub(crate) fn push_consumed(&mut self, port: InputPort) {
        self.sidebands.push_back(StageSideband::Consumed(port));
    }

    pub(crate) fn push_gate(
        &mut self,
        target: ActivationTarget,
        coverage: Range<u64>,
        rows: ActivationRows,
    ) {
        self.sidebands.push_back(StageSideband::Gate {
            target,
            coverage,
            rows,
        });
    }

    #[cfg(test)]
    pub(crate) fn push_demand(
        &mut self,
        target: DemandTarget,
        coverage: Range<u64>,
        selection: Mask,
    ) {
        self.sidebands.push_back(StageSideband::Demand {
            target,
            coverage,
            selection,
        });
    }

    pub(crate) fn take_batch(&mut self) -> Option<(PushBatch, bool)> {
        self.batch.take()
    }

    pub(crate) fn take_end(&mut self) -> bool {
        std::mem::take(&mut self.end)
    }

    pub(crate) fn take_sideband(&mut self) -> Option<StageSideband> {
        self.sidebands.pop_front()
    }

    pub(crate) fn take_inline_sideband(&mut self) -> Option<StageSideband> {
        matches!(
            self.sidebands.front(),
            Some(StageSideband::Consumed(_) | StageSideband::Gate { .. })
        )
        .then(|| self.sidebands.pop_front())
        .flatten()
    }

    pub(crate) fn has_sidebands(&self) -> bool {
        !self.sidebands.is_empty()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.batch.is_none() && !self.end && self.sidebands.is_empty()
    }

    pub(crate) fn into_sidebands(mut self) -> VecDeque<StageSideband> {
        self.sidebands.clear();
        self.sidebands
    }
}

/// The scheduler-visible state after a node consumes one push event.
pub enum NodeState {
    /// The node retained output that can be drained with [`ExecNode::push_resume`].
    Ready,
    /// The node is passive until another input event arrives.
    NeedInput,
    /// The node needs exact external dependencies before it can resume.
    Waiting(WaitSet),
    /// The fairness quantum ended after making progress.
    Yield(Progress),
    /// The node has produced everything it will produce for this morsel.
    Done,
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
    /// Benchmark-only physical-stage category.
    fn push_profile_kind(&self) -> PushProfileKind {
        PushProfileKind::Other
    }

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

    /// Activate a push source for one root-coordinate span.
    fn push_start(
        &mut self,
        _span: Range<u64>,
        _rows: ActivationRows,
        _cx: &mut PushCx<'_>,
        _out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        Err(vortex_err!("node is not a push source"))
    }

    /// Deliver one value batch directly to a typed boundary input.
    fn push_input(
        &mut self,
        _port: InputPort,
        _batch: PushBatch,
        _last_for_input: bool,
        _cx: &mut PushCx<'_>,
        _out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        Err(vortex_err!("node has no typed push input"))
    }

    /// Terminate a typed boundary input that emitted no batches.
    fn push_end(
        &mut self,
        port: InputPort,
        cx: &mut PushCx<'_>,
        _out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        let _ = (port, cx);
        Err(vortex_err!("node has no typed push input"))
    }

    /// Drain retained output from a source or boundary.
    fn push_resume(
        &mut self,
        cx: &mut PushCx<'_>,
        _out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        let _ = cx;
        Err(vortex_err!("node has no push continuation"))
    }

    /// Return one consumed batch credit directly to a producer.
    fn push_credit(
        &mut self,
        cx: &mut PushCx<'_>,
        _out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        let _ = cx;
        Err(vortex_err!("node does not accept push credit"))
    }

    /// Release anything this node holds for the finished morsel.
    fn retire(&mut self, cx: &mut RetireCx<'_>);

    /// This node's children, in edge order.
    fn children(&self) -> &[NodeId];
}

/// Benchmark-only physical-stage category.
#[derive(Clone, Copy)]
pub enum PushProfileKind {
    /// Flat segment source.
    Flat,
    /// Chunk concatenation boundary.
    Chunked,
    /// Struct field alignment boundary.
    Struct,
    /// Predicate conjunction boundary.
    Conjunct,
    /// Predicate/projection filter boundary.
    Filter,
    /// Test or extension node.
    Other,
}

/// One concrete execution node owned by a worker.
///
/// Payloads stay boxed so the arena retains the pointer-sized storage and per-node allocation
/// behavior of the former trait-object representation. Dispatch is closed over the operators the
/// plan builder can produce, allowing the compiler to specialize each forwarding arm. The
/// dynamic escape hatch preserves [`Arena::new`] compatibility for external arenas; plans built by
/// [`crate::build_plan`] never instantiate it.
pub(crate) enum Node {
    Flat(Box<crate::nodes::FlatExec>),
    Chunked(Box<crate::nodes::ChunkedExec>),
    Struct(Box<crate::nodes::StructExec>),
    Conjunct(Box<crate::nodes::ConjunctExec>),
    Filter(Box<crate::nodes::FilterExec>),
    Dynamic(Box<Box<dyn ExecNode>>),
}

const _: [(); size_of::<Box<dyn ExecNode>>()] = [(); size_of::<Node>()];

macro_rules! dispatch_node {
    ($node:expr, $inner:ident => $call:expr) => {
        match $node {
            Node::Flat($inner) => $call,
            Node::Chunked($inner) => $call,
            Node::Struct($inner) => $call,
            Node::Conjunct($inner) => $call,
            Node::Filter($inner) => $call,
            Node::Dynamic($inner) => $call,
        }
    };
}

impl Node {
    pub(crate) fn dynamic(node: Box<dyn ExecNode>) -> Self {
        Self::Dynamic(Box::new(node))
    }

    #[inline]
    pub(crate) fn try_accept_single_chunked_terminal(
        &mut self,
        port: InputPort,
        batch: &PushBatch,
        last_for_input: bool,
    ) -> VortexResult<bool> {
        match self {
            Self::Chunked(node) => node.try_accept_single_cut_terminal(port, batch, last_for_input),
            _ => Ok(false),
        }
    }
}

impl ExecNode for Node {
    #[inline]
    fn push_profile_kind(&self) -> PushProfileKind {
        dispatch_node!(self, node => node.push_profile_kind())
    }

    #[inline]
    fn reset(&mut self, range: Range<u64>) {
        dispatch_node!(self, node => node.reset(range))
    }

    #[inline]
    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
        dispatch_node!(self, node => node.next_plan(cx))
    }

    #[inline]
    fn execute(&mut self, cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
        dispatch_node!(self, node => node.execute(cx))
    }

    #[inline]
    fn push_start(
        &mut self,
        span: Range<u64>,
        rows: ActivationRows,
        cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        dispatch_node!(self, node => node.push_start(span, rows, cx, out))
    }

    #[inline]
    fn push_input(
        &mut self,
        port: InputPort,
        batch: PushBatch,
        last_for_input: bool,
        cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        dispatch_node!(self, node => node.push_input(port, batch, last_for_input, cx, out))
    }

    #[inline]
    fn push_end(
        &mut self,
        port: InputPort,
        cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        dispatch_node!(self, node => node.push_end(port, cx, out))
    }

    #[inline]
    fn push_resume(
        &mut self,
        cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        dispatch_node!(self, node => node.push_resume(cx, out))
    }

    #[inline]
    fn push_credit(
        &mut self,
        cx: &mut PushCx<'_>,
        out: &mut StageOutput,
    ) -> VortexResult<NodeState> {
        dispatch_node!(self, node => node.push_credit(cx, out))
    }

    #[inline]
    fn retire(&mut self, cx: &mut RetireCx<'_>) {
        dispatch_node!(self, node => node.retire(cx))
    }

    #[inline]
    fn children(&self) -> &[NodeId] {
        dispatch_node!(self, node => node.children())
    }
}

/// Context handed to the typed push-stage methods on [`ExecNode`].
///
/// Unlike [`ExecCx`], this context does not borrow the arena or carry row demand. Push sources
/// retain their authoritative activation selection, while typed input stages receive selection
/// with each [`PushBatch`]. This lets a morsel-owned physical runtime invoke stages directly.
pub struct PushCx<'a> {
    io: &'a IoPlane,
    cells: &'a SharedCells,
    session: &'a VortexSession,
    stats: &'a mut ScanStats,
}

impl<'a> PushCx<'a> {
    pub(crate) fn new(
        io: &'a IoPlane,
        cells: &'a SharedCells,
        session: &'a VortexSession,
        stats: &'a mut ScanStats,
    ) -> Self {
        Self {
            io,
            cells,
            session,
            stats,
        }
    }

    /// The session, for creating expression execution contexts.
    pub fn session(&self) -> &VortexSession {
        self.session
    }

    /// Clone ready bytes, first attempting a source-provided non-blocking inline read if unissued.
    pub fn ready(&mut self, ticket: IoTicket) -> VortexResult<Option<BufferHandle>> {
        self.io.ready(ticket, self.stats)
    }

    /// Take a decoded value from the shared cell for a unit, if already published.
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
}

/// An arena of nodes, owned by one worker and recycled across its morsels.
pub struct Arena {
    nodes: Vec<Option<Node>>,
    push_sidebands: Vec<VecDeque<StageSideband>>,
}

impl Arena {
    /// Build an arena from a list of dynamically dispatched nodes.
    ///
    /// This compatibility constructor retains vtable dispatch for the supplied nodes. Execution
    /// plans use a crate-private constructor whose five built-in node variants dispatch through a
    /// closed enum instead.
    pub fn new(nodes: Vec<Box<dyn ExecNode>>) -> Self {
        Self::new_compiled(nodes.into_iter().map(Node::dynamic).collect())
    }

    pub(crate) fn new_compiled(nodes: Vec<Node>) -> Self {
        let push_sidebands = (0..nodes.len()).map(|_| VecDeque::new()).collect();
        Self {
            nodes: nodes.into_iter().map(Some).collect(),
            push_sidebands,
        }
    }

    pub(crate) fn prepare_push_sidebands(&mut self, widths: impl IntoIterator<Item = usize>) {
        self.push_sidebands = widths
            .into_iter()
            .map(|width| VecDeque::with_capacity(width.saturating_mul(2).max(4)))
            .collect();
        assert_eq!(self.push_sidebands.len(), self.nodes.len());
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
    fn take(&mut self, id: NodeId) -> Node {
        self.nodes[id as usize].take().unwrap_or_else(|| {
            vortex_panic!("node {id} is already being driven: the exec graph is not a tree")
        })
    }

    fn put(&mut self, id: NodeId, node: Node) {
        self.nodes[id as usize] = Some(node);
    }

    /// Borrow one resident node for a non-recursive physical push-stage call.
    pub(crate) fn push_node_mut(&mut self, id: NodeId) -> &mut Node {
        self.nodes[id as usize].as_mut().unwrap_or_else(|| {
            vortex_panic!("node {id} is already being driven: the exec graph is not a tree")
        })
    }

    pub(crate) fn take_push_sidebands(&mut self) -> Vec<VecDeque<StageSideband>> {
        std::mem::take(&mut self.push_sidebands)
    }

    pub(crate) fn restore_push_sidebands(&mut self, sidebands: Vec<VecDeque<StageSideband>>) {
        assert_eq!(sidebands.len(), self.nodes.len());
        assert!(self.push_sidebands.is_empty());
        self.push_sidebands = sidebands;
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::IntoArray;
    use vortex_array::arrays::Primitive;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_buffer::BitBuffer;

    use super::ActivationRows;
    use super::Mask;
    use super::PushBatch;
    use super::Value;
    use super::VortexResult;

    #[test]
    fn push_batch_slice_uses_selection_rank_for_dense_array() -> VortexResult<()> {
        let selection = Mask::from_iter([true, false, true, false, true]);
        let array = PrimitiveArray::from_iter([10_i32, 20, 30]).into_array();
        let batch = PushBatch::try_new(10..15, selection, Value::Array(array))?;

        let sliced = batch.slice(12..15)?;

        assert_eq!(sliced.coverage, 12..15);
        assert_eq!(sliced.selection, Mask::from_iter([true, false, true]));
        let array = sliced.value.into_array()?;
        let array = array.as_::<Primitive>();
        assert_eq!(array.as_slice::<i32>(), &[20, 30]);
        Ok(())
    }

    #[test]
    fn push_batch_slice_preserves_misaligned_mask_refinement() -> VortexResult<()> {
        let selection = Mask::from_buffer(
            BitBuffer::from_iter([false, true, false, true, true, false, true, false, true])
                .slice(1..8),
        );
        let refined = Mask::from_buffer(
            BitBuffer::from_iter([false, true, false, false, true, false, false, false, true])
                .slice(1..8),
        );
        let batch = PushBatch::try_new(20..27, selection, Value::Mask(refined))?;

        let sliced = batch.slice(22..26)?;

        assert_eq!(sliced.selection, Mask::from_iter([true, true, false, true]));
        assert_eq!(
            sliced.value.into_mask()?,
            Mask::from_iter([false, true, false, false])
        );
        Ok(())
    }

    #[test]
    fn push_batch_rejects_mask_outside_authoritative_selection() {
        let selection = Mask::from_iter([true, false, true]);
        let refined = Mask::from_iter([true, true, false]);

        assert!(PushBatch::try_new(0..3, selection, Value::Mask(refined)).is_err());
    }

    #[test]
    fn activation_rows_reject_mismatch_and_selection_outside_materialization() {
        assert!(ActivationRows::try_new(Mask::new_true(2), Mask::new_true(3)).is_err());
        assert!(
            ActivationRows::try_new(
                Mask::from_iter([true, false, true]),
                Mask::from_iter([true, false, false]),
            )
            .is_err()
        );
    }

    #[test]
    fn activation_rows_slice_reuses_identical_random_mask() -> VortexResult<()> {
        let mask = Mask::from_iter(
            (0_usize..257).map(|index| index.wrapping_mul(17).wrapping_add(11).is_multiple_of(5)),
        );
        let rows = ActivationRows::try_new(mask.clone(), mask)?;

        let sliced = rows.slice(19..231);
        let expected = rows.logical().slice(19..231);
        assert_eq!(sliced.logical(), &expected);
        assert_eq!(sliced.materialized(), &expected);
        let shared = match (sliced.logical(), sliced.materialized()) {
            (Mask::Values(selected), Mask::Values(materialized)) => {
                Arc::ptr_eq(selected, materialized)
            }
            _ => false,
        };
        assert!(shared);
        Ok(())
    }

    #[test]
    fn activation_rows_slice_preserves_wider_dense_materialization() -> VortexResult<()> {
        let selected = Mask::from_iter([true, false, true, false, false, true, false, true]);
        let rows = ActivationRows::try_new(selected, Mask::new_true(8))?;

        let sliced = rows.slice(1..7);
        assert_eq!(
            sliced.logical(),
            &Mask::from_iter([false, true, false, false, true, false])
        );
        assert_eq!(sliced.materialized(), &Mask::new_true(6));
        Ok(())
    }

    #[test]
    fn activation_rows_slice_handles_misaligned_bitmap_views() -> VortexResult<()> {
        let buffer = BitBuffer::from_iter(
            (0_usize..173).map(|index| index.is_multiple_of(3) || index.is_multiple_of(11)),
        );
        let mask = Mask::from_buffer(buffer.slice(5..168));
        let rows = ActivationRows::selected(mask.clone());

        let sliced = rows.slice(7..149);
        let expected = mask.slice(7..149);
        assert_eq!(sliced.logical(), &expected);
        assert_eq!(sliced.materialized(), &expected);
        Ok(())
    }

    #[test]
    fn push_batch_slice_ranks_dense_array_by_materialization() -> VortexResult<()> {
        let selected = Mask::from_iter([true, false, false, true, false, false]);
        let materialized = Mask::from_iter([true, true, false, true, true, false]);
        let rows = ActivationRows::try_new(selected, materialized)?;
        let array = PrimitiveArray::from_iter([10_i32, 20, 30, 40]).into_array();
        let batch = PushBatch::try_new_materialized(10..16, rows, Value::Array(array))?;

        let sliced = batch.slice(11..15)?;

        assert_eq!(
            sliced.selection,
            Mask::from_iter([false, false, true, false])
        );
        assert_eq!(
            sliced.materialized,
            Mask::from_iter([true, false, true, true])
        );
        let array = sliced.value.into_array()?;
        assert_eq!(array.as_::<Primitive>().as_slice::<i32>(), &[20, 30, 40]);
        Ok(())
    }
}
