// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The scan2 tree: per-layout nodes with value, proof, and mask
//! capabilities (plan 017).
//!
//! Like the v1 scan, a file's layout tree expands into one node per
//! layout through session-registered rules, and the typed traits here are
//! author-facing: the engine works through the blanket-implemented
//! [`DynScanNode`] / [`DynLayoutScanRule`] adapters. Three things are
//! new:
//!
//! - expansion is *negotiation*: rules see the scoped scan request before
//!   expression pushdown plans reads and evidence (see [`super::request`]);
//! - expression pushdown returns another scan node whose root value is
//!   the pushed expression, so reads and evidence are planned from
//!   `root()` of that node instead of reparsing expressions; and
//! - executable value read plans use one scoped primitive: selection
//!   controls output cardinality, and demand controls which selected rows
//!   must contain meaningful values.

use std::any::TypeId;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::future::BoxFuture;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::expr::stats::Precision;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::LayoutEncodingId;
use crate::LayoutRef;
use crate::scan::v2::evidence::EvidenceFragment;
use crate::scan::v2::request::EvidenceRequest;
use crate::scan::v2::request::NodeRequest;
use crate::scan::v2::session::ScanV2SessionExt;
use crate::segments::SegmentPlanCtx;
use crate::segments::SegmentRequests;
use crate::segments::SegmentSource;

/// Per-file/query IO context for scan2 reads.
pub struct FileReader {
    segments: Arc<dyn SegmentSource>,
    session: VortexSession,
}

impl FileReader {
    /// Create a reader context from a segment source and session.
    pub fn new(segments: Arc<dyn SegmentSource>, session: VortexSession) -> Self {
        Self { segments, session }
    }

    /// Segment source for layout data.
    pub fn segments(&self) -> &Arc<dyn SegmentSource> {
        &self.segments
    }

    /// Session used to decode arrays and execute expressions.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }
}

/// A scan2 node's per-file/query global state, type-erased.
pub type ScanState = dyn std::any::Any + Send + Sync;

/// A reference to a scan2 node's per-file/query global state.
pub type ScanStateRef = Arc<ScanState>;

/// A reference-counted, type-erased scan2 node.
pub type ScanNodeRef = Arc<dyn DynScanNode>;

/// A reference-counted, type-erased scan2 rule.
pub type ScanRuleRef = Arc<dyn DynLayoutScanRule>;

/// A reference-counted, type-erased evidence plan.
pub type EvidencePlanRef = Arc<dyn DynEvidencePlan>;

/// A reference-counted, type-erased read plan.
pub type ReadPlanRef = Arc<dyn DynReadPlan>;

/// A reference-counted, type-erased split plan.
pub type SplitPlanRef = Arc<dyn DynSplitPlan>;

/// A reference-counted, type-erased ungrouped aggregate plan.
pub type AggregatePlanRef = Arc<dyn DynAggregatePlan>;

/// A reference-counted, type-erased metadata statistics plan.
pub type StatsPlanRef = Arc<dyn DynStatsPlan>;

/// Per-file/query cache of scan-node global state while a file's planned
/// reads are initialized.
pub type ScanStateCache = FxHashMap<usize, ScanStateRef>;

/// Key for evidence plans whose per-query state can be shared by several
/// planned predicates in the same file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EvidenceStateKey {
    type_id: TypeId,
    key: usize,
}

impl EvidenceStateKey {
    pub fn new<T: 'static>(key: usize) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            key,
        }
    }
}

/// Context for expression pushdown.
pub struct PushCtx {
    session: VortexSession,
}

impl PushCtx {
    pub fn new(session: VortexSession) -> Self {
        Self { session }
    }

    pub fn session(&self) -> &VortexSession {
        &self.session
    }
}

/// Context for turning pushed expressions into executable read/evidence plans.
pub struct PlanCtx {
    session: VortexSession,
}

impl PlanCtx {
    pub fn new(session: VortexSession) -> Self {
        Self { session }
    }

    pub fn session(&self) -> &VortexSession {
        &self.session
    }
}

/// Context for initializing node global state. All read plans for one file
/// share this context, so the same node instance gets one state object even
/// when several pushed expressions reference it.
pub struct StateCtx<'a> {
    session: &'a VortexSession,
    node_cache: &'a mut ScanStateCache,
}

impl<'a> StateCtx<'a> {
    pub fn new(session: &'a VortexSession, node_cache: &'a mut ScanStateCache) -> Self {
        Self {
            session,
            node_cache,
        }
    }

    pub fn session(&self) -> &VortexSession {
        self.session
    }

    pub fn init_node(&mut self, node: &ScanNodeRef) -> VortexResult<ScanStateRef> {
        let key = scan_node_key(node);
        if let Some(hit) = self.node_cache.get(&key) {
            return Ok(Arc::clone(hit));
        }
        let state = node.init_state(self)?;
        self.node_cache.insert(key, Arc::clone(&state));
        Ok(state)
    }
}

fn scan_node_key(node: &ScanNodeRef) -> usize {
    Arc::as_ptr(node) as *const () as usize
}

/// One operation's row scope in a scan2 node's input row domain.
#[derive(Clone, Copy, Debug)]
pub struct RowScope<'a> {
    /// Rows still semantically live in the input domain.
    pub selection: &'a Mask,
    /// Rows whose value/result is needed by this operation.
    pub demand: &'a Mask,
}

impl<'a> RowScope<'a> {
    pub fn selected(selection: &'a Mask) -> Self {
        Self {
            selection,
            demand: selection,
        }
    }

    pub fn try_new(selection: &'a Mask, demand: &'a Mask) -> VortexResult<Self> {
        if selection.len() != demand.len() {
            vortex_bail!(
                "row scope selection/demand length mismatch: {} vs {}",
                selection.len(),
                demand.len()
            );
        }
        if !demand.clone().bitand_not(selection).all_false() {
            vortex_bail!("row scope demand must be a subset of selection");
        }
        Ok(Self { selection, demand })
    }

    pub fn demands_all_selected(self) -> bool {
        std::ptr::eq(self.selection, self.demand)
            || self.demand.true_count() == self.selection.true_count()
    }
}

/// One aggregate plan's mixed-coverage answer.
///
/// The covered rows are the requested range minus `residual`; `partial`
/// accounts for exactly those rows, each once. An all-null span counts
/// as covered with no contribution. The caller reads and accumulates the
/// residual spans itself, so one statistics read can answer several
/// functions while each keeps its own unanswerable leftovers.
#[derive(Debug)]
pub struct AggregateAnswer {
    /// Combined partial state for the covered rows, in the function's
    /// partial dtype — ready for
    /// [`combine_partials`](vortex_array::aggregate_fn::DynAccumulator::combine_partials).
    /// `None` when no covered row contributed (empty coverage, or only
    /// provably all-null spans).
    pub partial: Option<Scalar>,
    /// Rows of the requested range the statistics could not answer, as
    /// disjoint ascending spans in this node's row coordinates.
    pub residual: Vec<Range<u64>>,
}

/// One layout encoding's scan2 behaviour, registered per
/// [`LayoutEncodingId`]. Not object safe; the engine resolves rules as
/// [`ScanRuleRef`]s through the blanket [`DynLayoutScanRule`] adapter.
pub trait LayoutScanRule: 'static + fmt::Debug + Send + Sync {
    /// The scan-tree node this rule expands to.
    type Node: ScanNode;

    /// The layout encoding this rule reads.
    fn id(&self) -> LayoutEncodingId;

    /// Expand one layout node into a scan node. Row-preserving children
    /// receive the same request object; children in another row domain
    /// receive [`NodeRequest::empty`].
    fn expand(
        &self,
        layout: &LayoutRef,
        req: &mut NodeRequest,
        cx: &ExpandCtx,
    ) -> VortexResult<Self::Node>;
}

/// A node in the expanded scan2 tree. Nodes are shared across queries;
/// all per-file/query caching lives in the node's `State`.
pub trait ScanNode: 'static + Send + Sync {
    /// Per-file/query global state: decoded arrays, decoded index state,
    /// child node states, and other frontier-released caches.
    type State: Send + Sync + 'static;

    /// Create this node's per-file/query state.
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State>;

    /// Try to push `expr` into this node's row domain. The returned node's
    /// root value is exactly `expr` in the input row domain.
    ///
    /// The default accepts `root()` as this node and otherwise builds a
    /// generic scalar-apply node over this node's root value. Layouts
    /// specialize when they can route or rewrite the expression, e.g.
    /// struct field access or list-offset functions.
    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        _cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanNodeRef>>
    where
        Self: Sized,
    {
        if is_root(expr) {
            Ok(Some(self))
        } else {
            Ok(Some(Arc::new(ApplyScanNode::new(self, expr.clone()))))
        }
    }

    /// Plan value reads for this node's root value.
    fn plan_read(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Option<ReadPlanRef>>;

    /// Plan natural row splits for this node's root value.
    ///
    /// The default converts this node's cheap split hints into an executable plan. Nodes can
    /// override this when split discovery needs request-specific state, I/O, or cost estimates.
    fn plan_splits(self: Arc<Self>, _cx: &mut PlanCtx) -> VortexResult<Option<SplitPlanRef>>
    where
        Self: Sized,
    {
        Ok(self
            .split_hints()
            .map(|hints| Arc::new(HintSplitPlan::new(hints.to_vec())) as SplitPlanRef))
    }

    /// Plan predicate evidence for this node's root boolean value.
    ///
    /// Planning performs no IO and returns a direct executable handle. The
    /// handle may precompute expression rewrites or accepted predicate
    /// fragments, but runtime state remains in [`Self::State`].
    fn plan_evidence(self: Arc<Self>, _cx: &mut PlanCtx) -> VortexResult<Vec<EvidencePlanRef>>
    where
        Self: Sized,
    {
        Ok(Vec::new())
    }

    /// Plan ungrouped aggregates over this node's root value.
    ///
    /// The returned plan answers all `funcs` together over a runtime row
    /// range, producing one [`AggregateAnswer`] per function. `None` means
    /// this node cannot answer these aggregates from layout metadata and
    /// the caller should read rows normally.
    fn plan_aggregate_partial(
        self: Arc<Self>,
        _funcs: &[AggregateFnRef],
        _cx: &mut PlanCtx,
    ) -> VortexResult<Option<AggregatePlanRef>>
    where
        Self: Sized,
    {
        Ok(None)
    }

    /// Plan metadata statistics for this node's root value.
    ///
    /// The returned plan answers the requested aggregate functions positionally over runtime row
    /// ranges using metadata only. `None` means this node cannot answer these functions from
    /// metadata.
    fn plan_stats(
        self: Arc<Self>,
        _funcs: &[AggregateFnRef],
        _cx: &mut PlanCtx,
    ) -> VortexResult<Option<StatsPlanRef>>
    where
        Self: Sized,
    {
        Ok(None)
    }

    /// Preferred morsel boundaries (chunk edges), for alignment hints.
    fn split_hints(&self) -> Option<&[u64]> {
        None
    }

    /// Rows below `frontier` will not be read again this query: drop
    /// per-file/query state retained solely for them. Releasing must be
    /// an optimization only; the default keeps everything.
    fn release(&self, _frontier: u64, _state: &Self::State) -> VortexResult<()> {
        Ok(())
    }

    /// Compact reader-chain description for plan display, e.g.
    /// `"zoned:chunked(8)"`.
    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

/// Read every row in `range` through a read plan.
pub(crate) fn read_dense<'a>(
    read: &'a dyn DynReadPlan,
    range: Range<u64>,
    io: &'a FileReader,
    state: &'a ScanState,
) -> BoxFuture<'a, VortexResult<ArrayRef>> {
    Box::pin(async move {
        let len = range_len(&range)?;
        let selection = Mask::new_true(len);
        let mut local = io.session().create_execution_ctx();
        read.read_scoped(range, RowScope::selected(&selection), io, state, &mut local)
            .await
    })
}

fn range_len(range: &Range<u64>) -> VortexResult<usize> {
    let len = range
        .end
        .checked_sub(range.start)
        .ok_or_else(|| vortex_err!("read range end is before start: {range:?}"))?;
    usize::try_from(len).map_err(|_| vortex_err!("read range exceeds usize"))
}

/// Object-safe view of a [`ScanNode`]. Blanket-implemented; never by
/// hand.
pub trait DynScanNode: Send + Sync {
    /// Create this node's per-file/query state, type-erased.
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef>;

    /// Try to push an expression into this node's row domain.
    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanNodeRef>>;

    /// Plan value reads for this node's root value.
    fn plan_read(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Option<ReadPlanRef>>;

    /// Plan natural row splits for this node's root value.
    fn plan_splits(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Option<SplitPlanRef>>;

    /// Plan predicate evidence for this node's root boolean value.
    fn plan_evidence(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Vec<EvidencePlanRef>>;

    /// Plan ungrouped aggregates for this node's root value.
    fn plan_aggregate_partial(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        cx: &mut PlanCtx,
    ) -> VortexResult<Option<AggregatePlanRef>>;

    /// Plan metadata statistics for this node's root value.
    fn plan_stats(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        cx: &mut PlanCtx,
    ) -> VortexResult<Option<StatsPlanRef>>;

    /// Preferred morsel boundaries (see [`ScanNode::split_hints`]).
    fn split_hints(&self) -> Option<&[u64]>;

    /// Release state behind the frontier (see [`ScanNode::release`]).
    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()>;

    /// Reader-chain description for plan display.
    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl<T: ScanNode> DynScanNode for T {
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(ScanNode::init_state(self, cx)?))
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanNodeRef>> {
        ScanNode::try_push_expr(self, expr, cx)
    }

    fn plan_read(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Option<ReadPlanRef>> {
        ScanNode::plan_read(self, cx)
    }

    fn plan_splits(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Option<SplitPlanRef>> {
        ScanNode::plan_splits(self, cx)
    }

    fn plan_evidence(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Vec<EvidencePlanRef>> {
        ScanNode::plan_evidence(self, cx)
    }

    fn plan_aggregate_partial(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        cx: &mut PlanCtx,
    ) -> VortexResult<Option<AggregatePlanRef>> {
        ScanNode::plan_aggregate_partial(self, funcs, cx)
    }

    fn plan_stats(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        cx: &mut PlanCtx,
    ) -> VortexResult<Option<StatsPlanRef>> {
        ScanNode::plan_stats(self, funcs, cx)
    }

    fn split_hints(&self) -> Option<&[u64]> {
        ScanNode::split_hints(self)
    }

    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()> {
        ScanNode::release(self, frontier, downcast_state::<T>(state)?)
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ScanNode::fmt_chain(self, f)
    }
}

/// Executable value read plan for one pushed expression.
pub trait ReadPlan: 'static + Send + Sync {
    /// The per-query state this read plan executes against.
    type State: Send + Sync + 'static;

    /// Create this read plan's per-file/query global state.
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State>;

    /// Read the live rows of `range`, with [`RowScope`] defining output
    /// cardinality (`selection`) and meaningful-value demand (`demand`).
    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a FileReader,
        state: &'a Self::State,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>>;

    /// Return scheduler-visible segment requests needed for this read, when known exactly.
    fn segment_requests(
        &self,
        _range: Range<u64>,
        _rows: RowScope<'_>,
        _state: &Self::State,
        _cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        Ok(SegmentRequests::unknown())
    }

    /// Release state behind the completed-row frontier.
    fn release(&self, _frontier: u64, _state: &Self::State) -> VortexResult<()> {
        Ok(())
    }

    /// Compact description for plan display.
    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "read")
    }
}

/// Object-safe view of a [`ReadPlan`].
pub trait DynReadPlan: Send + Sync {
    /// Create this read plan's per-file/query state, type-erased.
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef>;

    /// Read rows in a selection/demand scope.
    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a FileReader,
        state: &'a ScanState,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>>;

    /// Return scheduler-visible segment requests needed for this read, when known exactly.
    fn segment_requests(
        &self,
        range: Range<u64>,
        rows: RowScope<'_>,
        state: &ScanState,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests>;

    /// Release state behind the completed-row frontier.
    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()>;

    /// Compact description for plan display.
    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl<T: ReadPlan> DynReadPlan for T {
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(ReadPlan::init_state(self, cx)?))
    }

    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a FileReader,
        state: &'a ScanState,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        let state = match downcast_erased_state::<T::State>(state) {
            Ok(state) => state,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        ReadPlan::read_scoped(self, range, rows, io, state, local)
    }

    fn segment_requests(
        &self,
        range: Range<u64>,
        rows: RowScope<'_>,
        state: &ScanState,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        let state = downcast_erased_state::<T::State>(state)?;
        ReadPlan::segment_requests(self, range, rows, state, cx)
    }

    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()> {
        let state = downcast_erased_state::<T::State>(state)?;
        ReadPlan::release(self, frontier, state)
    }

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ReadPlan::fmt_plan(self, f)
    }
}

/// Executable split plan for one pushed expression.
pub trait SplitPlan: 'static + Send + Sync {
    /// The per-query state this split plan executes against.
    type State: Send + Sync + 'static;

    /// Create this split plan's per-query state.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<Self::State>;

    /// Return natural row ranges inside `range`.
    fn splits<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a Self::State,
    ) -> BoxFuture<'a, VortexResult<Vec<Range<u64>>>>;

    /// Compact description for plan display.
    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "splits")
    }
}

/// Object-safe view of a [`SplitPlan`].
pub trait DynSplitPlan: Send + Sync {
    /// Create this split plan's per-query state, type-erased.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef>;

    /// Execute the planned split query.
    fn splits<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Vec<Range<u64>>>>;

    /// Compact description for plan display.
    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl<T: SplitPlan> DynSplitPlan for T {
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(SplitPlan::init_state(self, ctx)?))
    }

    fn splits<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Vec<Range<u64>>>> {
        let state = match downcast_erased_state::<T::State>(state) {
            Ok(state) => state,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        SplitPlan::splits(self, range, io, state)
    }

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        SplitPlan::fmt_plan(self, f)
    }
}

struct HintSplitPlan {
    hints: Vec<u64>,
}

impl HintSplitPlan {
    fn new(hints: Vec<u64>) -> Self {
        Self { hints }
    }
}

impl SplitPlan for HintSplitPlan {
    type State = ();

    fn init_state(&self, _ctx: &VortexSession) -> VortexResult<Self::State> {
        Ok(())
    }

    fn splits<'a>(
        &'a self,
        range: Range<u64>,
        _io: &'a FileReader,
        _state: &'a Self::State,
    ) -> BoxFuture<'a, VortexResult<Vec<Range<u64>>>> {
        Box::pin(async move {
            let mut points = vec![range.start, range.end];
            points.extend(
                self.hints
                    .iter()
                    .copied()
                    .filter(|&hint| range.start < hint && hint < range.end),
            );
            points.sort_unstable();
            points.dedup();
            Ok(points
                .windows(2)
                .filter_map(|window| {
                    let range = window[0]..window[1];
                    (range.start < range.end).then_some(range)
                })
                .collect())
        })
    }

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "hint_splits")
    }
}

/// Executable ungrouped aggregate plan for one pushed expression.
pub trait AggregatePlan: 'static + Send + Sync {
    /// The per-query state this aggregate plan executes against.
    type State: Send + Sync + 'static;

    /// Create this aggregate plan's per-query state.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<Self::State>;

    /// Answer ungrouped aggregates over every row of `range`.
    ///
    /// Returns one [`AggregateAnswer`] per planned function. `None` means
    /// this plan cannot answer any function for this range and the caller
    /// should read and accumulate the range normally.
    fn aggregate_partial<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a Self::State,
    ) -> BoxFuture<'a, VortexResult<Option<Vec<AggregateAnswer>>>>;

    /// Compact description for plan display.
    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "aggregate")
    }
}

/// Object-safe view of an [`AggregatePlan`].
pub trait DynAggregatePlan: Send + Sync {
    /// Create this aggregate plan's per-query state, type-erased.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef>;

    /// Execute the planned aggregates.
    fn aggregate_partial<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Option<Vec<AggregateAnswer>>>>;

    /// Compact description for plan display.
    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl<T: AggregatePlan> DynAggregatePlan for T {
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(AggregatePlan::init_state(self, ctx)?))
    }

    fn aggregate_partial<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Option<Vec<AggregateAnswer>>>> {
        let state = match downcast_erased_state::<T::State>(state) {
            Ok(state) => state,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        AggregatePlan::aggregate_partial(self, range, io, state)
    }

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        AggregatePlan::fmt_plan(self, f)
    }
}

/// Executable metadata statistics plan for one pushed expression.
pub trait StatsPlan: 'static + Send + Sync {
    /// The per-query state this statistics plan executes against.
    type State: Send + Sync + 'static;

    /// Create this statistics plan's per-query state.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<Self::State>;

    /// Answer aggregate-function statistics over every row of `range`.
    ///
    /// The returned vector is positional against the functions passed to
    /// [`ScanNode::plan_stats`]. Each element is exact, inexact, or absent for the requested
    /// aggregate function over `range`. Implementations must not read row values merely to improve
    /// an estimate.
    fn stats<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a Self::State,
    ) -> BoxFuture<'a, VortexResult<Vec<Precision<Scalar>>>>;

    /// Compact description for plan display.
    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stats")
    }
}

/// Object-safe view of a [`StatsPlan`].
pub trait DynStatsPlan: Send + Sync {
    /// Create this statistics plan's per-query state, type-erased.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef>;

    /// Execute the planned statistics query.
    fn stats<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Vec<Precision<Scalar>>>>;

    /// Compact description for plan display.
    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl<T: StatsPlan> DynStatsPlan for T {
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(StatsPlan::init_state(self, ctx)?))
    }

    fn stats<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Vec<Precision<Scalar>>>> {
        let state = match downcast_erased_state::<T::State>(state) {
            Ok(state) => state,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        StatsPlan::stats(self, range, io, state)
    }

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        StatsPlan::fmt_plan(self, f)
    }
}

/// Virtual node that assembles a struct root value from child nodes in
/// the same row domain.
pub struct StructValueScanNode {
    names: FieldNames,
    fields: Vec<ScanNodeRef>,
    validity: Option<ScanNodeRef>,
    split_hints: OnceLock<Option<Vec<u64>>>,
}

impl StructValueScanNode {
    pub fn new(names: FieldNames, fields: Vec<ScanNodeRef>, validity: Option<ScanNodeRef>) -> Self {
        Self {
            names,
            fields,
            validity,
            split_hints: OnceLock::new(),
        }
    }

    fn compute_split_hints(&self) -> Option<Vec<u64>> {
        let mut points = Vec::new();
        for field in &self.fields {
            if let Some(hints) = field.split_hints() {
                points.extend_from_slice(hints);
            }
        }
        if let Some(validity) = &self.validity
            && let Some(hints) = validity.split_hints()
        {
            points.extend_from_slice(hints);
        }

        points.sort_unstable();
        points.dedup();
        (!points.is_empty()).then_some(points)
    }
}

/// Per-query state for a virtual struct-value node.
pub struct StructValueState {
    fields: Vec<ScanStateRef>,
    validity: Option<ScanStateRef>,
}

struct StructValueReadPlan {
    node: Arc<StructValueScanNode>,
    fields: Vec<ReadPlanRef>,
    validity: Option<ReadPlanRef>,
}

impl ScanNode for StructValueScanNode {
    type State = StructValueState;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        let fields = self
            .fields
            .iter()
            .map(|field| cx.init_node(field))
            .collect::<VortexResult<Vec<_>>>()?;
        let validity = self
            .validity
            .as_ref()
            .map(|validity| cx.init_node(validity))
            .transpose()?;
        Ok(StructValueState { fields, validity })
    }

    fn plan_read(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Option<ReadPlanRef>> {
        let fields = self
            .fields
            .iter()
            .map(|field| {
                Arc::clone(field)
                    .plan_read(cx)?
                    .ok_or_else(|| vortex_err!("struct field did not produce a read plan"))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let validity = self
            .validity
            .as_ref()
            .map(|validity| {
                Arc::clone(validity)
                    .plan_read(cx)?
                    .ok_or_else(|| vortex_err!("struct validity did not produce a read plan"))
            })
            .transpose()?;
        Ok(Some(Arc::new(StructValueReadPlan {
            node: self,
            fields,
            validity,
        })))
    }

    fn release(&self, frontier: u64, state: &Self::State) -> VortexResult<()> {
        for (field, state) in self.fields.iter().zip(&state.fields) {
            field.release(frontier, state.as_ref())?;
        }
        if let (Some(validity), Some(state)) = (&self.validity, &state.validity) {
            validity.release(frontier, state.as_ref())?;
        }
        Ok(())
    }

    fn split_hints(&self) -> Option<&[u64]> {
        self.split_hints
            .get_or_init(|| self.compute_split_hints())
            .as_deref()
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "struct_expr({})", self.names.len())
    }
}

impl ReadPlan for StructValueReadPlan {
    type State = StructValueState;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        let fields = self
            .fields
            .iter()
            .map(|field| field.init_state(cx))
            .collect::<VortexResult<Vec<_>>>()?;
        let validity = self
            .validity
            .as_ref()
            .map(|validity| validity.init_state(cx))
            .transpose()?;
        Ok(StructValueState { fields, validity })
    }

    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a FileReader,
        state: &'a Self::State,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        Box::pin(async move {
            if self.node.fields.len() != state.fields.len() {
                vortex_bail!(
                    "struct value state length {} does not match field count {}",
                    state.fields.len(),
                    self.node.fields.len()
                );
            }
            let mut arrays = Vec::with_capacity(self.fields.len());
            for (field, state) in self.fields.iter().zip(&state.fields) {
                arrays.push(
                    field
                        .read_scoped(range.clone(), rows, io, state.as_ref(), local)
                        .await?,
                );
            }
            let validity = match (&self.validity, &state.validity) {
                (Some(validity), Some(state)) => {
                    let array = validity
                        .read_scoped(range, rows, io, state.as_ref(), local)
                        .await?;
                    Validity::Array(array)
                }
                (None, None) => Validity::NonNullable,
                _ => vortex_bail!("struct value validity plan/state mismatch"),
            };
            Ok(StructArray::try_new(
                self.node.names.clone(),
                arrays,
                rows.selection.true_count(),
                validity,
            )?
            .into_array())
        })
    }

    fn segment_requests(
        &self,
        range: Range<u64>,
        rows: RowScope<'_>,
        state: &Self::State,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        if self.node.fields.len() != state.fields.len() {
            vortex_bail!(
                "struct value state length {} does not match field count {}",
                state.fields.len(),
                self.node.fields.len()
            );
        }

        let mut requests = SegmentRequests::none();
        for (field, state) in self.fields.iter().zip(&state.fields) {
            requests.extend(field.segment_requests(range.clone(), rows, state.as_ref(), cx)?);
            if requests.is_unknown() {
                return Ok(requests);
            }
        }
        match (&self.validity, &state.validity) {
            (Some(validity), Some(state)) => {
                requests.extend(validity.segment_requests(range, rows, state.as_ref(), cx)?);
            }
            (None, None) => {}
            _ => vortex_bail!("struct value validity plan/state mismatch"),
        }
        Ok(requests)
    }

    fn release(&self, frontier: u64, state: &Self::State) -> VortexResult<()> {
        for (field, state) in self.fields.iter().zip(&state.fields) {
            field.release(frontier, state.as_ref())?;
        }
        if let (Some(validity), Some(state)) = (&self.validity, &state.validity) {
            validity.release(frontier, state.as_ref())?;
        }
        Ok(())
    }

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ScanNode::fmt_chain(self.node.as_ref(), f)
    }
}

/// Virtual node that applies a scalar expression to another node's root
/// value.
pub struct ApplyScanNode {
    input: ScanNodeRef,
    expr: Expression,
}

impl ApplyScanNode {
    pub fn new(input: ScanNodeRef, expr: Expression) -> Self {
        Self { input, expr }
    }
}

struct ApplyReadPlan {
    node: Arc<ApplyScanNode>,
    input: ReadPlanRef,
}

impl ScanNode for ApplyScanNode {
    type State = ScanStateRef;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        cx.init_node(&self.input)
    }

    fn plan_read(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Option<ReadPlanRef>> {
        let input = Arc::clone(&self.input)
            .plan_read(cx)?
            .ok_or_else(|| vortex_err!("apply input did not produce a read plan"))?;
        Ok(Some(Arc::new(ApplyReadPlan { node: self, input })))
    }

    fn release(&self, frontier: u64, state: &Self::State) -> VortexResult<()> {
        self.input.release(frontier, state.as_ref())
    }

    fn split_hints(&self) -> Option<&[u64]> {
        self.input.split_hints()
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "apply({})", self.expr)
    }
}

impl ReadPlan for ApplyReadPlan {
    type State = ScanStateRef;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        self.input.init_state(cx)
    }

    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a FileReader,
        state: &'a Self::State,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        Box::pin(async move {
            let input = self
                .input
                .read_scoped(range, rows, io, state.as_ref(), local)
                .await?;
            input.apply(&self.node.expr)?.execute::<ArrayRef>(local)
        })
    }

    fn segment_requests(
        &self,
        range: Range<u64>,
        rows: RowScope<'_>,
        state: &Self::State,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        self.input.segment_requests(range, rows, state.as_ref(), cx)
    }

    fn release(&self, frontier: u64, state: &Self::State) -> VortexResult<()> {
        self.input.release(frontier, state.as_ref())
    }

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ScanNode::fmt_chain(self.node.as_ref(), f)
    }
}

/// Executable predicate evidence for one planned predicate expression.
pub trait EvidencePlan: 'static + Send + Sync {
    /// The per-query state this evidence plan executes against.
    type State: Send + Sync + 'static;

    /// Create this evidence plan's per-query state.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<Self::State>;

    /// Produce evidence for the planned predicate over `req.range`.
    fn evidence<'a>(
        &'a self,
        req: &'a EvidenceRequest<'a>,
        io: &'a FileReader,
        state: &'a Self::State,
    ) -> BoxFuture<'a, VortexResult<Vec<EvidenceFragment>>>;

    /// Return scheduler-visible segment requests needed for this evidence, when known exactly.
    fn segment_requests(
        &self,
        _req: &EvidenceRequest<'_>,
        _state: &Self::State,
        _cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        Ok(SegmentRequests::unknown())
    }

    /// A key for sharing this plan's state with sibling evidence plans
    /// in the same file. The default keeps one state per planned route.
    fn state_cache_key(&self) -> Option<EvidenceStateKey> {
        None
    }

    /// Whether this plan is cheap enough to re-run immediately before a
    /// projection read when a dynamic predicate boundary changes while
    /// the morsel is in flight.
    fn recheck_before_projection(&self) -> bool {
        false
    }

    /// Compact description for plan display.
    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "evidence")
    }
}

/// Object-safe view of an [`EvidencePlan`].
pub trait DynEvidencePlan: Send + Sync {
    /// Create this evidence plan's per-query state, type-erased.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef>;

    /// Produce evidence for the planned predicate over `req.range`.
    fn evidence<'a>(
        &'a self,
        req: &'a EvidenceRequest<'a>,
        io: &'a FileReader,
        state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Vec<EvidenceFragment>>>;

    /// Return scheduler-visible segment requests needed for this evidence, when known exactly.
    fn segment_requests(
        &self,
        req: &EvidenceRequest<'_>,
        state: &ScanState,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests>;

    /// A key for sharing this plan's state with sibling evidence plans.
    fn state_cache_key(&self) -> Option<EvidenceStateKey>;

    /// Whether this plan should run in the projection recheck pass.
    fn recheck_before_projection(&self) -> bool;

    /// Compact description for plan display.
    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl<T: EvidencePlan> DynEvidencePlan for T {
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(EvidencePlan::init_state(self, ctx)?))
    }

    fn evidence<'a>(
        &'a self,
        req: &'a EvidenceRequest<'a>,
        io: &'a FileReader,
        state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Vec<EvidenceFragment>>> {
        let state = match downcast_erased_state::<T::State>(state) {
            Ok(state) => state,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        EvidencePlan::evidence(self, req, io, state)
    }

    fn segment_requests(
        &self,
        req: &EvidenceRequest<'_>,
        state: &ScanState,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        let state = downcast_erased_state::<T::State>(state)?;
        EvidencePlan::segment_requests(self, req, state, cx)
    }

    fn state_cache_key(&self) -> Option<EvidenceStateKey> {
        EvidencePlan::state_cache_key(self)
    }

    fn recheck_before_projection(&self) -> bool {
        EvidencePlan::recheck_before_projection(self)
    }

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        EvidencePlan::fmt_plan(self, f)
    }
}

fn downcast_erased_state<T: Send + Sync + 'static>(state: &ScanState) -> VortexResult<&T> {
    state.downcast_ref::<T>().ok_or_else(|| {
        vortex_err!(
            "scan2 state type mismatch: expected {}",
            std::any::type_name::<T>()
        )
    })
}

/// Object-safe view of a [`LayoutScanRule`]. Blanket-implemented; never
/// by hand.
pub trait DynLayoutScanRule: fmt::Debug + Send + Sync {
    /// The layout encoding this rule reads.
    fn id(&self) -> LayoutEncodingId;

    /// Expand one layout node into a type-erased scan2 node.
    fn expand(
        &self,
        layout: &LayoutRef,
        req: &mut NodeRequest,
        cx: &ExpandCtx,
    ) -> VortexResult<ScanNodeRef>;
}

impl<R: LayoutScanRule> DynLayoutScanRule for R {
    fn id(&self) -> LayoutEncodingId {
        LayoutScanRule::id(self)
    }

    fn expand(
        &self,
        layout: &LayoutRef,
        req: &mut NodeRequest,
        cx: &ExpandCtx,
    ) -> VortexResult<ScanNodeRef> {
        Ok(Arc::new(LayoutScanRule::expand(self, layout, req, cx)?))
    }
}

/// Recover a node's concrete file/query global state from its erased form.
pub(crate) fn downcast_state<T: ScanNode>(state: &ScanState) -> VortexResult<&T::State> {
    state.downcast_ref::<T::State>().ok_or_else(|| {
        vortex_err!(
            "scan2 state type mismatch: expected {}",
            std::any::type_name::<T::State>()
        )
    })
}

/// Resolves layout encodings to their registered scan2 rules during
/// expansion. Rules recurse into child layouts through
/// [`ExpandCtx::expand`] (passing the scoped request through
/// row-preserving children) or [`ExpandCtx::expand_free`] (for children
/// in another row domain, and for lazy runtime expansion).
#[derive(Clone)]
pub struct ExpandCtx {
    session: VortexSession,
}

impl ExpandCtx {
    /// An expansion context resolving rules from `session`.
    pub fn new(session: VortexSession) -> Self {
        Self { session }
    }

    /// The session rules are resolved from.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// Expand `layout` through its encoding's registered scan2 rule,
    /// negotiating `req` on the way down.
    pub fn expand(&self, layout: &LayoutRef, req: &mut NodeRequest) -> VortexResult<ScanNodeRef> {
        let id = layout.encoding_id();
        let rule = self.session.scan_v2_rules().find(&id).ok_or_else(|| {
            vortex_err!(
                "no scan2 rule registered for layout encoding {id}; register one with \
                 ScanV2Session::register"
            )
        })?;
        rule.expand(layout, req, self)
    }

    /// Expand `layout` with an empty request: for children in another row
    /// domain (dictionary values, zone tables, index postings) and for
    /// chunk children expanded lazily at runtime.
    pub fn expand_free(&self, layout: &LayoutRef) -> VortexResult<ScanNodeRef> {
        self.expand(layout, &mut NodeRequest::empty())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::dtype::Nullability;

    use super::*;
    use crate::segments::TestSegments;

    struct TestStatsNode;

    impl ScanNode for TestStatsNode {
        type State = ();

        fn init_state(&self, _cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
            Ok(())
        }

        fn plan_read(self: Arc<Self>, _cx: &mut PlanCtx) -> VortexResult<Option<ReadPlanRef>> {
            Ok(None)
        }

        fn plan_stats(
            self: Arc<Self>,
            funcs: &[AggregateFnRef],
            _cx: &mut PlanCtx,
        ) -> VortexResult<Option<StatsPlanRef>> {
            Ok(Some(Arc::new(TestStatsPlan { len: funcs.len() })))
        }

        fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test_stats")
        }
    }

    struct TestStatsPlan {
        len: usize,
    }

    impl StatsPlan for TestStatsPlan {
        type State = ();

        fn init_state(&self, _ctx: &VortexSession) -> VortexResult<Self::State> {
            Ok(())
        }

        fn stats<'a>(
            &'a self,
            range: Range<u64>,
            _io: &'a FileReader,
            _state: &'a Self::State,
        ) -> BoxFuture<'a, VortexResult<Vec<Precision<Scalar>>>> {
            Box::pin(async move {
                let mut stats = Vec::with_capacity(self.len);
                for idx in 0..self.len {
                    let idx = u64::try_from(idx)?;
                    stats.push(Precision::exact(Scalar::primitive(
                        range.start + idx,
                        Nullability::NonNullable,
                    )));
                }
                Ok(stats)
            })
        }
    }

    #[test]
    fn stats_plan_erasure_preserves_positional_results() -> VortexResult<()> {
        let session = VortexSession::empty();
        let node: ScanNodeRef = Arc::new(TestStatsNode);
        let funcs = vec![Min.bind(EmptyOptions), Max.bind(EmptyOptions)];

        let plan = node
            .plan_stats(&funcs, &mut PlanCtx::new(session.clone()))?
            .ok_or_else(|| vortex_err!("test scan node did not return a stats plan"))?;
        let state = plan.init_state(&session)?;
        let io = FileReader::new(Arc::new(TestSegments::default()), session);
        let stats = futures::executor::block_on(plan.stats(10..20, &io, state.as_ref()))?;

        assert_eq!(stats.len(), funcs.len());
        assert!(matches!(stats[0], Precision::Exact(_)));
        assert!(matches!(stats[1], Precision::Exact(_)));

        Ok(())
    }
}
