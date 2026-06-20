// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Physical scan plans with value, proof, and mask capabilities.
//!
//! A [`ScanPlan`] is immutable physical scan structure. Layouts are one way to
//! instantiate scan plans, but the runtime traits in this module are not tied to
//! serialized layouts. Engines work through the blanket-implemented
//! [`DynScanPlan`] adapter:
//!
//! - expression pushdown returns another scan plan whose root value is
//!   the pushed expression, so reads and evidence are prepared from
//!   `root()` of that plan instead of reparsing expressions; and
//! - executable prepared reads use one scoped primitive: selection
//!   controls output cardinality, and demand controls which selected rows
//!   must contain meaningful values.

pub mod evidence;
pub mod request;

use std::any::TypeId;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::StructArray;
use vortex_array::builtins::ArrayBuiltins;
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

use self::evidence::EvidenceFragment;
use self::request::EvidenceRequest;
use self::request::OwnedEvidenceRequest;
use crate::segments::SegmentPlanCtx;
use crate::segments::SegmentRequests;
use crate::segments::SegmentSource;

/// Per-file/query IO context for scan plan reads.
#[derive(Clone)]
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

/// A scan plan's per-file/query global state, type-erased.
pub type ScanState = dyn std::any::Any + Send + Sync;

/// A reference to a scan plan's per-file/query global state.
pub type ScanStateRef = Arc<ScanState>;

/// A reference-counted, type-erased scan plan.
pub type ScanPlanRef = Arc<dyn DynScanPlan>;

/// A reference-counted, type-erased prepared evidence handle.
pub type PreparedEvidenceRef = Arc<dyn PreparedEvidence>;

/// A reference-counted, type-erased prepared read handle.
pub type PreparedReadRef = Arc<dyn PreparedRead>;

/// A reference-counted, type-erased prepared split handle.
pub type PreparedSplitRef = Arc<dyn DynPreparedSplit>;

/// A reference-counted, type-erased prepared ungrouped aggregate handle.
pub type PreparedAggregateRef = Arc<dyn DynPreparedAggregate>;

/// A reference-counted, type-erased prepared metadata statistics handle.
pub type PreparedStatsRef = Arc<dyn DynPreparedStats>;

/// Per-file/query cache of scan-plan global state while a file's planned
/// reads are initialized.
pub type ScanStateCache = FxHashMap<usize, ScanStateRef>;

/// Context for expression pushdown.
pub struct PushCtx {
    session: VortexSession,
}

impl PushCtx {
    /// Create an expression-pushdown context for one scan session.
    pub fn new(session: VortexSession) -> Self {
        Self { session }
    }

    /// Return the scan session used while pushing expressions.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }
}

/// Context for turning pushed expressions into prepared read/evidence handles.
pub struct PrepareCtx {
    session: VortexSession,
    state_cache: PreparedStateCacheRef,
}

impl PrepareCtx {
    /// Create a preparation context with an empty prepared-state cache.
    pub fn new(session: VortexSession) -> Self {
        Self::with_state_cache(session, Arc::new(PreparedStateCache::default()))
    }

    /// Create a preparation context backed by an existing prepared-state cache.
    pub fn with_state_cache(session: VortexSession, state_cache: PreparedStateCacheRef) -> Self {
        Self {
            session,
            state_cache,
        }
    }

    /// Return the scan session used while preparing runtime handles.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// The prepared-state cache backing this context.
    pub fn state_cache(&self) -> PreparedStateCacheRef {
        Arc::clone(&self.state_cache)
    }

    /// Return shared prepared state for `key`, initializing it on first use.
    pub fn shared_state<T>(
        &mut self,
        key: PreparedStateKey,
        init: impl FnOnce() -> VortexResult<T>,
    ) -> VortexResult<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        if let Some(hit) = self.state_cache.shared_states.lock().get(&key) {
            return Arc::downcast::<T>(Arc::clone(hit))
                .map_err(|_| vortex_err!("prepared shared state type mismatch"));
        }

        let state = Arc::new(init()?);
        let mut shared_states = self.state_cache.shared_states.lock();
        if let Some(hit) = shared_states.get(&key) {
            return Arc::downcast::<T>(Arc::clone(hit))
                .map_err(|_| vortex_err!("prepared shared state type mismatch"));
        }
        shared_states.insert(key, Arc::<T>::clone(&state));
        Ok(state)
    }
}

/// Shared cache for scan/file-level prepared state.
#[derive(Default)]
pub struct PreparedStateCache {
    shared_states: Mutex<FxHashMap<PreparedStateKey, ScanStateRef>>,
}

/// Reference-counted prepared-state cache.
pub type PreparedStateCacheRef = Arc<PreparedStateCache>;

/// A typed key for prepared-file shared state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PreparedStateKey {
    type_id: TypeId,
    key: usize,
}

impl PreparedStateKey {
    /// Create a key scoped by the caller's concrete state type and numeric identity.
    pub fn new<T: 'static>(key: usize) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            key,
        }
    }
}

/// Context for initializing type-erased scan-plan state used by release and
/// non-read prepared paths.
pub struct StateCtx<'a> {
    session: &'a VortexSession,
    plan_cache: &'a mut ScanStateCache,
}

impl<'a> StateCtx<'a> {
    /// Create a state-initialization context backed by a scan-plan state cache.
    pub fn new(session: &'a VortexSession, plan_cache: &'a mut ScanStateCache) -> Self {
        Self {
            session,
            plan_cache,
        }
    }

    /// Return the scan session used while initializing plan state.
    pub fn session(&self) -> &VortexSession {
        self.session
    }

    /// Initialize or reuse state for a child plan.
    pub fn init_plan(&mut self, plan: &ScanPlanRef) -> VortexResult<ScanStateRef> {
        let key = scan_plan_key(plan);
        if let Some(hit) = self.plan_cache.get(&key) {
            return Ok(Arc::clone(hit));
        }
        let state = plan.init_state(self)?;
        self.plan_cache.insert(key, Arc::clone(&state));
        Ok(state)
    }
}

fn scan_plan_key(plan: &ScanPlanRef) -> usize {
    Arc::as_ptr(plan) as *const () as usize
}

/// One operation's row scope in a scan plan's input row domain.
#[derive(Clone, Copy, Debug)]
pub struct RowScope<'a> {
    /// Rows still semantically live in the input domain.
    pub selection: &'a Mask,
    /// Rows whose value/result is needed by this operation.
    pub demand: &'a Mask,
}

impl<'a> RowScope<'a> {
    /// Create a scope where every selected row is demanded.
    pub fn selected(selection: &'a Mask) -> Self {
        Self {
            selection,
            demand: selection,
        }
    }

    /// Create a scope, validating that demand is a subset of selection.
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

    /// Return whether every selected row is demanded.
    pub fn demands_all_selected(self) -> bool {
        std::ptr::eq(self.selection, self.demand)
            || self.demand.true_count() == self.selection.true_count()
    }
}

/// Owned row scope for a morsel-level read task.
#[derive(Clone, Debug)]
pub struct OwnedRowScope {
    selection: Mask,
    demand: Mask,
}

impl OwnedRowScope {
    /// Create an owned scope where every selected row is demanded.
    pub fn selected(selection: Mask) -> Self {
        Self {
            demand: selection.clone(),
            selection,
        }
    }

    /// Create an owned scope, validating that demand is a subset of selection.
    pub fn try_new(selection: Mask, demand: Mask) -> VortexResult<Self> {
        RowScope::try_new(&selection, &demand)?;
        Ok(Self { selection, demand })
    }

    /// Borrow this owned scope as a [`RowScope`].
    pub fn as_scope(&self) -> RowScope<'_> {
        RowScope {
            selection: &self.selection,
            demand: &self.demand,
        }
    }
}

/// One prepared aggregate's mixed-coverage answer.
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
    /// disjoint ascending spans in this plan's row coordinates.
    pub residual: Vec<Range<u64>>,
}

/// A plan in a physical scan tree.
///
/// A `ScanPlan` is immutable physical scan structure: metadata, child plan
/// references, pushdown behavior, and split hints. Runtime caches live in state
/// objects created while preparing reads, evidence, statistics, and aggregates for
/// a file scan.
pub trait ScanPlan: 'static + Send + Sync {
    /// Per-file/query plan state: decoded arrays, decoded index state, child plan states, and
    /// other frontier-released caches shared by prepared handles for this plan.
    type State: Send + Sync + 'static;

    /// Create this plan's per-file/query state.
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State>;

    /// Try to push `expr` into this plan's row domain. The returned plan's
    /// root value is exactly `expr` in the input row domain.
    ///
    /// The default accepts `root()` as this plan and otherwise builds a
    /// generic scalar-apply plan over this plan's root value. Layouts
    /// specialize when they can route or rewrite the expression, e.g.
    /// struct field access or list-offset functions.
    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        _cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>>
    where
        Self: Sized,
    {
        if is_root(expr) {
            Ok(Some(self))
        } else {
            Ok(Some(Arc::new(ApplyScanPlan::new(self, expr.clone()))))
        }
    }

    /// Prepare value reads for this plan's root value.
    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>>;

    /// Prepare natural row splits for this plan's root value.
    ///
    /// The default converts this plan's cheap split hints into an executable handle. Plans can
    /// override this when split discovery needs request-specific state, I/O, or cost estimates.
    fn prepare_splits(
        self: Arc<Self>,
        _cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedSplitRef>>
    where
        Self: Sized,
    {
        Ok(self
            .split_hints()
            .map(|hints| Arc::new(HintPreparedSplit::new(hints.to_vec())) as PreparedSplitRef))
    }

    /// Prepare predicate evidence for this plan's root boolean value.
    ///
    /// Preparation performs no IO and returns a direct executable handle. The
    /// handle may precompute expression rewrites or accepted predicate
    /// fragments, but runtime state remains in [`Self::State`].
    fn prepare_evidence(
        self: Arc<Self>,
        _cx: &mut PrepareCtx,
    ) -> VortexResult<Vec<PreparedEvidenceRef>>
    where
        Self: Sized,
    {
        Ok(Vec::new())
    }

    /// Prepare ungrouped aggregates over this plan's root value.
    ///
    /// The returned handle answers all `funcs` together over a runtime row
    /// range, producing one [`AggregateAnswer`] per function. `None` means
    /// this plan cannot answer these aggregates from layout metadata and
    /// the caller should read rows normally.
    fn prepare_aggregate_partial(
        self: Arc<Self>,
        _funcs: &[AggregateFnRef],
        _cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedAggregateRef>>
    where
        Self: Sized,
    {
        Ok(None)
    }

    /// Prepare metadata statistics for this plan's root value.
    ///
    /// The returned handle answers the requested aggregate functions positionally over runtime row
    /// ranges using metadata only. `None` means this plan cannot answer these functions from
    /// metadata.
    fn prepare_stats(
        self: Arc<Self>,
        _funcs: &[AggregateFnRef],
        _cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedStatsRef>>
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

/// Read every row in `range` through a prepared read.
pub fn read_dense<'a>(
    read: &'a PreparedReadRef,
    range: Range<u64>,
    io: &'a FileReader,
) -> BoxFuture<'a, VortexResult<ArrayRef>> {
    Box::pin(async move {
        let len = range_len(&range)?;
        let rows = OwnedRowScope::selected(Mask::new_true(len));
        let mut local = io.session().create_execution_ctx();
        let task = Arc::clone(read).begin_read(range, rows)?;
        task.read(io, &mut local).await
    })
}

fn range_len(range: &Range<u64>) -> VortexResult<usize> {
    let len = range
        .end
        .checked_sub(range.start)
        .ok_or_else(|| vortex_err!("read range end is before start: {range:?}"))?;
    usize::try_from(len).map_err(|_| vortex_err!("read range exceeds usize"))
}

/// Object-safe view of a [`ScanPlan`]. Blanket-implemented; never by
/// hand.
pub trait DynScanPlan: Send + Sync {
    /// Create this plan's per-file/query state, type-erased.
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef>;

    /// Try to push an expression into this plan's row domain.
    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>>;

    /// Prepare value reads for this plan's root value.
    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>>;

    /// Prepare natural row splits for this plan's root value.
    fn prepare_splits(
        self: Arc<Self>,
        cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedSplitRef>>;

    /// Prepare predicate evidence for this plan's root boolean value.
    fn prepare_evidence(
        self: Arc<Self>,
        cx: &mut PrepareCtx,
    ) -> VortexResult<Vec<PreparedEvidenceRef>>;

    /// Prepare ungrouped aggregates for this plan's root value.
    fn prepare_aggregate_partial(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedAggregateRef>>;

    /// Prepare metadata statistics for this plan's root value.
    fn prepare_stats(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedStatsRef>>;

    /// Preferred morsel boundaries (see [`ScanPlan::split_hints`]).
    fn split_hints(&self) -> Option<&[u64]>;

    /// Release state behind the frontier (see [`ScanPlan::release`]).
    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()>;

    /// Reader-chain description for plan display.
    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl<T: ScanPlan> DynScanPlan for T {
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(ScanPlan::init_state(self, cx)?))
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
        ScanPlan::try_push_expr(self, expr, cx)
    }

    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>> {
        ScanPlan::prepare_read(self, cx)
    }

    fn prepare_splits(
        self: Arc<Self>,
        cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedSplitRef>> {
        ScanPlan::prepare_splits(self, cx)
    }

    fn prepare_evidence(
        self: Arc<Self>,
        cx: &mut PrepareCtx,
    ) -> VortexResult<Vec<PreparedEvidenceRef>> {
        ScanPlan::prepare_evidence(self, cx)
    }

    fn prepare_aggregate_partial(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedAggregateRef>> {
        ScanPlan::prepare_aggregate_partial(self, funcs, cx)
    }

    fn prepare_stats(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedStatsRef>> {
        ScanPlan::prepare_stats(self, funcs, cx)
    }

    fn split_hints(&self) -> Option<&[u64]> {
        ScanPlan::split_hints(self)
    }

    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()> {
        ScanPlan::release(self, frontier, downcast_state::<T>(state)?)
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ScanPlan::fmt_chain(self, f)
    }
}

/// Prepared value read for one pushed expression.
///
/// A `PreparedRead` is the scan-level runtime handle for a fixed read route. It
/// may hold child prepared reads and initializes route-scoped state once per
/// prepared file scan; each `read_scoped` call executes that route for one
/// morsel row scope.
pub trait PreparedRead: 'static + Send + Sync {
    /// Read the live rows of `range`, with [`RowScope`] defining output
    /// cardinality (`selection`) and meaningful-value demand (`demand`).
    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a FileReader,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>>;

    /// Return scheduler-visible segment requests needed for this read, when known exactly.
    fn segment_requests(
        &self,
        _range: Range<u64>,
        _rows: RowScope<'_>,
        _cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        Ok(SegmentRequests::unknown())
    }

    /// Release state behind the completed-row frontier.
    fn release(&self, _frontier: u64) -> VortexResult<()> {
        Ok(())
    }

    /// Compact description for plan display.
    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "read")
    }
}

impl dyn PreparedRead {
    /// Create a morsel-level read task for this prepared read.
    pub fn begin_read(
        self: Arc<Self>,
        range: Range<u64>,
        rows: OwnedRowScope,
    ) -> VortexResult<Box<dyn ReadTask>> {
        Ok(Box::new(DefaultReadTask {
            read: self,
            range,
            rows,
        }))
    }
}

/// A morsel-level read task.
pub trait ReadTask: Send {
    /// Return scheduler-visible segment requests needed for this task, when known exactly.
    fn segment_requests(&self, cx: &mut SegmentPlanCtx) -> VortexResult<SegmentRequests>;

    /// Execute the read task.
    fn read<'a>(
        self: Box<Self>,
        io: &'a FileReader,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>>;
}

struct DefaultReadTask {
    read: PreparedReadRef,
    range: Range<u64>,
    rows: OwnedRowScope,
}

impl ReadTask for DefaultReadTask {
    fn segment_requests(&self, cx: &mut SegmentPlanCtx) -> VortexResult<SegmentRequests> {
        self.read
            .segment_requests(self.range.clone(), self.rows.as_scope(), cx)
    }

    fn read<'a>(
        self: Box<Self>,
        io: &'a FileReader,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        Box::pin(async move {
            self.read
                .read_scoped(self.range, self.rows.as_scope(), io, local)
                .await
        })
    }
}

/// Prepared split discovery for one pushed expression.
pub trait PreparedSplit: 'static + Send + Sync {
    /// The per-query state this prepared split executes against.
    type State: Send + Sync + 'static;

    /// Create this prepared split's per-query state.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<Self::State>;

    /// Return natural row ranges inside `range`.
    fn splits<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a Self::State,
    ) -> BoxFuture<'a, VortexResult<Vec<Range<u64>>>>;

    /// Compact description for plan display.
    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "splits")
    }
}

/// Object-safe view of a [`PreparedSplit`].
pub trait DynPreparedSplit: Send + Sync {
    /// Create this prepared split's per-query state, type-erased.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef>;

    /// Execute the prepared split query.
    fn splits<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Vec<Range<u64>>>>;

    /// Compact description for plan display.
    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl<T: PreparedSplit> DynPreparedSplit for T {
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(PreparedSplit::init_state(self, ctx)?))
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
        PreparedSplit::splits(self, range, io, state)
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        PreparedSplit::fmt_prepared(self, f)
    }
}

struct HintPreparedSplit {
    hints: Vec<u64>,
}

impl HintPreparedSplit {
    fn new(hints: Vec<u64>) -> Self {
        Self { hints }
    }
}

impl PreparedSplit for HintPreparedSplit {
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

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "hint_splits")
    }
}

/// Prepared ungrouped aggregate for one pushed expression.
pub trait PreparedAggregate: 'static + Send + Sync {
    /// The per-query state this prepared aggregate executes against.
    type State: Send + Sync + 'static;

    /// Create this prepared aggregate's per-query state.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<Self::State>;

    /// Answer ungrouped aggregates over every row of `range`.
    ///
    /// Returns one [`AggregateAnswer`] per prepared function. `None` means
    /// this prepared aggregate cannot answer any function for this range and the caller
    /// should read and accumulate the range normally.
    fn aggregate_partial<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a Self::State,
    ) -> BoxFuture<'a, VortexResult<Option<Vec<AggregateAnswer>>>>;

    /// Compact description for plan display.
    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "aggregate")
    }
}

/// Object-safe view of a [`PreparedAggregate`].
pub trait DynPreparedAggregate: Send + Sync {
    /// Create this prepared aggregate's per-query state, type-erased.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef>;

    /// Execute the prepared aggregates.
    fn aggregate_partial<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Option<Vec<AggregateAnswer>>>>;

    /// Compact description for plan display.
    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl<T: PreparedAggregate> DynPreparedAggregate for T {
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(PreparedAggregate::init_state(self, ctx)?))
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
        PreparedAggregate::aggregate_partial(self, range, io, state)
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        PreparedAggregate::fmt_prepared(self, f)
    }
}

/// Prepared metadata statistics for one pushed expression.
pub trait PreparedStats: 'static + Send + Sync {
    /// The per-query state this prepared statistics handle executes against.
    type State: Send + Sync + 'static;

    /// Create this prepared statistics handle's per-query state.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<Self::State>;

    /// Answer aggregate-function statistics over every row of `range`.
    ///
    /// The returned vector is positional against the functions passed to
    /// [`ScanPlan::prepare_stats`]. Each element is exact, inexact, or absent for the requested
    /// aggregate function over `range`. Implementations must not read row values merely to improve
    /// an estimate.
    fn stats<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a Self::State,
    ) -> BoxFuture<'a, VortexResult<Vec<Precision<Scalar>>>>;

    /// Compact description for plan display.
    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stats")
    }
}

/// Object-safe view of a [`PreparedStats`].
pub trait DynPreparedStats: Send + Sync {
    /// Create this prepared statistics handle's per-query state, type-erased.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef>;

    /// Execute the prepared statistics query.
    fn stats<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Vec<Precision<Scalar>>>>;

    /// Compact description for plan display.
    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl<T: PreparedStats> DynPreparedStats for T {
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(PreparedStats::init_state(self, ctx)?))
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
        PreparedStats::stats(self, range, io, state)
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        PreparedStats::fmt_prepared(self, f)
    }
}

/// Virtual plan that assembles a struct root value from child plans in
/// the same row domain.
pub struct StructValueScanPlan {
    names: FieldNames,
    fields: Vec<ScanPlanRef>,
    validity: Option<ScanPlanRef>,
    split_hints: OnceLock<Option<Vec<u64>>>,
}

impl StructValueScanPlan {
    /// Create a virtual plan that assembles a struct from child field plans.
    pub fn new(names: FieldNames, fields: Vec<ScanPlanRef>, validity: Option<ScanPlanRef>) -> Self {
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

/// Per-query state for a virtual struct-value plan.
pub struct StructValueState {
    fields: Vec<ScanStateRef>,
    validity: Option<ScanStateRef>,
}

struct StructValuePreparedRead {
    plan: Arc<StructValueScanPlan>,
    fields: Vec<PreparedReadRef>,
    validity: Option<PreparedReadRef>,
}

impl ScanPlan for StructValueScanPlan {
    type State = StructValueState;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        let fields = self
            .fields
            .iter()
            .map(|field| cx.init_plan(field))
            .collect::<VortexResult<Vec<_>>>()?;
        let validity = self
            .validity
            .as_ref()
            .map(|validity| cx.init_plan(validity))
            .transpose()?;
        Ok(StructValueState { fields, validity })
    }

    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>> {
        let fields = self
            .fields
            .iter()
            .map(|field| {
                Arc::clone(field)
                    .prepare_read(cx)?
                    .ok_or_else(|| vortex_err!("struct field did not produce a prepared read"))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let validity = self
            .validity
            .as_ref()
            .map(|validity| {
                Arc::clone(validity)
                    .prepare_read(cx)?
                    .ok_or_else(|| vortex_err!("struct validity did not produce a prepared read"))
            })
            .transpose()?;
        Ok(Some(Arc::new(StructValuePreparedRead {
            plan: self,
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

impl PreparedRead for StructValuePreparedRead {
    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a FileReader,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        Box::pin(async move {
            let mut arrays = Vec::with_capacity(self.fields.len());
            for field in &self.fields {
                arrays.push(field.read_scoped(range.clone(), rows, io, local).await?);
            }
            let validity = match &self.validity {
                Some(validity) => {
                    let array = validity.read_scoped(range, rows, io, local).await?;
                    Validity::Array(array)
                }
                None => Validity::NonNullable,
            };
            Ok(StructArray::try_new(
                self.plan.names.clone(),
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
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        let mut requests = SegmentRequests::none();
        for field in &self.fields {
            requests.extend(field.segment_requests(range.clone(), rows, cx)?);
            if requests.is_unknown() {
                return Ok(requests);
            }
        }
        if let Some(validity) = &self.validity {
            requests.extend(validity.segment_requests(range, rows, cx)?);
        }
        Ok(requests)
    }

    fn release(&self, frontier: u64) -> VortexResult<()> {
        for field in &self.fields {
            field.release(frontier)?;
        }
        if let Some(validity) = &self.validity {
            validity.release(frontier)?;
        }
        Ok(())
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ScanPlan::fmt_chain(self.plan.as_ref(), f)
    }
}

/// Virtual plan that applies a scalar expression to another plan's root
/// value.
pub struct ApplyScanPlan {
    input: ScanPlanRef,
    expr: Expression,
}

impl ApplyScanPlan {
    /// Create a virtual plan that applies `expr` to `input`.
    pub fn new(input: ScanPlanRef, expr: Expression) -> Self {
        Self { input, expr }
    }
}

struct ApplyPreparedRead {
    plan: Arc<ApplyScanPlan>,
    input: PreparedReadRef,
}

impl ScanPlan for ApplyScanPlan {
    type State = ScanStateRef;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        cx.init_plan(&self.input)
    }

    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>> {
        let input = Arc::clone(&self.input)
            .prepare_read(cx)?
            .ok_or_else(|| vortex_err!("apply input did not produce a prepared read"))?;
        Ok(Some(Arc::new(ApplyPreparedRead { plan: self, input })))
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

impl PreparedRead for ApplyPreparedRead {
    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a FileReader,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        Box::pin(async move {
            let input = self.input.read_scoped(range, rows, io, local).await?;
            input.apply(&self.plan.expr)?.execute::<ArrayRef>(local)
        })
    }

    fn segment_requests(
        &self,
        range: Range<u64>,
        rows: RowScope<'_>,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        self.input.segment_requests(range, rows, cx)
    }

    fn release(&self, frontier: u64) -> VortexResult<()> {
        self.input.release(frontier)
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ScanPlan::fmt_chain(self.plan.as_ref(), f)
    }
}

/// Virtual plan that applies a parent struct's validity to another plan's root
/// value.
///
/// Reads the `input` value and a non-nullable boolean `validity` array in the
/// same row domain and produces `mask(input, validity)`: rows where validity is
/// false become null. This preserves parent-struct validity when a single field
/// is projected out of a nullable struct.
pub struct MaskScanPlan {
    input: ScanPlanRef,
    validity: ScanPlanRef,
}

impl MaskScanPlan {
    /// Create a plan that masks `input` with a parent struct's `validity`.
    ///
    /// `validity` must read a non-nullable boolean array in the same row domain
    /// as `input` (the struct layout's validity child).
    pub fn new(input: ScanPlanRef, validity: ScanPlanRef) -> Self {
        Self { input, validity }
    }
}

/// Per-query state for a [`MaskScanPlan`].
pub struct MaskState {
    input: ScanStateRef,
    validity: ScanStateRef,
}

struct MaskPreparedRead {
    plan: Arc<MaskScanPlan>,
    input: PreparedReadRef,
    validity: PreparedReadRef,
}

impl ScanPlan for MaskScanPlan {
    type State = MaskState;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        Ok(MaskState {
            input: cx.init_plan(&self.input)?,
            validity: cx.init_plan(&self.validity)?,
        })
    }

    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>> {
        let input = Arc::clone(&self.input)
            .prepare_read(cx)?
            .ok_or_else(|| vortex_err!("mask input did not produce a prepared read"))?;
        let validity = Arc::clone(&self.validity)
            .prepare_read(cx)?
            .ok_or_else(|| vortex_err!("mask validity did not produce a prepared read"))?;
        Ok(Some(Arc::new(MaskPreparedRead {
            plan: self,
            input,
            validity,
        })))
    }

    fn release(&self, frontier: u64, state: &Self::State) -> VortexResult<()> {
        self.input.release(frontier, state.input.as_ref())?;
        self.validity.release(frontier, state.validity.as_ref())
    }

    fn split_hints(&self) -> Option<&[u64]> {
        self.input.split_hints()
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mask:")?;
        self.input.fmt_chain(f)
    }
}

impl PreparedRead for MaskPreparedRead {
    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a FileReader,
        local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        Box::pin(async move {
            let input = self
                .input
                .read_scoped(range.clone(), rows, io, local)
                .await?;
            let validity = self.validity.read_scoped(range, rows, io, local).await?;
            input.mask(validity)?.execute::<ArrayRef>(local)
        })
    }

    fn release(&self, frontier: u64) -> VortexResult<()> {
        self.input.release(frontier)?;
        self.validity.release(frontier)
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ScanPlan::fmt_chain(self.plan.as_ref(), f)
    }
}

/// Prepared predicate evidence for one predicate expression.
pub trait PreparedEvidence: 'static + Send + Sync {
    /// Produce evidence for the prepared predicate over `req.range`.
    fn evidence<'a>(
        &'a self,
        req: &'a EvidenceRequest<'a>,
        io: &'a FileReader,
    ) -> BoxFuture<'a, VortexResult<Vec<EvidenceFragment>>>;

    /// Return scheduler-visible segment requests needed for this evidence, when known exactly.
    fn segment_requests(
        &self,
        _req: &EvidenceRequest<'_>,
        _cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        Ok(SegmentRequests::unknown())
    }

    /// Whether this handle is cheap enough to re-run immediately before a
    /// projection read when a dynamic predicate boundary changes while
    /// the morsel is in flight.
    fn recheck_before_projection(&self) -> bool {
        false
    }

    /// Compact description for plan display.
    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "evidence")
    }
}

impl dyn PreparedEvidence {
    /// Create a morsel-level evidence task for this prepared evidence handle.
    pub fn begin_evidence(
        self: Arc<Self>,
        req: OwnedEvidenceRequest,
    ) -> VortexResult<Box<dyn EvidenceTask>> {
        Ok(Box::new(DefaultEvidenceTask {
            evidence: self,
            req,
        }))
    }
}

/// A morsel-level evidence task.
pub trait EvidenceTask: Send {
    /// Return scheduler-visible segment requests needed for this task, when known exactly.
    fn segment_requests(&self, cx: &mut SegmentPlanCtx) -> VortexResult<SegmentRequests>;

    /// Execute the evidence task.
    fn evidence<'a>(
        self: Box<Self>,
        io: &'a FileReader,
    ) -> BoxFuture<'a, VortexResult<Vec<EvidenceFragment>>>;
}

struct DefaultEvidenceTask {
    evidence: PreparedEvidenceRef,
    req: OwnedEvidenceRequest,
}

impl EvidenceTask for DefaultEvidenceTask {
    fn segment_requests(&self, cx: &mut SegmentPlanCtx) -> VortexResult<SegmentRequests> {
        self.evidence.segment_requests(&self.req.as_request(), cx)
    }

    fn evidence<'a>(
        self: Box<Self>,
        io: &'a FileReader,
    ) -> BoxFuture<'a, VortexResult<Vec<EvidenceFragment>>> {
        Box::pin(async move { self.evidence.evidence(&self.req.as_request(), io).await })
    }
}

fn downcast_erased_state<T: Send + Sync + 'static>(state: &ScanState) -> VortexResult<&T> {
    state.downcast_ref::<T>().ok_or_else(|| {
        vortex_err!(
            "scan plan state type mismatch: expected {}",
            std::any::type_name::<T>()
        )
    })
}

/// Recover a plan's concrete file/query global state from its erased form.
pub(crate) fn downcast_state<T: ScanPlan>(state: &ScanState) -> VortexResult<&T::State> {
    state.downcast_ref::<T::State>().ok_or_else(|| {
        vortex_err!(
            "scan plan state type mismatch: expected {}",
            std::any::type_name::<T::State>()
        )
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::dtype::Nullability;
    use vortex_buffer::ByteBuffer;

    use super::*;

    struct TestSegments;

    impl SegmentSource for TestSegments {
        fn request(&self, _id: crate::segments::SegmentId) -> crate::segments::SegmentFuture {
            Box::pin(async { Ok(BufferHandle::new_host(ByteBuffer::from(Vec::<u8>::new()))) })
        }
    }

    struct TestStatsNode;

    impl ScanPlan for TestStatsNode {
        type State = ();

        fn init_state(&self, _cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
            Ok(())
        }

        fn prepare_read(
            self: Arc<Self>,
            _cx: &mut PrepareCtx,
        ) -> VortexResult<Option<PreparedReadRef>> {
            Ok(None)
        }

        fn prepare_stats(
            self: Arc<Self>,
            funcs: &[AggregateFnRef],
            _cx: &mut PrepareCtx,
        ) -> VortexResult<Option<PreparedStatsRef>> {
            Ok(Some(Arc::new(TestPreparedStats { len: funcs.len() })))
        }

        fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test_stats")
        }
    }

    struct TestPreparedStats {
        len: usize,
    }

    impl PreparedStats for TestPreparedStats {
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
        let plan_root: ScanPlanRef = Arc::new(TestStatsNode);
        let funcs = vec![Min.bind(EmptyOptions), Max.bind(EmptyOptions)];

        let plan = plan_root
            .prepare_stats(&funcs, &mut PrepareCtx::new(session.clone()))?
            .ok_or_else(|| vortex_err!("test scan plan did not return a stats plan"))?;
        let state = plan.init_state(&session)?;
        let io = FileReader::new(Arc::new(TestSegments), session);
        let stats = futures::executor::block_on(plan.stats(10..20, &io, state.as_ref()))?;

        assert_eq!(stats.len(), funcs.len());
        assert!(matches!(stats[0], Precision::Exact(_)));
        assert!(matches!(stats[1], Precision::Exact(_)));

        Ok(())
    }
}
