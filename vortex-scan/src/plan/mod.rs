// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Physical scan plans with value, proof, and mask capabilities.
//!
//! A [`ScanPlan`] is immutable physical scan structure. Layouts are one way to
//! instantiate scan plans, but the runtime traits in this module are not tied to
//! serialized layouts. Engines work through [`ScanPlanRef`] trait objects:
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
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::StructArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::get_item;
use vortex_array::expr::is_root;
use vortex_array::expr::root;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::literal::Literal;
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
pub type ScanPlanRef = Arc<dyn ScanPlan>;

/// A reference-counted, type-erased prepared evidence handle.
pub type PreparedEvidenceRef = Arc<dyn PreparedEvidence>;

/// A reference-counted, type-erased prepared read handle.
pub type PreparedReadRef = Arc<dyn PreparedRead>;

/// A reference-counted, type-erased prepared split handle.
pub type PreparedSplitRef = Arc<dyn PreparedSplit>;

/// A reference-counted, type-erased prepared ungrouped aggregate handle.
pub type PreparedAggregateRef = Arc<dyn PreparedAggregate>;

/// A reference-counted, type-erased prepared metadata statistics handle.
pub type PreparedStatsRef = Arc<dyn PreparedStats>;

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
    /// Create this plan's per-file/query state.
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef>;

    /// Try to push `expr` into this plan's row domain. The returned plan's
    /// root value is exactly `expr` in the input row domain.
    ///
    /// Implementations that do not specialize expression pushdown should call
    /// [`default_try_push_expr`].
    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>>;

    /// Prepare value reads for this plan's root value.
    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>>;

    /// Prepare natural row splits for this plan's root value.
    ///
    /// The default converts this plan's cheap split hints into an executable handle. Plans can
    /// override this when split discovery needs request-specific state, I/O, or cost estimates.
    fn prepare_splits(
        self: Arc<Self>,
        _cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedSplitRef>> {
        Ok(self
            .split_hints()
            .map(|hints| Arc::new(HintPreparedSplit::new(hints.to_vec())) as PreparedSplitRef))
    }

    /// Prepare predicate evidence for this plan's root boolean value.
    ///
    /// Preparation performs no IO and returns a direct executable handle. The
    /// handle may precompute expression rewrites or accepted predicate
    /// fragments, but runtime state remains in the plan's erased scan state.
    fn prepare_evidence(
        self: Arc<Self>,
        _cx: &mut PrepareCtx,
    ) -> VortexResult<Vec<PreparedEvidenceRef>> {
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
    ) -> VortexResult<Option<PreparedAggregateRef>> {
        Ok(None)
    }

    /// Prepare metadata statistics for a field path rooted at this plan's root value.
    ///
    /// The root path means statistics for this plan's root value. Non-root field paths default to
    /// expression pushdown followed by preparing stats for the pushed root value. The returned
    /// handle answers the requested aggregate functions positionally over runtime row ranges using
    /// metadata only. `None` means this plan cannot answer these functions from metadata.
    fn prepare_field_stats(
        self: Arc<Self>,
        field_path: &FieldPath,
        funcs: &[AggregateFnRef],
        cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedStatsRef>> {
        if field_path.is_root() {
            return Ok(None);
        }
        let Some(expr) = field_path_expr(field_path) else {
            return Ok(None);
        };
        let Some(pushed) =
            Arc::clone(&self).try_push_expr(&expr, &mut PushCtx::new(cx.session().clone()))?
        else {
            return Ok(None);
        };
        pushed.prepare_field_stats(&FieldPath::root(), funcs, cx)
    }

    /// Preferred morsel boundaries (chunk edges), for alignment hints.
    fn split_hints(&self) -> Option<&[u64]> {
        None
    }

    /// Rows below `frontier` will not be read again this query: drop
    /// per-file/query state retained solely for them. Releasing must be
    /// an optimization only; the default keeps everything.
    fn release(&self, _frontier: u64, _state: &ScanState) -> VortexResult<()> {
        Ok(())
    }

    /// Compact reader-chain description for plan display, e.g.
    /// `"zoned:chunked(8)"`.
    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

/// Default expression pushdown for plans that only know how to read their root value.
pub fn default_try_push_expr(
    plan: ScanPlanRef,
    expr: &Expression,
) -> VortexResult<Option<ScanPlanRef>> {
    if is_root(expr) {
        Ok(Some(plan))
    } else {
        Ok(Some(Arc::new(ApplyScanPlan::new(plan, expr.clone()))))
    }
}

/// Return a scan plan for a scalar literal expression.
pub fn literal_scan_plan(expr: &Expression, row_count: u64) -> Option<ScanPlanRef> {
    let scalar = expr.as_opt::<Literal>()?;
    Some(Arc::new(LiteralScanPlan::new(scalar.clone(), row_count)) as ScanPlanRef)
}

fn field_path_expr(field_path: &FieldPath) -> Option<Expression> {
    let mut expr = root();
    for field in field_path.parts() {
        let Field::Name(name) = field else {
            return None;
        };
        expr = get_item(name.clone(), expr);
    }
    Some(expr)
}

/// Virtual plan that reads a scalar literal in any row domain.
pub struct LiteralScanPlan {
    scalar: Scalar,
    row_count: u64,
}

impl LiteralScanPlan {
    /// Create a plan that produces `scalar` for every selected row.
    pub fn new(scalar: Scalar, row_count: u64) -> Self {
        Self { scalar, row_count }
    }
}

struct LiteralPreparedRead {
    scalar: Scalar,
    row_count: u64,
}

struct LiteralPreparedStats {
    scalar: Scalar,
    row_count: u64,
    funcs: Vec<AggregateFnRef>,
}

impl ScanPlan for LiteralScanPlan {
    fn init_state(&self, _cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(()))
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        _cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
        if let Some(literal) = literal_scan_plan(expr, self.row_count) {
            return Ok(Some(literal));
        }
        default_try_push_expr(self, expr)
    }

    fn prepare_read(
        self: Arc<Self>,
        _cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedReadRef>> {
        Ok(Some(Arc::new(LiteralPreparedRead {
            scalar: self.scalar.clone(),
            row_count: self.row_count,
        })))
    }

    fn prepare_field_stats(
        self: Arc<Self>,
        field_path: &FieldPath,
        funcs: &[AggregateFnRef],
        _cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedStatsRef>> {
        if !field_path.is_root() {
            return Ok(None);
        }
        Ok(Some(Arc::new(LiteralPreparedStats {
            scalar: self.scalar.clone(),
            row_count: self.row_count,
            funcs: funcs.to_vec(),
        })))
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "literal({}, rows={})", self.scalar, self.row_count)
    }
}

impl PreparedRead for LiteralPreparedRead {
    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        _io: &'a FileReader,
        _local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        Box::pin(async move {
            check_scan_range(&range, self.row_count)?;
            Ok(ConstantArray::new(self.scalar.clone(), rows.selection.true_count()).into_array())
        })
    }

    fn segment_requests(
        &self,
        _range: Range<u64>,
        _rows: RowScope<'_>,
        _cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        Ok(SegmentRequests::none())
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "literal")
    }
}

impl PreparedStats for LiteralPreparedStats {
    fn init_state(&self, _ctx: &VortexSession) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(()))
    }

    fn stats<'a>(
        &'a self,
        range: Range<u64>,
        _io: &'a FileReader,
        _state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Vec<Precision<Scalar>>>> {
        Box::pin(async move {
            check_scan_range(&range, self.row_count)?;
            self.funcs
                .iter()
                .map(|func| self.stat_for_func(func, range.end - range.start))
                .collect()
        })
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "literal_stats")
    }
}

impl LiteralPreparedStats {
    fn stat_for_func(&self, func: &AggregateFnRef, len: u64) -> VortexResult<Precision<Scalar>> {
        let Some(stat) = Stat::from_aggregate_fn(func) else {
            return Ok(Precision::Absent);
        };
        let Some(dtype) = func.return_dtype(self.scalar.dtype()) else {
            return Ok(Precision::Absent);
        };
        let value = match stat {
            Stat::Min | Stat::Max => {
                if len == 0 {
                    return Ok(Precision::Absent);
                }
                if self.scalar.value().is_some() {
                    self.scalar.cast(&dtype)?
                } else if dtype.is_nullable() {
                    Scalar::null(dtype)
                } else {
                    return Ok(Precision::Absent);
                }
            }
            Stat::NullCount => Scalar::primitive(
                if self.scalar.value().is_none() {
                    len
                } else {
                    0
                },
                Nullability::NonNullable,
            ),
            _ => return Ok(Precision::Absent),
        };
        Ok(Precision::exact(value))
    }
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

fn check_scan_range(range: &Range<u64>, row_count: u64) -> VortexResult<()> {
    if range.start > range.end || range.end > row_count {
        vortex_bail!(
            "scan row range {:?} is out of bounds for row count {}",
            range,
            row_count
        );
    }
    range_len(range).map(|_| ())
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
    /// Create this prepared split's per-query state.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef>;

    /// Return natural row ranges inside `range`.
    fn splits<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Vec<Range<u64>>>>;

    /// Compact description for plan display.
    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "splits")
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
    fn init_state(&self, _ctx: &VortexSession) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(()))
    }

    fn splits<'a>(
        &'a self,
        range: Range<u64>,
        _io: &'a FileReader,
        _state: &'a ScanState,
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
    /// Create this prepared aggregate's per-query state.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef>;

    /// Answer ungrouped aggregates over every row of `range`.
    ///
    /// Returns one [`AggregateAnswer`] per prepared function. `None` means
    /// this prepared aggregate cannot answer any function for this range and the caller
    /// should read and accumulate the range normally.
    fn aggregate_partial<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Option<Vec<AggregateAnswer>>>>;

    /// Compact description for plan display.
    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "aggregate")
    }
}

/// Prepared metadata statistics for one pushed expression.
pub trait PreparedStats: 'static + Send + Sync {
    /// Create this prepared statistics handle's per-query state.
    fn init_state(&self, ctx: &VortexSession) -> VortexResult<ScanStateRef>;

    /// Answer aggregate-function statistics over every row of `range`.
    ///
    /// The returned vector is positional against the functions passed to
    /// [`ScanPlan::prepare_field_stats`]. Each element is exact, inexact, or absent for the
    /// requested aggregate function over `range`. Implementations must not read row values merely
    /// to improve an estimate.
    fn stats<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a FileReader,
        state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Vec<Precision<Scalar>>>>;

    /// Compact description for plan display.
    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stats")
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
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
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
        Ok(Arc::new(StructValueState { fields, validity }))
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        _cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
        default_try_push_expr(self, expr)
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

    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()> {
        let state = downcast_state::<StructValueState>(state)?;
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
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        cx.init_plan(&self.input)
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        _cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
        default_try_push_expr(self, expr)
    }

    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>> {
        let input = Arc::clone(&self.input)
            .prepare_read(cx)?
            .ok_or_else(|| vortex_err!("apply input did not produce a prepared read"))?;
        Ok(Some(Arc::new(ApplyPreparedRead { plan: self, input })))
    }

    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()> {
        let state = downcast_state::<ScanStateRef>(state)?;
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
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(MaskState {
            input: cx.init_plan(&self.input)?,
            validity: cx.init_plan(&self.validity)?,
        }))
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        _cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
        default_try_push_expr(self, expr)
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

    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()> {
        let state = downcast_state::<MaskState>(state)?;
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

/// Recover concrete state from its erased scan-state form.
pub fn downcast_state<T: Send + Sync + 'static>(state: &ScanState) -> VortexResult<&T> {
    state.downcast_ref::<T>().ok_or_else(|| {
        vortex_err!(
            "scan plan state type mismatch: expected {}",
            std::any::type_name::<T>()
        )
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::arrays::Constant;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::lit;
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
        fn init_state(&self, _cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
            Ok(Arc::new(()))
        }

        fn try_push_expr(
            self: Arc<Self>,
            expr: &Expression,
            _cx: &mut PushCtx,
        ) -> VortexResult<Option<ScanPlanRef>> {
            if let Some(literal) = literal_scan_plan(expr, 20) {
                return Ok(Some(literal));
            }
            default_try_push_expr(self, expr)
        }

        fn prepare_read(
            self: Arc<Self>,
            _cx: &mut PrepareCtx,
        ) -> VortexResult<Option<PreparedReadRef>> {
            Ok(None)
        }

        fn prepare_field_stats(
            self: Arc<Self>,
            field_path: &FieldPath,
            funcs: &[AggregateFnRef],
            _cx: &mut PrepareCtx,
        ) -> VortexResult<Option<PreparedStatsRef>> {
            if !field_path.is_root() {
                return Ok(None);
            }
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
        fn init_state(&self, _ctx: &VortexSession) -> VortexResult<ScanStateRef> {
            Ok(Arc::new(()))
        }

        fn stats<'a>(
            &'a self,
            range: Range<u64>,
            _io: &'a FileReader,
            _state: &'a ScanState,
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
        let funcs = vec![
            Min.bind(NumericalAggregateOpts::default()),
            Max.bind(NumericalAggregateOpts::default()),
        ];

        let plan = plan_root
            .prepare_field_stats(
                &FieldPath::root(),
                &funcs,
                &mut PrepareCtx::new(session.clone()),
            )?
            .ok_or_else(|| vortex_err!("test scan plan did not return a stats plan"))?;
        let state = plan.init_state(&session)?;
        let io = FileReader::new(Arc::new(TestSegments), session);
        let stats = futures::executor::block_on(plan.stats(10..20, &io, state.as_ref()))?;

        assert_eq!(stats.len(), funcs.len());
        assert!(matches!(stats[0], Precision::Exact(_)));
        assert!(matches!(stats[1], Precision::Exact(_)));

        Ok(())
    }

    #[test]
    fn literal_pushdown_prepares_without_input_read() -> VortexResult<()> {
        let session = VortexSession::empty();
        let plan_root: ScanPlanRef = Arc::new(TestStatsNode);
        let literal = lit(42i32);

        let plan = plan_root
            .try_push_expr(&literal, &mut PushCtx::new(session.clone()))?
            .ok_or_else(|| vortex_err!("literal expression was not pushed"))?;
        let read = plan
            .prepare_read(&mut PrepareCtx::new(session.clone()))?
            .ok_or_else(|| vortex_err!("literal scan plan did not return a prepared read"))?;
        let io = FileReader::new(Arc::new(TestSegments), session);
        let array = futures::executor::block_on(read_dense(&read, 10..15, &io))?;
        let constant = array
            .as_opt::<Constant>()
            .ok_or_else(|| vortex_err!("literal read did not produce a constant array"))?;

        assert_eq!(array.len(), 5);
        assert_eq!(constant.scalar(), &Scalar::from(42i32));

        Ok(())
    }
}
