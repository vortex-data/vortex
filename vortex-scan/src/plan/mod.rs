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

pub mod data_source;
pub mod evidence;
pub mod request;

use std::any::TypeId;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

pub use data_source::ScanPlanDataSource;
pub use data_source::ScanPlanFactory;
pub use data_source::scan_plan_projected_splits;
pub use data_source::scan_plan_split_ranges;
pub use data_source::scan_plan_statistics;
pub use data_source::scan_plan_statistics_many;
pub use data_source::scan_plan_stream;
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
use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
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
use crate::read::ReadResults;
use crate::read::ScanIoPhase;
use crate::read::ScanRead;

/// Execution context for prepared scan tasks.
#[derive(Clone)]
pub struct ReadContext {
    session: VortexSession,
}

impl ReadContext {
    /// Create a read context from a session.
    pub fn new(session: VortexSession) -> Self {
        Self { session }
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
    /// Logical dtype produced by this plan's root value.
    fn dtype(&self) -> &DType;

    /// Number of rows in this plan's row domain.
    fn row_count(&self) -> u64;

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
        Ok(Some(Arc::new(ApplyScanPlan::try_new(plan, expr.clone())?)))
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

struct LiteralReadTask {
    scalar: Scalar,
    row_count: u64,
    range: Range<u64>,
    len: usize,
}

impl ScanPlan for LiteralScanPlan {
    fn dtype(&self) -> &DType {
        self.scalar.dtype()
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

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
    fn create_task(
        self: Arc<Self>,
        range: Range<u64>,
        rows: OwnedRowScope,
        _phase: ScanIoPhase,
    ) -> VortexResult<Box<dyn ReadTask>> {
        check_scan_range(&range, self.row_count)?;
        Ok(Box::new(LiteralReadTask {
            scalar: self.scalar.clone(),
            row_count: self.row_count,
            range,
            len: rows.selection.true_count(),
        }))
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "literal")
    }
}

impl ReadTask for LiteralReadTask {
    fn into_step(self: Box<Self>) -> VortexResult<ReadStep> {
        Ok(ReadStep::new(Vec::new(), Vec::new(), move |_, _, _| {
            check_scan_range(&self.range, self.row_count)?;
            Ok(ReadTaskOutput::Ready(
                ConstantArray::new(self.scalar.clone(), self.len).into_array(),
            ))
        }))
    }
}

impl PreparedStats for LiteralPreparedStats {
    fn init_state(&self, _ctx: &VortexSession) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(()))
    }

    fn stats<'a>(
        &'a self,
        range: Range<u64>,
        _io: &'a ReadContext,
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
/// prepared file scan; each morsel execution is represented as a [`ReadTask`].
pub trait PreparedRead: 'static + Send + Sync {
    /// Create a morsel-level read task for this prepared read.
    fn create_task(
        self: Arc<Self>,
        range: Range<u64>,
        rows: OwnedRowScope,
        phase: ScanIoPhase,
    ) -> VortexResult<Box<dyn ReadTask>>;

    /// Release state behind the completed-row frontier.
    fn release(&self, _frontier: u64) -> VortexResult<()> {
        Ok(())
    }

    /// Compact description for plan display.
    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "read")
    }
}

/// Result of executing a morsel-level read task continuation.
pub enum ReadTaskOutput {
    /// The task produced its final array.
    Ready(ArrayRef),
    /// The task needs another scheduler-admitted read step.
    Continue(Box<dyn ReadTask>),
}

/// Continuation called after a read step's required reads have resolved.
pub trait ReadContinuation: Send {
    /// Execute the continuation.
    fn run(
        self: Box<Self>,
        io: &ReadContext,
        local: &mut ExecutionCtx,
        results: ReadResults,
    ) -> VortexResult<ReadTaskOutput>;
}

impl<F> ReadContinuation for F
where
    F: FnOnce(&ReadContext, &mut ExecutionCtx, ReadResults) -> VortexResult<ReadTaskOutput> + Send,
{
    fn run(
        self: Box<Self>,
        io: &ReadContext,
        local: &mut ExecutionCtx,
        results: ReadResults,
    ) -> VortexResult<ReadTaskOutput> {
        self(io, local, results)
    }
}

/// One scheduler-visible step of a layout read task.
pub struct ReadStep {
    /// Reads that must resolve before the continuation runs.
    pub required_reads: Vec<ScanRead>,
    /// Reads that may be fetched speculatively while this step is queued.
    pub prefetch_reads: Vec<ScanRead>,
    /// Continuation to execute after required reads resolve.
    pub continuation: Box<dyn ReadContinuation>,
}

impl ReadStep {
    /// Create a read step.
    pub fn new(
        required_reads: Vec<ScanRead>,
        prefetch_reads: Vec<ScanRead>,
        continuation: impl FnOnce(
            &ReadContext,
            &mut ExecutionCtx,
            ReadResults,
        ) -> VortexResult<ReadTaskOutput>
        + Send
        + 'static,
    ) -> Self {
        Self {
            required_reads,
            prefetch_reads,
            continuation: Box::new(continuation),
        }
    }
}

/// A morsel-level read task.
pub trait ReadTask: Send {
    /// Convert this task into its next scheduler-visible step.
    fn into_step(self: Box<Self>) -> VortexResult<ReadStep>;
}

enum StructReadPart {
    Ready(ArrayRef),
    Pending(Box<dyn ReadTask>),
}

struct StructReadTask {
    names: FieldNames,
    len: usize,
    fields: Vec<StructReadPart>,
    validity: Option<StructReadPart>,
}

impl ReadTask for StructReadTask {
    fn into_step(self: Box<Self>) -> VortexResult<ReadStep> {
        let Self {
            names,
            len,
            fields,
            validity,
        } = *self;
        let mut field_steps = Vec::with_capacity(fields.len());
        let mut step_fields = Vec::with_capacity(fields.len());
        let mut required_reads = Vec::new();
        let mut prefetch_reads = Vec::new();
        for field in fields {
            match field {
                StructReadPart::Ready(array) => step_fields.push(StructReadPart::Ready(array)),
                StructReadPart::Pending(task) => {
                    let step = task.into_step()?;
                    required_reads.extend(step.required_reads);
                    prefetch_reads.extend(step.prefetch_reads);
                    field_steps.push((step_fields.len(), step.continuation));
                    step_fields.push(StructReadPart::Pending(Box::new(DeferredReadTask)));
                }
            }
        }
        let (validity_step, step_validity) = match validity {
            Some(StructReadPart::Ready(array)) => (None, Some(StructReadPart::Ready(array))),
            Some(StructReadPart::Pending(task)) => {
                let step = task.into_step()?;
                required_reads.extend(step.required_reads);
                prefetch_reads.extend(step.prefetch_reads);
                (
                    Some(step.continuation),
                    Some(StructReadPart::Pending(Box::new(DeferredReadTask))),
                )
            }
            None => (None, None),
        };
        Ok(ReadStep::new(
            required_reads,
            prefetch_reads,
            move |io, local, results| {
                let session = local.session().clone();
                let mut fields = step_fields;
                let mut pending = false;
                for (idx, continuation) in field_steps {
                    let mut child_ctx = session.create_execution_ctx();
                    match continuation.run(io, &mut child_ctx, results.clone())? {
                        ReadTaskOutput::Ready(array) => fields[idx] = StructReadPart::Ready(array),
                        ReadTaskOutput::Continue(task) => {
                            fields[idx] = StructReadPart::Pending(task);
                            pending = true;
                        }
                    }
                }
                let mut next_validity = step_validity;
                let validity = match (next_validity, validity_step) {
                    (Some(StructReadPart::Ready(array)), _) => {
                        next_validity = Some(StructReadPart::Ready(array.clone()));
                        Validity::Array(array)
                    }
                    (Some(StructReadPart::Pending(_)), Some(continuation)) => {
                        match continuation.run(io, local, results)? {
                            ReadTaskOutput::Ready(array) => {
                                next_validity = Some(StructReadPart::Ready(array.clone()));
                                Validity::Array(array)
                            }
                            ReadTaskOutput::Continue(task) => {
                                next_validity = Some(StructReadPart::Pending(task));
                                pending = true;
                                Validity::NonNullable
                            }
                        }
                    }
                    (None, _) => {
                        next_validity = None;
                        Validity::NonNullable
                    }
                    (Some(StructReadPart::Pending(_)), None) => {
                        vortex_bail!("struct validity continuation missing")
                    }
                };
                if pending {
                    return Ok(ReadTaskOutput::Continue(Box::new(StructReadTask {
                        names,
                        len,
                        fields,
                        validity: next_validity,
                    })));
                }
                let arrays = fields
                    .into_iter()
                    .map(|field| match field {
                        StructReadPart::Ready(array) => Ok(array),
                        StructReadPart::Pending(_) => {
                            vortex_bail!("struct field continuation missing")
                        }
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
                Ok(ReadTaskOutput::Ready(
                    StructArray::try_new(names, arrays, len, validity)?.into_array(),
                ))
            },
        ))
    }
}

/// Placeholder read task for recursive plan continuations that replace child tasks later.
///
/// This task is never valid to execute directly. It exists so composite plan implementations can
/// build a self-referential continuation state before the next child task is available.
pub struct DeferredReadTask;

impl ReadTask for DeferredReadTask {
    fn into_step(self: Box<Self>) -> VortexResult<ReadStep> {
        vortex_bail!("deferred read task should be replaced before stepping")
    }
}

struct ApplyReadTask {
    expr: Expression,
    input: Box<dyn ReadTask>,
}

impl ReadTask for ApplyReadTask {
    fn into_step(self: Box<Self>) -> VortexResult<ReadStep> {
        let Self { expr, input } = *self;
        let step = input.into_step()?;
        Ok(ReadStep::new(
            step.required_reads,
            step.prefetch_reads,
            move |io, local, results| match step.continuation.run(io, local, results)? {
                ReadTaskOutput::Ready(input) => Ok(ReadTaskOutput::Ready(
                    input.apply(&expr)?.execute::<ArrayRef>(local)?,
                )),
                ReadTaskOutput::Continue(input) => {
                    Ok(ReadTaskOutput::Continue(Box::new(ApplyReadTask {
                        expr,
                        input,
                    })))
                }
            },
        ))
    }
}

struct MaskReadTask {
    input: Box<dyn ReadTask>,
    validity: Box<dyn ReadTask>,
}

impl ReadTask for MaskReadTask {
    fn into_step(self: Box<Self>) -> VortexResult<ReadStep> {
        let Self { input, validity } = *self;
        let input_step = input.into_step()?;
        let validity_step = validity.into_step()?;
        let mut required_reads = input_step.required_reads;
        required_reads.extend(validity_step.required_reads);
        let mut prefetch_reads = input_step.prefetch_reads;
        prefetch_reads.extend(validity_step.prefetch_reads);
        Ok(ReadStep::new(
            required_reads,
            prefetch_reads,
            move |io, local, results| {
                let input = match input_step.continuation.run(io, local, results.clone())? {
                    ReadTaskOutput::Ready(input) => input,
                    ReadTaskOutput::Continue(input) => {
                        return Ok(ReadTaskOutput::Continue(Box::new(MaskReadTask {
                            input,
                            validity: Box::new(StepReadTask::new(validity_step.continuation)),
                        })));
                    }
                };
                let validity = match validity_step.continuation.run(io, local, results)? {
                    ReadTaskOutput::Ready(validity) => validity,
                    ReadTaskOutput::Continue(validity) => {
                        return Ok(ReadTaskOutput::Continue(Box::new(MaskReadTask {
                            input: Box::new(ReadyReadTask(input)),
                            validity,
                        })));
                    }
                };
                Ok(ReadTaskOutput::Ready(
                    input.mask(validity)?.execute::<ArrayRef>(local)?,
                ))
            },
        ))
    }
}

struct ReadyReadTask(ArrayRef);

impl ReadTask for ReadyReadTask {
    fn into_step(self: Box<Self>) -> VortexResult<ReadStep> {
        Ok(ReadStep::new(Vec::new(), Vec::new(), move |_, _, _| {
            Ok(ReadTaskOutput::Ready(self.0))
        }))
    }
}

struct StepReadTask {
    continuation: Box<dyn ReadContinuation>,
}

impl StepReadTask {
    fn new(continuation: Box<dyn ReadContinuation>) -> Self {
        Self { continuation }
    }
}

impl ReadTask for StepReadTask {
    fn into_step(self: Box<Self>) -> VortexResult<ReadStep> {
        Ok(ReadStep {
            required_reads: Vec::new(),
            prefetch_reads: Vec::new(),
            continuation: self.continuation,
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
        io: &'a ReadContext,
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
        _io: &'a ReadContext,
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
        io: &'a ReadContext,
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
        io: &'a ReadContext,
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
    dtype: DType,
    row_count: u64,
    split_hints: OnceLock<Option<Vec<u64>>>,
}

impl StructValueScanPlan {
    /// Create a virtual plan that assembles a struct from child field plans.
    pub fn try_new(
        names: FieldNames,
        fields: Vec<ScanPlanRef>,
        validity: Option<ScanPlanRef>,
        row_count: u64,
    ) -> VortexResult<Self> {
        if names.len() != fields.len() {
            vortex_bail!(
                "struct scan plan has {} names for {} fields",
                names.len(),
                fields.len()
            );
        }
        for field in &fields {
            if field.row_count() != row_count {
                vortex_bail!(
                    "struct field row count {} does not match row domain {}",
                    field.row_count(),
                    row_count
                );
            }
        }
        if let Some(validity) = &validity
            && validity.row_count() != row_count
        {
            vortex_bail!(
                "struct validity row count {} does not match row domain {}",
                validity.row_count(),
                row_count
            );
        }
        let nullability = if validity.is_some() {
            Nullability::Nullable
        } else {
            Nullability::NonNullable
        };
        let dtypes = fields
            .iter()
            .map(|field| field.dtype().clone())
            .collect::<Vec<_>>();
        let dtype = DType::Struct(StructFields::new(names.clone(), dtypes), nullability);
        Ok(Self {
            names,
            fields,
            validity,
            dtype,
            row_count,
            split_hints: OnceLock::new(),
        })
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
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

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
    fn create_task(
        self: Arc<Self>,
        range: Range<u64>,
        rows: OwnedRowScope,
        phase: ScanIoPhase,
    ) -> VortexResult<Box<dyn ReadTask>> {
        let mut fields = Vec::with_capacity(self.fields.len());
        for field in &self.fields {
            fields.push(StructReadPart::Pending(Arc::clone(field).create_task(
                range.clone(),
                rows.clone(),
                phase,
            )?));
        }
        let validity = self
            .validity
            .as_ref()
            .map(|validity| {
                Arc::clone(validity)
                    .create_task(range.clone(), rows.clone(), phase)
                    .map(StructReadPart::Pending)
            })
            .transpose()?;
        Ok(Box::new(StructReadTask {
            names: self.plan.names.clone(),
            len: rows.selection.true_count(),
            fields,
            validity,
        }))
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
    dtype: DType,
}

impl ApplyScanPlan {
    /// Create a virtual plan that applies `expr` to `input`.
    pub fn try_new(input: ScanPlanRef, expr: Expression) -> VortexResult<Self> {
        let dtype = expr.return_dtype(input.dtype())?;
        Ok(Self { input, expr, dtype })
    }
}

struct ApplyPreparedRead {
    plan: Arc<ApplyScanPlan>,
    input: PreparedReadRef,
}

impl ScanPlan for ApplyScanPlan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.input.row_count()
    }

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
    fn create_task(
        self: Arc<Self>,
        range: Range<u64>,
        rows: OwnedRowScope,
        phase: ScanIoPhase,
    ) -> VortexResult<Box<dyn ReadTask>> {
        let input = Arc::clone(&self.input).create_task(range, rows, phase)?;
        Ok(Box::new(ApplyReadTask {
            expr: self.plan.expr.clone(),
            input,
        }))
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
    dtype: DType,
}

impl MaskScanPlan {
    /// Create a plan that masks `input` with a parent struct's `validity`.
    ///
    /// `validity` must read a non-nullable boolean array in the same row domain
    /// as `input` (the struct layout's validity child).
    pub fn new(input: ScanPlanRef, validity: ScanPlanRef) -> Self {
        let dtype = input.dtype().as_nullable();
        Self {
            input,
            validity,
            dtype,
        }
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
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.input.row_count()
    }

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
    fn create_task(
        self: Arc<Self>,
        range: Range<u64>,
        rows: OwnedRowScope,
        phase: ScanIoPhase,
    ) -> VortexResult<Box<dyn ReadTask>> {
        let input = Arc::clone(&self.input).create_task(range.clone(), rows.clone(), phase)?;
        let validity = Arc::clone(&self.validity).create_task(range, rows, phase)?;
        Ok(Box::new(MaskReadTask { input, validity }))
    }

    fn release(&self, frontier: u64) -> VortexResult<()> {
        self.input.release(frontier)?;
        self.validity.release(frontier)
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ScanPlan::fmt_chain(self.plan.as_ref(), f)
    }
}

/// Static cost class for a predicate-evidence provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceCost {
    /// Cheap metadata evidence, such as zone maps.
    Metadata,
    /// Index-like evidence that may touch more state than metadata but avoids row decoding.
    Index,
    /// Evidence that performs row-level compute.
    Compute,
    /// Evidence with unknown cost.
    Unknown,
}

/// Natural execution scope for a predicate-evidence provider.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EvidenceScope {
    /// Evidence is produced independently for each morsel range.
    #[default]
    Morsel,
    /// Evidence is produced over the scan row domain and consumed by many morsels.
    Scan,
}

impl EvidenceCost {
    /// Convert this cost class and estimated read bytes into a scheduling priority.
    ///
    /// Lower priorities run first. The returned value is intentionally coarse; runtime selectivity
    /// feedback should dominate once a lane has observations.
    pub fn priority(self, read_bytes: u64, dynamic_recheck: bool) -> u64 {
        let base = match self {
            Self::Metadata => 1_000,
            Self::Index => 10_000,
            Self::Unknown => 100_000,
            Self::Compute => 1_000_000,
        };
        let read_penalty = read_bytes / 1024;
        let priority = base + read_penalty;
        if dynamic_recheck {
            priority / 2
        } else {
            priority
        }
    }
}

/// Prepared predicate evidence for one predicate expression.
pub trait PreparedEvidence: 'static + Send + Sync {
    /// Produce evidence for the prepared predicate over `req.range`.
    fn evidence<'a>(
        &'a self,
        req: &'a EvidenceRequest<'a>,
        io: &'a ReadContext,
        results: ReadResults,
    ) -> VortexResult<Vec<EvidenceFragment>>;

    /// Whether this handle is cheap enough to re-run immediately before a
    /// projection read when a dynamic predicate boundary changes while
    /// the morsel is in flight.
    fn recheck_before_projection(&self) -> bool {
        false
    }

    /// Static cost class used by the scan scheduler when ordering evidence tasks.
    fn cost(&self, _req: &EvidenceRequest<'_>) -> EvidenceCost {
        EvidenceCost::Unknown
    }

    /// Return the natural execution scope for this evidence provider.
    fn scope(&self) -> EvidenceScope {
        EvidenceScope::Morsel
    }

    /// Compact description for plan display.
    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "evidence")
    }

    /// Create a morsel-level evidence task for this prepared evidence handle.
    fn create_task(
        self: Arc<Self>,
        req: OwnedEvidenceRequest,
        phase: ScanIoPhase,
    ) -> VortexResult<Box<dyn EvidenceTask>>;
}

/// A morsel-level evidence task.
pub trait EvidenceTask: Send {
    /// Convert this task into its scheduler-visible step.
    fn into_step(self: Box<Self>) -> VortexResult<EvidenceStep>;
}

/// Continuation called after an evidence step's required reads have resolved.
pub trait EvidenceContinuation: Send {
    /// Execute the continuation.
    fn run(
        self: Box<Self>,
        io: &ReadContext,
        results: ReadResults,
    ) -> VortexResult<Vec<EvidenceFragment>>;
}

impl<F> EvidenceContinuation for F
where
    F: FnOnce(&ReadContext, ReadResults) -> VortexResult<Vec<EvidenceFragment>> + Send,
{
    fn run(
        self: Box<Self>,
        io: &ReadContext,
        results: ReadResults,
    ) -> VortexResult<Vec<EvidenceFragment>> {
        self(io, results)
    }
}

/// One scheduler-visible step of an evidence task.
pub struct EvidenceStep {
    /// Reads that must resolve before the continuation runs.
    pub required_reads: Vec<ScanRead>,
    /// Reads that may be fetched speculatively while this step is queued.
    pub prefetch_reads: Vec<ScanRead>,
    /// Continuation to execute after required reads resolve.
    pub continuation: Box<dyn EvidenceContinuation>,
}

impl EvidenceStep {
    /// Create an evidence step.
    pub fn new(
        required_reads: Vec<ScanRead>,
        prefetch_reads: Vec<ScanRead>,
        continuation: impl FnOnce(&ReadContext, ReadResults) -> VortexResult<Vec<EvidenceFragment>>
        + Send
        + 'static,
    ) -> Self {
        Self {
            required_reads,
            prefetch_reads,
            continuation: Box::new(continuation),
        }
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
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::lit;

    use super::*;
    use crate::read::ReadStore;

    struct TestStatsNode {
        dtype: DType,
        row_count: u64,
    }

    impl ScanPlan for TestStatsNode {
        fn dtype(&self) -> &DType {
            &self.dtype
        }

        fn row_count(&self) -> u64 {
            self.row_count
        }

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
            _io: &'a ReadContext,
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
        let plan_root: ScanPlanRef = Arc::new(TestStatsNode {
            dtype: DType::Primitive(PType::I32, Nullability::NonNullable),
            row_count: 20,
        });
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
        let io = ReadContext::new(session);
        let stats = futures::executor::block_on(plan.stats(10..20, &io, state.as_ref()))?;

        assert_eq!(stats.len(), funcs.len());
        assert!(matches!(stats[0], Precision::Exact(_)));
        assert!(matches!(stats[1], Precision::Exact(_)));

        Ok(())
    }

    #[test]
    fn literal_pushdown_prepares_without_input_read() -> VortexResult<()> {
        let session = VortexSession::empty();
        let plan_root: ScanPlanRef = Arc::new(TestStatsNode {
            dtype: DType::Primitive(PType::I32, Nullability::NonNullable),
            row_count: 20,
        });
        let literal = lit(42i32);

        let plan = plan_root
            .try_push_expr(&literal, &mut PushCtx::new(session.clone()))?
            .ok_or_else(|| vortex_err!("literal expression was not pushed"))?;
        let read = plan
            .prepare_read(&mut PrepareCtx::new(session.clone()))?
            .ok_or_else(|| vortex_err!("literal scan plan did not return a prepared read"))?;
        let io = ReadContext::new(session);
        let rows = OwnedRowScope::selected(Mask::new_true(5));
        let task = read.create_task(10..15, rows, ScanIoPhase::ProjectionRead)?;
        let results = ReadResults::new(Arc::new(ReadStore::new()));
        let step = task.into_step()?;
        if !step.required_reads.is_empty() || !step.prefetch_reads.is_empty() {
            vortex_bail!("literal read unexpectedly requested reads");
        }
        let mut local = io.session().create_execution_ctx();
        let array = match step.continuation.run(&io, &mut local, results)? {
            ReadTaskOutput::Ready(array) => array,
            ReadTaskOutput::Continue(_) => vortex_bail!("literal read unexpectedly continued"),
        };
        let constant = array
            .as_opt::<Constant>()
            .ok_or_else(|| vortex_err!("literal read did not produce a constant array"))?;

        assert_eq!(array.len(), 5);
        assert_eq!(constant.scalar(), &Scalar::from(42i32));

        Ok(())
    }
}
