// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scan2 vtable support for chunked layouts.
//!
//! Chunks stay *lazy*: children are resolved from the footer and expanded
//! through their own layout scan vtables per request, never pre-planned. Chunked is
//! therefore a lazy pushdown boundary: pushed expressions are recorded
//! once, then replayed into each concrete child only when a read,
//! evidence request, or aggregate touches that chunk. This lets
//! child-local layouts such as zoned, dictionary, or index wrappers keep
//! their scan behavior without expanding every chunk up front.
//!
//! The selected read path is where chunking pays off (plan 017 SP5): a
//! chunk whose selection slice is empty is skipped outright — its node is
//! never expanded, its state never created, its segments never fetched.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;
#[cfg(debug_assertions)]
use std::sync::atomic::AtomicU64;
#[cfg(debug_assertions)]
use std::sync::atomic::Ordering;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::expr::root;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_scan::plan::AggregateAnswer;
use vortex_scan::plan::DeferredReadTask;
use vortex_scan::plan::EvidenceStep;
use vortex_scan::plan::EvidenceTask;
use vortex_scan::plan::OwnedRowScope;
use vortex_scan::plan::PrepareCtx;
use vortex_scan::plan::PreparedAggregate;
use vortex_scan::plan::PreparedAggregateRef;
use vortex_scan::plan::PreparedEvidence;
use vortex_scan::plan::PreparedEvidenceRef;
use vortex_scan::plan::PreparedRead;
use vortex_scan::plan::PreparedReadRef;
use vortex_scan::plan::PreparedStateCacheRef;
use vortex_scan::plan::PreparedStateKey;
use vortex_scan::plan::PushCtx;
use vortex_scan::plan::ReadContext;
use vortex_scan::plan::ReadStep;
use vortex_scan::plan::ReadTask;
use vortex_scan::plan::ReadTaskOutput;
use vortex_scan::plan::ScanPlan;
use vortex_scan::plan::ScanPlanRef;
use vortex_scan::plan::ScanState;
use vortex_scan::plan::ScanStateRef;
use vortex_scan::plan::StateCtx;
use vortex_scan::plan::default_try_push_expr;
use vortex_scan::plan::downcast_state;
use vortex_scan::plan::evidence::EvidenceFragment;
use vortex_scan::plan::request::EvidenceMode;
use vortex_scan::plan::request::EvidenceRequest;
use vortex_scan::plan::request::OwnedEvidenceRequest;
use vortex_scan::plan::request::ScanRequest;
use vortex_scan::read::ReadResults;
use vortex_scan::read::ScanIoPhase;
use vortex_session::VortexSession;

use crate::layout_v2::Layout;
use crate::layout_v2::LayoutRef;
use crate::layout_v2::LayoutScanPlanCtx;
use crate::layouts_v2::chunked::Chunked;

pub(crate) fn new_scan_plan(
    layout: Layout<Chunked>,
    _req: &mut ScanRequest,
    ctx: &LayoutScanPlanCtx,
) -> VortexResult<ScanPlanRef> {
    Ok(Arc::new(ChunkedScanPlan {
        layout: layout.to_layout(),
        offsets: layout.data().chunk_offsets().to_vec(),
        ctx: ctx.clone(),
        children: Mutex::new(FxHashMap::default()),
    }))
}

/// Reads a chunked layout: cumulative chunk offsets
/// (`offsets.len() == chunks + 1`), with chunk children expanded lazily
/// through their own layout vtables.
pub struct ChunkedScanPlan {
    layout: LayoutRef,
    offsets: Vec<u64>,
    ctx: LayoutScanPlanCtx,
    /// Lazily expanded chunk nodes, shared across queries.
    children: Mutex<FxHashMap<usize, ScanPlanRef>>,
}

/// Per-query states of the lazily expanded chunk nodes. Chunk states
/// behind the scan's morsel frontier are dropped by
/// [`ScanPlan::release`], so a long scan retains the working set, not
/// every chunk it touched.
#[derive(Default)]
pub struct ChunkedScanState {
    reads: Mutex<FxHashMap<usize, PreparedReadRef>>,
    child_state_caches: Mutex<FxHashMap<usize, PreparedStateCacheRef>>,
    /// Every chunk whose state was ever created (never cleared by
    /// release), for read-avoidance tests.
    #[cfg(any(test, debug_assertions))]
    created: Mutex<rustc_hash::FxHashSet<usize>>,
    /// Highest released frontier, for the debug no-read-behind check.
    #[cfg(debug_assertions)]
    released: AtomicU64,
}

/// A pushed expression over a chunked layout.
///
/// Chunk children remain lazy: this node records the expression once and
/// replays expression pushdown into each concrete child only when a read,
/// evidence request, or aggregate touches that chunk.
pub struct ChunkedExprScanPlan {
    chunked: Arc<ChunkedScanPlan>,
    expr: Expression,
    dtype: DType,
    children: Mutex<FxHashMap<usize, ScanPlanRef>>,
}

/// Per-query states of lazily pushed chunk children.
pub struct ChunkedExprScanState {
    chunked: Arc<ChunkedScanState>,
    reads: Mutex<FxHashMap<usize, PreparedReadRef>>,
    #[cfg(debug_assertions)]
    released: AtomicU64,
}

struct ChunkedPreparedEvidence {
    node: Arc<ChunkedExprScanPlan>,
    state: Arc<ChunkedEvidenceState>,
    session: VortexSession,
}

struct ChunkedEvidenceTask {
    evidence: Arc<ChunkedPreparedEvidence>,
    req: OwnedEvidenceRequest,
    phase: ScanIoPhase,
}

enum ChunkedAggregateNode {
    Root(Arc<ChunkedScanPlan>),
    Expr(Arc<ChunkedExprScanPlan>),
}

struct ChunkedPreparedAggregate {
    node: ChunkedAggregateNode,
    chunked_state: Arc<ChunkedScanState>,
    dtype: DType,
    funcs: Vec<AggregateFnRef>,
}

struct ChunkedPreparedRead {
    node: Arc<ChunkedScanPlan>,
    state: Arc<ChunkedScanState>,
}

struct ChunkedExprPreparedRead {
    node: Arc<ChunkedExprScanPlan>,
    state: Arc<ChunkedExprScanState>,
}

enum ChunkedReadPart {
    Ready(ArrayRef),
    Pending {
        expected_len: usize,
        task: Box<dyn ReadTask>,
    },
}

struct ChunkedReadTask {
    dtype: DType,
    parts: Vec<ChunkedReadPart>,
}

impl ReadTask for ChunkedReadTask {
    fn into_step(self: Box<Self>) -> VortexResult<ReadStep> {
        let Self { dtype, parts } = *self;
        let mut step_parts = Vec::with_capacity(parts.len());
        let mut continuations = Vec::new();
        let mut required_reads = Vec::new();
        let mut prefetch_reads = Vec::new();
        for part in parts {
            match part {
                ChunkedReadPart::Ready(array) => step_parts.push(ChunkedReadPart::Ready(array)),
                ChunkedReadPart::Pending { expected_len, task } => {
                    let step = task.into_step()?;
                    required_reads.extend(step.required_reads);
                    prefetch_reads.extend(step.prefetch_reads);
                    continuations.push((step_parts.len(), expected_len, step.continuation));
                    step_parts.push(ChunkedReadPart::Pending {
                        expected_len,
                        task: Box::new(DeferredReadTask),
                    });
                }
            }
        }
        Ok(ReadStep::new(
            required_reads,
            prefetch_reads,
            move |io, local, results| {
                let mut parts = step_parts;
                let mut pending = false;
                for (idx, expected_len, continuation) in continuations {
                    match continuation.run(io, local, results.clone())? {
                        ReadTaskOutput::Ready(chunk) => {
                            if chunk.len() != expected_len {
                                vortex_bail!(
                                    "scoped chunk read returned length {}, expected {}",
                                    chunk.len(),
                                    expected_len
                                );
                            }
                            parts[idx] = ChunkedReadPart::Ready(chunk);
                        }
                        ReadTaskOutput::Continue(task) => {
                            parts[idx] = ChunkedReadPart::Pending { expected_len, task };
                            pending = true;
                        }
                    }
                }
                if pending {
                    return Ok(ReadTaskOutput::Continue(Box::new(ChunkedReadTask {
                        dtype,
                        parts,
                    })));
                }
                let mut arrays = parts
                    .into_iter()
                    .map(|part| match part {
                        ChunkedReadPart::Ready(array) => Ok(array),
                        ChunkedReadPart::Pending { .. } => {
                            vortex_bail!("chunked read part remained pending after step completion")
                        }
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
                let array = match arrays.len() {
                    0 => vortex_bail!("chunked scoped read produced no parts"),
                    1 => arrays.swap_remove(0),
                    _ => ChunkedArray::try_new(arrays, dtype)?.into_array(),
                };
                Ok(ReadTaskOutput::Ready(array))
            },
        ))
    }
}

struct ChunkedEvidenceState {
    chunked: Arc<ChunkedScanState>,
    children: Mutex<FxHashMap<usize, Vec<PreparedEvidenceRef>>>,
    recheck_children: Mutex<FxHashMap<usize, Vec<PreparedEvidenceRef>>>,
}

#[derive(Default)]
struct ChunkedAggregateState {
    children: Mutex<FxHashMap<usize, Option<(PreparedAggregateRef, ScanStateRef)>>>,
}

impl ChunkedScanState {
    fn child_prepare_ctx(&self, idx: usize, session: &VortexSession) -> PrepareCtx {
        if let Some(hit) = self.child_state_caches.lock().get(&idx) {
            return PrepareCtx::with_state_cache(session.clone(), Arc::clone(hit));
        }
        let cache = Default::default();
        let mut caches = self.child_state_caches.lock();
        let cache = Arc::clone(caches.entry(idx).or_insert(cache));
        PrepareCtx::with_state_cache(session.clone(), cache)
    }

    /// The number of chunk states currently retained.
    #[allow(dead_code)]
    #[cfg(any(test, debug_assertions))]
    pub fn retained_children(&self) -> usize {
        self.reads.lock().len()
    }

    /// Whether chunk `idx` was ever read this query (release does not
    /// clear this).
    #[allow(dead_code)]
    #[cfg(any(test, debug_assertions))]
    pub fn touched(&self, idx: usize) -> bool {
        self.created.lock().contains(&idx)
    }
}

impl ChunkedEvidenceState {
    fn new(chunked: Arc<ChunkedScanState>) -> Self {
        Self {
            chunked,
            children: Mutex::new(FxHashMap::default()),
            recheck_children: Mutex::new(FxHashMap::default()),
        }
    }
}

impl ChunkedScanPlan {
    fn scan_state(&self, cx: &mut PrepareCtx) -> VortexResult<Arc<ChunkedScanState>> {
        let key =
            PreparedStateKey::new::<ChunkedScanState>(self as *const Self as *const () as usize);
        cx.shared_state(key, || Ok(ChunkedScanState::default()))
    }

    /// The scan plan for chunk `idx`, expanding it on first use. Lazy
    /// expansion is independent of pushed predicate expressions.
    fn child(&self, idx: usize) -> VortexResult<ScanPlanRef> {
        if let Some(hit) = self.children.lock().get(&idx) {
            return Ok(Arc::clone(hit));
        }
        let mut req = ScanRequest::empty();
        let plan = self.layout.child(idx)?.new_scan_plan(&mut req, &self.ctx)?;
        self.children.lock().insert(idx, Arc::clone(&plan));
        Ok(plan)
    }

    /// The planned value read for chunk `idx`, creating it on first use.
    fn child_read(
        &self,
        idx: usize,
        state: &ChunkedScanState,
        session: &VortexSession,
    ) -> VortexResult<PreparedReadRef> {
        if let Some(hit) = state.reads.lock().get(&idx) {
            return Ok(Arc::clone(hit));
        }
        let node = self.child(idx)?;
        let mut cx = state.child_prepare_ctx(idx, session);
        let read = node
            .prepare_read(&mut cx)?
            .ok_or_else(|| vortex_err!("chunked child {idx} did not produce a prepared read"))?;
        let mut reads = state.reads.lock();
        #[cfg(any(test, debug_assertions))]
        state.created.lock().insert(idx);
        Ok(Arc::clone(reads.entry(idx).or_insert(read)))
    }

    fn first_chunk(&self, start: u64) -> usize {
        self.offsets
            .partition_point(|&offset| offset <= start)
            .saturating_sub(1)
    }
}

impl ChunkedExprScanPlan {
    fn new(chunked: Arc<ChunkedScanPlan>, expr: Expression, dtype: DType) -> Self {
        Self {
            chunked,
            expr,
            dtype,
            children: Mutex::new(FxHashMap::default()),
        }
    }

    fn child(&self, idx: usize, session: &VortexSession) -> VortexResult<ScanPlanRef> {
        if let Some(hit) = self.children.lock().get(&idx) {
            return Ok(Arc::clone(hit));
        }
        let child = self.chunked.child(idx)?;
        let mut cx = PushCtx::new(session.clone());
        let pushed = child.try_push_expr(&self.expr, &mut cx)?.ok_or_else(|| {
            vortex_err!(
                "chunked child {idx} could not push expression {}",
                self.expr
            )
        })?;
        let mut children = self.children.lock();
        Ok(Arc::clone(children.entry(idx).or_insert(pushed)))
    }

    /// The planned value read for pushed chunk child `idx`.
    fn child_read(
        &self,
        idx: usize,
        state: &ChunkedExprScanState,
        session: &VortexSession,
    ) -> VortexResult<PreparedReadRef> {
        if let Some(hit) = state.reads.lock().get(&idx) {
            return Ok(Arc::clone(hit));
        }
        let node = self.child(idx, session)?;
        let mut cx = state.chunked.child_prepare_ctx(idx, session);
        let read = node.prepare_read(&mut cx)?.ok_or_else(|| {
            vortex_err!("chunked expression child {idx} did not produce a prepared read")
        })?;
        let mut reads = state.reads.lock();
        Ok(Arc::clone(reads.entry(idx).or_insert(read)))
    }
}

impl ChunkedAggregateNode {
    fn offsets(&self) -> &[u64] {
        match self {
            Self::Root(node) => &node.offsets,
            Self::Expr(node) => &node.chunked.offsets,
        }
    }

    fn first_chunk(&self, start: u64) -> usize {
        match self {
            Self::Root(node) => node.first_chunk(start),
            Self::Expr(node) => node.chunked.first_chunk(start),
        }
    }

    fn child(&self, idx: usize, io: &ReadContext) -> VortexResult<ScanPlanRef> {
        match self {
            Self::Root(node) => node.child(idx),
            Self::Expr(node) => node.child(idx, io.session()),
        }
    }
}

impl ChunkedPreparedAggregate {
    fn child_plan(
        &self,
        idx: usize,
        state: &ChunkedAggregateState,
        io: &ReadContext,
    ) -> VortexResult<Option<(PreparedAggregateRef, ScanStateRef)>> {
        if let Some(hit) = state.children.lock().get(&idx) {
            return Ok(hit.clone());
        }
        let child = self.node.child(idx, io)?;
        let mut plan_ctx = self.chunked_state.child_prepare_ctx(idx, io.session());
        let planned = match child.prepare_aggregate_partial(&self.funcs, &mut plan_ctx)? {
            Some(plan) => {
                let plan_state = plan.init_state(io.session())?;
                Some((plan, plan_state))
            }
            None => None,
        };
        let mut children = state.children.lock();
        Ok(children.entry(idx).or_insert(planned).clone())
    }
}

impl PreparedAggregate for ChunkedPreparedAggregate {
    fn init_state(&self, _ctx: &VortexSession) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(ChunkedAggregateState::default()))
    }

    fn aggregate_partial<'a>(
        &'a self,
        range: Range<u64>,
        io: &'a ReadContext,
        state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Option<Vec<AggregateAnswer>>>> {
        Box::pin(async move {
            let state = downcast_state::<ChunkedAggregateState>(state)?;
            if range.start >= range.end {
                return Ok(None);
            }
            let mut accumulators = self
                .funcs
                .iter()
                .map(|func| {
                    func.state_dtype(&self.dtype)
                        .map(|_| func.accumulator(&self.dtype))
                        .transpose()
                })
                .collect::<VortexResult<Vec<_>>>()?;
            let mut contributed = vec![false; self.funcs.len()];
            let mut covered = vec![false; self.funcs.len()];
            let mut residuals: Vec<Vec<Range<u64>>> = vec![Vec::new(); self.funcs.len()];
            let push_residual =
                |residual: &mut Vec<Range<u64>>, span: Range<u64>| match residual.last_mut() {
                    Some(last) if last.end == span.start => last.end = span.end,
                    _ => residual.push(span),
                };

            let offsets = self.node.offsets();
            let mut idx = self.node.first_chunk(range.start);
            while idx + 1 < offsets.len() && offsets[idx] < range.end {
                let chunk_start = offsets[idx];
                let chunk_end = offsets[idx + 1];
                let local = range.start.saturating_sub(chunk_start)
                    ..(range.end.min(chunk_end) - chunk_start);
                let answers = match self.child_plan(idx, state, io)? {
                    Some((plan, plan_state)) => {
                        plan.aggregate_partial(local.clone(), io, plan_state.as_ref())
                            .await?
                    }
                    None => None,
                };
                match answers {
                    Some(answers) => {
                        for (func_idx, answer) in answers.into_iter().enumerate() {
                            let has_partial = answer.partial.is_some();
                            let mut residual_rows = 0;
                            for span in answer.residual {
                                residual_rows += span.end - span.start;
                                push_residual(
                                    &mut residuals[func_idx],
                                    chunk_start + span.start..chunk_start + span.end,
                                );
                            }
                            if let Some(partial) = answer.partial {
                                let Some(Some(acc)) = accumulators.get_mut(func_idx) else {
                                    vortex_bail!("chunk answered an unsupported aggregate");
                                };
                                acc.combine_partials(partial)?;
                                contributed[func_idx] = true;
                            }
                            covered[func_idx] |=
                                has_partial || residual_rows < local.end - local.start;
                        }
                    }
                    None => {
                        for residual in residuals.iter_mut() {
                            push_residual(
                                residual,
                                chunk_start + local.start..chunk_start + local.end,
                            );
                        }
                    }
                }
                idx += 1;
            }
            if !covered.iter().any(|&covered| covered) {
                return Ok(None);
            }
            let mut answers = Vec::with_capacity(self.funcs.len());
            for ((accumulator, contributed), residual) in
                accumulators.iter_mut().zip(contributed).zip(residuals)
            {
                let partial = match accumulator {
                    Some(acc) if contributed => Some(acc.flush()?),
                    _ => None,
                };
                answers.push(AggregateAnswer { partial, residual });
            }
            Ok(Some(answers))
        })
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "chunked")
    }
}

impl ScanPlan for ChunkedScanPlan {
    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn init_state(&self, _cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(ChunkedScanState::default()))
    }

    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>> {
        let state = self.scan_state(cx)?;
        Ok(Some(Arc::new(ChunkedPreparedRead { node: self, state })))
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        _cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
        if is_root(expr) {
            return Ok(Some(self));
        }
        let dtype = expr.return_dtype(self.layout.dtype())?;
        Ok(Some(Arc::new(ChunkedExprScanPlan::new(
            self,
            expr.clone(),
            dtype,
        ))))
    }

    fn prepare_evidence(
        self: Arc<Self>,
        cx: &mut PrepareCtx,
    ) -> VortexResult<Vec<PreparedEvidenceRef>> {
        let node = Arc::new(ChunkedExprScanPlan::new(
            Arc::clone(&self),
            root(),
            self.layout.dtype().clone(),
        ));
        let chunked_state = self.scan_state(cx)?;
        let key =
            PreparedStateKey::new::<ChunkedEvidenceState>(Arc::as_ptr(&self) as *const () as usize);
        let state = cx.shared_state(key, || Ok(ChunkedEvidenceState::new(chunked_state)))?;
        Ok(vec![Arc::new(ChunkedPreparedEvidence {
            node,
            state,
            session: cx.session().clone(),
        })])
    }

    fn prepare_aggregate_partial(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedAggregateRef>> {
        if funcs.is_empty() {
            return Ok(None);
        }
        let chunked_state = self.scan_state(cx)?;
        Ok(Some(Arc::new(ChunkedPreparedAggregate {
            node: ChunkedAggregateNode::Root(Arc::clone(&self)),
            chunked_state,
            dtype: self.layout.dtype().clone(),
            funcs: funcs.to_vec(),
        })))
    }

    fn split_hints(&self) -> Option<&[u64]> {
        Some(&self.offsets)
    }

    /// Drop chunk states wholly behind the frontier and recurse into the
    /// boundary chunk so nested layouts release their own state. The
    /// expanded chunk *nodes* stay: they are shared across queries and
    /// hold no data.
    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()> {
        let state = downcast_state::<ChunkedScanState>(state)?;
        state
            .reads
            .lock()
            .retain(|&idx, _| self.offsets[idx + 1] > frontier);
        state
            .child_state_caches
            .lock()
            .retain(|&idx, _| self.offsets[idx + 1] > frontier);
        let idx = self.first_chunk(frontier);
        if idx + 1 < self.offsets.len() && self.offsets[idx] < frontier {
            let child = state.reads.lock().get(&idx).cloned();
            if let Some(child) = child {
                child.release(frontier - self.offsets[idx])?;
            }
        }
        #[cfg(debug_assertions)]
        state.released.fetch_max(frontier, Ordering::Relaxed);
        Ok(())
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "chunked({})", self.offsets.len().saturating_sub(1))
    }
}

impl PreparedRead for ChunkedPreparedRead {
    fn create_task(
        self: Arc<Self>,
        range: Range<u64>,
        rows: OwnedRowScope,
        phase: ScanIoPhase,
    ) -> VortexResult<Box<dyn ReadTask>> {
        if range.start >= range.end {
            vortex_bail!("empty chunked scoped read range");
        }
        #[cfg(debug_assertions)]
        {
            let released = self.state.released.load(Ordering::Relaxed);
            debug_assert!(
                range.start >= released,
                "chunked read {range:?} below the released frontier {released}"
            );
        }
        let range_len = usize::try_from(range.end - range.start)
            .map_err(|_| vortex_err!("read range exceeds usize"))?;
        let row_scope = rows.as_scope();
        if row_scope.selection.len() != range_len {
            vortex_bail!(
                "selection length {} does not match range length {range_len}",
                row_scope.selection.len()
            );
        }
        if row_scope.demand.len() != range_len {
            vortex_bail!(
                "demand length {} does not match range length {range_len}",
                row_scope.demand.len()
            );
        }
        if row_scope.selection.all_false() {
            return Ok(Box::new(ChunkedReadTask {
                dtype: self.node.layout.dtype().clone(),
                parts: vec![ChunkedReadPart::Ready(
                    ConstantArray::new(Scalar::default_value(self.node.layout.dtype()), 0)
                        .into_array(),
                )],
            }));
        }

        let dtype = self.node.layout.dtype().clone();
        let dense_scope = row_scope.selection.all_true() && row_scope.demand.all_true();
        let selected_scope = !dense_scope && row_scope.demands_all_selected();
        let mut parts = Vec::new();
        let mut idx = self.node.first_chunk(range.start);
        while idx + 1 < self.node.offsets.len() && self.node.offsets[idx] < range.end {
            let chunk_start = self.node.offsets[idx];
            let chunk_end = self.node.offsets[idx + 1];
            let local =
                range.start.saturating_sub(chunk_start)..(range.end.min(chunk_end) - chunk_start);
            let sel_start = usize::try_from(chunk_start.max(range.start) - range.start)
                .map_err(|_| vortex_err!("read range exceeds usize"))?;
            let sel_end = usize::try_from(chunk_end.min(range.end) - range.start)
                .map_err(|_| vortex_err!("read range exceeds usize"))?;
            let chunk_selection = row_scope.selection.slice(sel_start..sel_end);
            idx += 1;
            if chunk_selection.all_false() {
                continue;
            }
            let chunk_demand = row_scope.demand.slice(sel_start..sel_end);
            if chunk_demand.all_false() {
                parts.push(ChunkedReadPart::Ready(
                    ConstantArray::new(Scalar::default_value(&dtype), chunk_selection.true_count())
                        .into_array(),
                ));
                continue;
            }
            let chunk_idx = idx - 1;
            let read = self
                .node
                .child_read(chunk_idx, &self.state, self.node.ctx.session())?;
            let chunk_rows = if dense_scope || selected_scope {
                OwnedRowScope::selected(chunk_selection.clone())
            } else {
                OwnedRowScope::try_new(chunk_selection.clone(), chunk_demand)?
            };
            let expected_len = chunk_selection.true_count();
            parts.push(ChunkedReadPart::Pending {
                expected_len,
                task: Arc::clone(&read).create_task(local, chunk_rows, phase)?,
            });
        }
        match parts.len() {
            0 => vortex_bail!("chunked scoped read range {range:?} out of bounds"),
            _ => Ok(Box::new(ChunkedReadTask { dtype, parts })),
        }
    }

    fn release(&self, frontier: u64) -> VortexResult<()> {
        self.node.release(frontier, self.state.as_ref())
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt_chain(f)
    }
}

impl ScanPlan for ChunkedExprScanPlan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.chunked.layout.row_count()
    }

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        let _ = cx;
        Ok(Arc::new(ChunkedExprScanState {
            chunked: Arc::new(ChunkedScanState::default()),
            reads: Mutex::new(FxHashMap::default()),
            #[cfg(debug_assertions)]
            released: AtomicU64::new(0),
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
        let key =
            PreparedStateKey::new::<ChunkedExprScanState>(Arc::as_ptr(&self) as *const () as usize);
        let chunked = self.chunked.scan_state(cx)?;
        let state = cx.shared_state(key, || {
            Ok(ChunkedExprScanState {
                chunked,
                reads: Mutex::new(FxHashMap::default()),
                #[cfg(debug_assertions)]
                released: AtomicU64::new(0),
            })
        })?;
        Ok(Some(Arc::new(ChunkedExprPreparedRead {
            node: self,
            state,
        })))
    }

    fn prepare_evidence(
        self: Arc<Self>,
        cx: &mut PrepareCtx,
    ) -> VortexResult<Vec<PreparedEvidenceRef>> {
        let key =
            PreparedStateKey::new::<ChunkedEvidenceState>(Arc::as_ptr(&self) as *const () as usize);
        let chunked = self.chunked.scan_state(cx)?;
        let state = cx.shared_state(key, || Ok(ChunkedEvidenceState::new(chunked)))?;
        Ok(vec![Arc::new(ChunkedPreparedEvidence {
            node: self,
            state,
            session: cx.session().clone(),
        })])
    }

    fn prepare_aggregate_partial(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedAggregateRef>> {
        if funcs.is_empty() {
            return Ok(None);
        }
        let chunked_state = self.chunked.scan_state(cx)?;
        Ok(Some(Arc::new(ChunkedPreparedAggregate {
            node: ChunkedAggregateNode::Expr(Arc::clone(&self)),
            chunked_state,
            dtype: self.dtype.clone(),
            funcs: funcs.to_vec(),
        })))
    }

    fn split_hints(&self) -> Option<&[u64]> {
        Some(&self.chunked.offsets)
    }

    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()> {
        let state = downcast_state::<ChunkedExprScanState>(state)?;
        state
            .reads
            .lock()
            .retain(|&idx, _| self.chunked.offsets[idx + 1] > frontier);
        let idx = self.chunked.first_chunk(frontier);
        if idx + 1 < self.chunked.offsets.len() && self.chunked.offsets[idx] < frontier {
            let child = state.reads.lock().get(&idx).cloned();
            if let Some(child) = child {
                child.release(frontier - self.chunked.offsets[idx])?;
            }
        }
        #[cfg(debug_assertions)]
        state.released.fetch_max(frontier, Ordering::Relaxed);
        Ok(())
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "chunked_expr({})", self.expr)
    }
}

impl PreparedRead for ChunkedExprPreparedRead {
    fn create_task(
        self: Arc<Self>,
        range: Range<u64>,
        rows: OwnedRowScope,
        phase: ScanIoPhase,
    ) -> VortexResult<Box<dyn ReadTask>> {
        if range.start >= range.end {
            vortex_bail!("empty chunked scoped read range");
        }
        #[cfg(debug_assertions)]
        {
            let released = self.state.released.load(Ordering::Relaxed);
            debug_assert!(
                range.start >= released,
                "chunked expression read {range:?} below the released frontier {released}"
            );
        }
        let range_len = usize::try_from(range.end - range.start)
            .map_err(|_| vortex_err!("read range exceeds usize"))?;
        let row_scope = rows.as_scope();
        if row_scope.selection.len() != range_len {
            vortex_bail!(
                "selection length {} does not match range length {range_len}",
                row_scope.selection.len()
            );
        }
        if row_scope.demand.len() != range_len {
            vortex_bail!(
                "demand length {} does not match range length {range_len}",
                row_scope.demand.len()
            );
        }
        if row_scope.selection.all_false() {
            return Ok(Box::new(ChunkedReadTask {
                dtype: self.node.dtype.clone(),
                parts: vec![ChunkedReadPart::Ready(
                    ConstantArray::new(Scalar::default_value(&self.node.dtype), 0).into_array(),
                )],
            }));
        }

        let dense_scope = row_scope.selection.all_true() && row_scope.demand.all_true();
        let selected_scope = !dense_scope && row_scope.demands_all_selected();
        let mut parts = Vec::new();
        let mut idx = self.node.chunked.first_chunk(range.start);
        while idx + 1 < self.node.chunked.offsets.len()
            && self.node.chunked.offsets[idx] < range.end
        {
            let chunk_start = self.node.chunked.offsets[idx];
            let chunk_end = self.node.chunked.offsets[idx + 1];
            let local =
                range.start.saturating_sub(chunk_start)..(range.end.min(chunk_end) - chunk_start);
            let sel_start = usize::try_from(chunk_start.max(range.start) - range.start)
                .map_err(|_| vortex_err!("read range exceeds usize"))?;
            let sel_end = usize::try_from(chunk_end.min(range.end) - range.start)
                .map_err(|_| vortex_err!("read range exceeds usize"))?;
            let chunk_selection = row_scope.selection.slice(sel_start..sel_end);
            idx += 1;
            if chunk_selection.all_false() {
                continue;
            }
            let chunk_demand = row_scope.demand.slice(sel_start..sel_end);
            if chunk_demand.all_false() {
                parts.push(ChunkedReadPart::Ready(
                    ConstantArray::new(
                        Scalar::default_value(&self.node.dtype),
                        chunk_selection.true_count(),
                    )
                    .into_array(),
                ));
                continue;
            }
            let chunk_idx = idx - 1;
            let read =
                self.node
                    .child_read(chunk_idx, &self.state, self.node.chunked.ctx.session())?;
            let chunk_rows = if dense_scope || selected_scope {
                OwnedRowScope::selected(chunk_selection.clone())
            } else {
                OwnedRowScope::try_new(chunk_selection.clone(), chunk_demand)?
            };
            let expected_len = chunk_selection.true_count();
            parts.push(ChunkedReadPart::Pending {
                expected_len,
                task: Arc::clone(&read).create_task(local, chunk_rows, phase)?,
            });
        }
        match parts.len() {
            0 => vortex_bail!("chunked scoped read range {range:?} out of bounds"),
            _ => Ok(Box::new(ChunkedReadTask {
                dtype: self.node.dtype.clone(),
                parts,
            })),
        }
    }

    fn release(&self, frontier: u64) -> VortexResult<()> {
        self.node.release(frontier, self.state.as_ref())
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt_chain(f)
    }
}

impl PreparedEvidence for ChunkedPreparedEvidence {
    fn evidence<'a>(
        &'a self,
        req: &'a EvidenceRequest<'a>,
        io: &'a ReadContext,
        results: ReadResults,
    ) -> VortexResult<Vec<EvidenceFragment>> {
        if req.range.start >= req.range.end {
            return Ok(Vec::new());
        }
        let mut fragments = Vec::new();
        let mut idx = self.node.chunked.first_chunk(req.range.start);
        while idx + 1 < self.node.chunked.offsets.len()
            && self.node.chunked.offsets[idx] < req.range.end
        {
            let chunk_start = self.node.chunked.offsets[idx];
            let chunk_end = self.node.chunked.offsets[idx + 1];
            let local = req.range.start.saturating_sub(chunk_start)
                ..(req.range.end.min(chunk_end) - chunk_start);
            let recheck = req.mode == EvidenceMode::RecheckBeforeProjection;
            let child_plans = if let Some(hit) = self.state.children.lock().get(&idx) {
                hit.clone()
            } else if recheck {
                if let Some(hit) = self.state.recheck_children.lock().get(&idx) {
                    hit.clone()
                } else {
                    let node = self.node.child(idx, io.session())?;
                    let mut plan_ctx = self.state.chunked.child_prepare_ctx(idx, io.session());
                    let plans = node.prepare_evidence(&mut plan_ctx)?;
                    let planned = plans
                        .into_iter()
                        .filter(|plan| plan.recheck_before_projection())
                        .collect::<Vec<_>>();
                    let mut children = self.state.recheck_children.lock();
                    children.entry(idx).or_insert(planned).clone()
                }
            } else {
                let node = self.node.child(idx, io.session())?;
                let mut plan_ctx = self.state.chunked.child_prepare_ctx(idx, io.session());
                let planned = node.prepare_evidence(&mut plan_ctx)?;
                let mut children = self.state.children.lock();
                children.entry(idx).or_insert(planned).clone()
            };
            if !child_plans.is_empty() {
                let child_req = EvidenceRequest {
                    id: req.id,
                    version: req.version,
                    predicate: req.predicate,
                    range: local,
                    mode: req.mode,
                };
                for plan in child_plans {
                    if recheck && !plan.recheck_before_projection() {
                        continue;
                    }
                    for fragment in plan.evidence(&child_req, io, results.clone())? {
                        fragments.push(translate_fragment(fragment, chunk_start));
                    }
                }
            }
            idx += 1;
        }
        Ok(fragments)
    }

    fn recheck_before_projection(&self) -> bool {
        true
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "chunked")
    }

    fn create_task(
        self: Arc<Self>,
        req: OwnedEvidenceRequest,
        phase: ScanIoPhase,
    ) -> VortexResult<Box<dyn EvidenceTask>> {
        Ok(Box::new(ChunkedEvidenceTask {
            evidence: self,
            req,
            phase,
        }))
    }
}

impl EvidenceTask for ChunkedEvidenceTask {
    fn into_step(self: Box<Self>) -> VortexResult<EvidenceStep> {
        let Self {
            evidence,
            req,
            phase,
        } = *self;
        if req.range.start >= req.range.end {
            return Ok(EvidenceStep::new(
                Vec::new(),
                Vec::new(),
                move |io, results| evidence.evidence(&req.as_request(), io, results),
            ));
        }

        let mut required_reads = Vec::new();
        let mut prefetch_reads = Vec::new();
        let mut idx = evidence.node.chunked.first_chunk(req.range.start);
        while idx + 1 < evidence.node.chunked.offsets.len()
            && evidence.node.chunked.offsets[idx] < req.range.end
        {
            let chunk_start = evidence.node.chunked.offsets[idx];
            let chunk_end = evidence.node.chunked.offsets[idx + 1];
            let local = req.range.start.saturating_sub(chunk_start)
                ..(req.range.end.min(chunk_end) - chunk_start);
            let recheck = req.mode == EvidenceMode::RecheckBeforeProjection;
            let child_plans = if let Some(hit) = evidence.state.children.lock().get(&idx) {
                hit.clone()
            } else if recheck {
                if let Some(hit) = evidence.state.recheck_children.lock().get(&idx) {
                    hit.clone()
                } else {
                    let node = evidence.node.child(idx, &evidence.session)?;
                    let mut plan_ctx = evidence
                        .state
                        .chunked
                        .child_prepare_ctx(idx, &evidence.session);
                    let plans = node.prepare_evidence(&mut plan_ctx)?;
                    let planned = plans
                        .into_iter()
                        .filter(|plan| plan.recheck_before_projection())
                        .collect::<Vec<_>>();
                    let mut children = evidence.state.recheck_children.lock();
                    children.entry(idx).or_insert(planned).clone()
                }
            } else {
                let node = evidence.node.child(idx, &evidence.session)?;
                let mut plan_ctx = evidence
                    .state
                    .chunked
                    .child_prepare_ctx(idx, &evidence.session);
                let planned = node.prepare_evidence(&mut plan_ctx)?;
                let mut children = evidence.state.children.lock();
                children.entry(idx).or_insert(planned).clone()
            };
            if !child_plans.is_empty() {
                let child_req = OwnedEvidenceRequest {
                    id: req.id,
                    version: req.version,
                    predicate: req.predicate.clone(),
                    range: local,
                    mode: req.mode,
                };
                for plan in child_plans {
                    if recheck && !plan.recheck_before_projection() {
                        continue;
                    }
                    let step = Arc::clone(&plan)
                        .create_task(child_req.clone(), phase)?
                        .into_step()?;
                    required_reads.extend(step.required_reads);
                    prefetch_reads.extend(step.prefetch_reads);
                }
            }
            idx += 1;
        }
        Ok(EvidenceStep::new(
            required_reads,
            prefetch_reads,
            move |io, results| evidence.evidence(&req.as_request(), io, results),
        ))
    }
}

fn translate_fragment(mut fragment: EvidenceFragment, offset: u64) -> EvidenceFragment {
    fragment.rows = fragment.rows.start + offset..fragment.rows.end + offset;
    fragment
}
