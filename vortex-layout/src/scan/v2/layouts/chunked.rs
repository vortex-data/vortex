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
use vortex_array::ExecutionCtx;
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
use vortex_scan::plan::RowScope;
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
use vortex_scan::plan::request::ScanRequest;
use vortex_session::VortexSession;

use crate::layout_v2::Chunked;
use crate::layout_v2::Layout;
use crate::layout_v2::LayoutRef;
use crate::segments::SegmentPlanCtx;
use crate::segments::SegmentRequests;

pub(crate) fn new_scan_plan(
    layout: Layout<Chunked>,
    _req: &mut ScanRequest,
    session: &VortexSession,
) -> VortexResult<ScanPlanRef> {
    Ok(Arc::new(ChunkedScanPlan {
        layout: layout.to_layout(),
        offsets: layout.data().chunk_offsets().to_vec(),
        session: session.clone(),
        children: Mutex::new(FxHashMap::default()),
    }))
}

/// Reads a chunked layout: cumulative chunk offsets
/// (`offsets.len() == chunks + 1`), with chunk children expanded lazily
/// through their own layout vtables.
pub struct ChunkedScanPlan {
    layout: LayoutRef,
    offsets: Vec<u64>,
    session: VortexSession,
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
        let plan = self
            .layout
            .child(idx)?
            .new_scan_plan(&mut req, &self.session)?;
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
        Ok(vec![Arc::new(ChunkedPreparedEvidence { node, state })])
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
    /// The chunked scoped read: slice the selection and demand per
    /// overlapping chunk, skip chunks whose selection is all-false, and
    /// represent selected-but-undemanded chunks with dtype-default filler
    /// without expanding or reading the child.
    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a ReadContext,
        local_ctx: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        Box::pin(async move {
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
            if rows.selection.len() != range_len {
                vortex_bail!(
                    "selection length {} does not match range length {range_len}",
                    rows.selection.len()
                );
            }
            if rows.demand.len() != range_len {
                vortex_bail!(
                    "demand length {} does not match range length {range_len}",
                    rows.demand.len()
                );
            }
            if rows.selection.all_false() {
                return Ok(
                    ConstantArray::new(Scalar::default_value(self.node.layout.dtype()), 0)
                        .into_array(),
                );
            }

            let dtype = self.node.layout.dtype().clone();
            let dense_scope = rows.selection.all_true() && rows.demand.all_true();
            let selected_scope = !dense_scope && rows.demands_all_selected();
            let mut parts = Vec::new();
            let mut idx = self.node.first_chunk(range.start);
            while idx + 1 < self.node.offsets.len() && self.node.offsets[idx] < range.end {
                let chunk_start = self.node.offsets[idx];
                let chunk_end = self.node.offsets[idx + 1];
                let local = range.start.saturating_sub(chunk_start)
                    ..(range.end.min(chunk_end) - chunk_start);
                let sel_start = usize::try_from(chunk_start.max(range.start) - range.start)
                    .map_err(|_| vortex_err!("read range exceeds usize"))?;
                let sel_end = usize::try_from(chunk_end.min(range.end) - range.start)
                    .map_err(|_| vortex_err!("read range exceeds usize"))?;
                let chunk_selection = rows.selection.slice(sel_start..sel_end);
                idx += 1;
                if chunk_selection.all_false() {
                    continue;
                }
                let chunk_demand = rows.demand.slice(sel_start..sel_end);
                if chunk_demand.all_false() {
                    parts.push(
                        ConstantArray::new(
                            Scalar::default_value(&dtype),
                            chunk_selection.true_count(),
                        )
                        .into_array(),
                    );
                    continue;
                }
                let chunk_idx = idx - 1;
                let read = self.node.child_read(chunk_idx, &self.state, io.session())?;
                let chunk = if dense_scope || selected_scope {
                    read.read_scoped(local, RowScope::selected(&chunk_selection), io, local_ctx)
                        .await?
                } else {
                    let chunk_rows = RowScope::try_new(&chunk_selection, &chunk_demand)?;
                    read.read_scoped(local, chunk_rows, io, local_ctx).await?
                };
                if chunk.len() != chunk_selection.true_count() {
                    vortex_bail!(
                        "scoped chunk read returned length {}, expected {}",
                        chunk.len(),
                        chunk_selection.true_count()
                    );
                }
                parts.push(chunk);
            }
            match parts.len() {
                0 => vortex_bail!("chunked scoped read range {range:?} out of bounds"),
                1 => Ok(parts.swap_remove(0)),
                _ => Ok(ChunkedArray::try_new(parts, dtype)?.into_array()),
            }
        })
    }

    fn segment_requests(
        &self,
        range: Range<u64>,
        rows: RowScope<'_>,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        if range.start >= range.end {
            vortex_bail!("empty chunked scoped read range");
        }
        #[cfg(debug_assertions)]
        {
            let released = self.state.released.load(Ordering::Relaxed);
            debug_assert!(
                range.start >= released,
                "chunked request planning {range:?} below the released frontier {released}"
            );
        }
        let range_len = usize::try_from(range.end - range.start)
            .map_err(|_| vortex_err!("read range exceeds usize"))?;
        if rows.selection.len() != range_len {
            vortex_bail!(
                "selection length {} does not match range length {range_len}",
                rows.selection.len()
            );
        }
        if rows.demand.len() != range_len {
            vortex_bail!(
                "demand length {} does not match range length {range_len}",
                rows.demand.len()
            );
        }
        if rows.selection.all_false() {
            return Ok(SegmentRequests::none());
        }

        let dense_scope = rows.selection.all_true() && rows.demand.all_true();
        let selected_scope = !dense_scope && rows.demands_all_selected();
        let mut requests = SegmentRequests::none();
        let mut saw_overlap = false;
        let mut idx = self.node.first_chunk(range.start);
        while idx + 1 < self.node.offsets.len() && self.node.offsets[idx] < range.end {
            saw_overlap = true;
            let chunk_start = self.node.offsets[idx];
            let chunk_end = self.node.offsets[idx + 1];
            let local =
                range.start.saturating_sub(chunk_start)..(range.end.min(chunk_end) - chunk_start);
            let sel_start = usize::try_from(chunk_start.max(range.start) - range.start)
                .map_err(|_| vortex_err!("read range exceeds usize"))?;
            let sel_end = usize::try_from(chunk_end.min(range.end) - range.start)
                .map_err(|_| vortex_err!("read range exceeds usize"))?;
            let chunk_selection = rows.selection.slice(sel_start..sel_end);
            idx += 1;
            if chunk_selection.all_false() {
                continue;
            }
            let chunk_demand = rows.demand.slice(sel_start..sel_end);
            if chunk_demand.all_false() {
                continue;
            }
            let chunk_idx = idx - 1;
            let read = self.node.child_read(chunk_idx, &self.state, cx.session())?;
            let chunk_requests = if dense_scope || selected_scope {
                read.segment_requests(local, RowScope::selected(&chunk_selection), cx)?
            } else {
                let chunk_rows = RowScope::try_new(&chunk_selection, &chunk_demand)?;
                read.segment_requests(local, chunk_rows, cx)?
            };
            requests.extend(chunk_requests);
            if requests.is_unknown() {
                return Ok(requests);
            }
        }
        if !saw_overlap {
            vortex_bail!("chunked scoped read range {range:?} out of bounds");
        }
        Ok(requests)
    }

    fn release(&self, frontier: u64) -> VortexResult<()> {
        self.node.release(frontier, &self.state)
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt_chain(f)
    }
}

impl ScanPlan for ChunkedExprScanPlan {
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
    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a ReadContext,
        local_ctx: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        Box::pin(async move {
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
            if rows.selection.len() != range_len {
                vortex_bail!(
                    "selection length {} does not match range length {range_len}",
                    rows.selection.len()
                );
            }
            if rows.demand.len() != range_len {
                vortex_bail!(
                    "demand length {} does not match range length {range_len}",
                    rows.demand.len()
                );
            }
            if rows.selection.all_false() {
                return Ok(
                    ConstantArray::new(Scalar::default_value(&self.node.dtype), 0).into_array(),
                );
            }

            let dense_scope = rows.selection.all_true() && rows.demand.all_true();
            let selected_scope = !dense_scope && rows.demands_all_selected();
            let mut parts = Vec::new();
            let mut idx = self.node.chunked.first_chunk(range.start);
            while idx + 1 < self.node.chunked.offsets.len()
                && self.node.chunked.offsets[idx] < range.end
            {
                let chunk_start = self.node.chunked.offsets[idx];
                let chunk_end = self.node.chunked.offsets[idx + 1];
                let local = range.start.saturating_sub(chunk_start)
                    ..(range.end.min(chunk_end) - chunk_start);
                let sel_start = usize::try_from(chunk_start.max(range.start) - range.start)
                    .map_err(|_| vortex_err!("read range exceeds usize"))?;
                let sel_end = usize::try_from(chunk_end.min(range.end) - range.start)
                    .map_err(|_| vortex_err!("read range exceeds usize"))?;
                let chunk_selection = rows.selection.slice(sel_start..sel_end);
                idx += 1;
                if chunk_selection.all_false() {
                    continue;
                }
                let chunk_demand = rows.demand.slice(sel_start..sel_end);
                if chunk_demand.all_false() {
                    parts.push(
                        ConstantArray::new(
                            Scalar::default_value(&self.node.dtype),
                            chunk_selection.true_count(),
                        )
                        .into_array(),
                    );
                    continue;
                }
                let chunk_idx = idx - 1;
                let read = self.node.child_read(chunk_idx, &self.state, io.session())?;
                let chunk = if dense_scope || selected_scope {
                    read.read_scoped(local, RowScope::selected(&chunk_selection), io, local_ctx)
                        .await?
                } else {
                    let chunk_rows = RowScope::try_new(&chunk_selection, &chunk_demand)?;
                    read.read_scoped(local, chunk_rows, io, local_ctx).await?
                };
                if chunk.len() != chunk_selection.true_count() {
                    vortex_bail!(
                        "scoped chunk read returned length {}, expected {}",
                        chunk.len(),
                        chunk_selection.true_count()
                    );
                }
                parts.push(chunk);
            }
            match parts.len() {
                0 => vortex_bail!("chunked scoped read range {range:?} out of bounds"),
                1 => Ok(parts.swap_remove(0)),
                _ => Ok(ChunkedArray::try_new(parts, self.node.dtype.clone())?.into_array()),
            }
        })
    }

    fn segment_requests(
        &self,
        range: Range<u64>,
        rows: RowScope<'_>,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        if range.start >= range.end {
            vortex_bail!("empty chunked scoped read range");
        }
        #[cfg(debug_assertions)]
        {
            let released = self.state.released.load(Ordering::Relaxed);
            debug_assert!(
                range.start >= released,
                "chunked expression request planning {range:?} below the released frontier {released}"
            );
        }
        let range_len = usize::try_from(range.end - range.start)
            .map_err(|_| vortex_err!("read range exceeds usize"))?;
        if rows.selection.len() != range_len {
            vortex_bail!(
                "selection length {} does not match range length {range_len}",
                rows.selection.len()
            );
        }
        if rows.demand.len() != range_len {
            vortex_bail!(
                "demand length {} does not match range length {range_len}",
                rows.demand.len()
            );
        }
        if rows.selection.all_false() {
            return Ok(SegmentRequests::none());
        }

        let dense_scope = rows.selection.all_true() && rows.demand.all_true();
        let selected_scope = !dense_scope && rows.demands_all_selected();
        let mut requests = SegmentRequests::none();
        let mut saw_overlap = false;
        let mut idx = self.node.chunked.first_chunk(range.start);
        while idx + 1 < self.node.chunked.offsets.len()
            && self.node.chunked.offsets[idx] < range.end
        {
            saw_overlap = true;
            let chunk_start = self.node.chunked.offsets[idx];
            let chunk_end = self.node.chunked.offsets[idx + 1];
            let local =
                range.start.saturating_sub(chunk_start)..(range.end.min(chunk_end) - chunk_start);
            let sel_start = usize::try_from(chunk_start.max(range.start) - range.start)
                .map_err(|_| vortex_err!("read range exceeds usize"))?;
            let sel_end = usize::try_from(chunk_end.min(range.end) - range.start)
                .map_err(|_| vortex_err!("read range exceeds usize"))?;
            let chunk_selection = rows.selection.slice(sel_start..sel_end);
            idx += 1;
            if chunk_selection.all_false() {
                continue;
            }
            let chunk_demand = rows.demand.slice(sel_start..sel_end);
            if chunk_demand.all_false() {
                continue;
            }
            let chunk_idx = idx - 1;
            let read = self.node.child_read(chunk_idx, &self.state, cx.session())?;
            let chunk_requests = if dense_scope || selected_scope {
                read.segment_requests(local, RowScope::selected(&chunk_selection), cx)?
            } else {
                let chunk_rows = RowScope::try_new(&chunk_selection, &chunk_demand)?;
                read.segment_requests(local, chunk_rows, cx)?
            };
            requests.extend(chunk_requests);
            if requests.is_unknown() {
                return Ok(requests);
            }
        }
        if !saw_overlap {
            vortex_bail!("chunked scoped read range {range:?} out of bounds");
        }
        Ok(requests)
    }

    fn release(&self, frontier: u64) -> VortexResult<()> {
        self.node.release(frontier, &self.state)
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
    ) -> BoxFuture<'a, VortexResult<Vec<EvidenceFragment>>> {
        Box::pin(async move {
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
                        for fragment in plan.evidence(&child_req, io).await? {
                            fragments.push(translate_fragment(fragment, chunk_start));
                        }
                    }
                }
                idx += 1;
            }
            Ok(fragments)
        })
    }

    fn segment_requests(
        &self,
        req: &EvidenceRequest<'_>,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        if req.range.start >= req.range.end {
            return Ok(SegmentRequests::none());
        }

        let mut requests = SegmentRequests::none();
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
                    let node = self.node.child(idx, cx.session())?;
                    let mut plan_ctx = self.state.chunked.child_prepare_ctx(idx, cx.session());
                    let plans = node.prepare_evidence(&mut plan_ctx)?;
                    let planned = plans
                        .into_iter()
                        .filter(|plan| plan.recheck_before_projection())
                        .collect::<Vec<_>>();
                    let mut children = self.state.recheck_children.lock();
                    children.entry(idx).or_insert(planned).clone()
                }
            } else {
                let node = self.node.child(idx, cx.session())?;
                let mut plan_ctx = self.state.chunked.child_prepare_ctx(idx, cx.session());
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
                    requests.extend(plan.segment_requests(&child_req, cx)?);
                    if requests.is_unknown() {
                        return Ok(requests);
                    }
                }
            }
            idx += 1;
        }
        Ok(requests)
    }

    fn recheck_before_projection(&self) -> bool {
        true
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "chunked")
    }
}

fn translate_fragment(mut fragment: EvidenceFragment, offset: u64) -> EvidenceFragment {
    fragment.rows = fragment.rows.start + offset..fragment.rows.end + offset;
    fragment
}
