// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`LayoutPlan`] trait and its supporting types.
//!
//! A `LayoutPlan` is the unit of recursive plan-tree construction
//! returned by [`crate::Layout::plan`]. See `LAYOUT_PLAN.md` § Model.

use std::any::Any;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_utils::dyn_traits::DynEq;
use vortex_utils::dyn_traits::DynHash;

use crate::segments::SegmentSource;
use crate::v2::dataflow::LayoutLoweringCtx;
use crate::v2::dataflow::OutputFrontier;
use crate::v2::demand::DemandSource;
use crate::v2::demand::RowDemand;
use crate::v2::scan_ctx::ScanCtx;

pub type LayoutPlanRef = Arc<dyn LayoutPlan>;

/// A node in a layout plan tree. Each node produces output in one
/// row domain.
///
/// `partition_count()` / `partition_stats(i)` describe the natural
/// splits a plan exposes — each split is a `Range<u64>` in this
/// plan's row coordinate space. Engines pick a row range and call
/// [`LayoutPlan::execute`]; aligning with `partition_stats` minimises
/// slicing at the leaves but isn't required.
///
/// **Plans are mostly pure descriptions.** Plan nodes should not hold
/// decoded arrays or derived execution caches. The current exception
/// is flat leaf nodes, which pre-register shared segment futures so
/// the file I/O driver can coalesce all leaves before execution order
/// starts pulling them. Cross-execute sharing of derived values
/// arrives via `Let` / `Use` (see `LAYOUT_PLAN.md` § Tee and
/// CommonSubplanElimination).
///
/// See `LAYOUT_PLAN.md` § Model.
pub trait LayoutPlan: DynEq + DynHash + Send + Sync + 'static {
    /// The output schema of this plan node.
    fn schema(&self) -> &DType;

    /// Number of natural splits this plan exposes. Each split is a
    /// row range that the plan can produce atomically. Engines
    /// typically use this to decide how to spread work across
    /// partitions; the actual unit of execution is a `Range<u64>`
    /// passed to [`LayoutPlan::execute`].
    fn partition_count(&self) -> usize;

    /// The row range and stats for partition `i`. Drives engine
    /// partitioning decisions; the row range is what an aligned
    /// `execute` call would pass.
    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats>;

    /// True iff this plan emits rows in the layout's natural row order:
    /// within an `execute` call, rows in row-id order; across calls,
    /// caller-supplied ranges in supplied order.
    fn output_ordered(&self) -> bool;

    /// For each child, true iff this plan needs that child row-ordered.
    fn required_input_ordered(&self) -> Vec<bool>;

    /// For each child, true iff this plan's output preserves the
    /// child's ordering (i.e., reading our output gives back its rows
    /// in the same row order as the child produced them).
    fn maintains_input_order(&self) -> Vec<bool>;

    /// Coalesce or split partitions to match a target count. Default
    /// is `Err` unless `n == partition_count()`; layouts that can
    /// rebalance (typically `Chunked`) override.
    fn repartition(self: Arc<Self>, _n: usize) -> VortexResult<LayoutPlanRef> {
        Err(vortex_err!("repartition not supported by this layout plan"))
    }

    /// Children of this plan. Used by pushdown rules to walk the tree.
    fn children(&self) -> &[LayoutPlanRef];

    /// Rebuild this node with new children. Used by pushdown rules
    /// to produce a rewritten subtree without each node having to
    /// know how to reconstruct itself.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef>;

    /// Try to absorb `mask_plan` (a Bool-stream plan covering this
    /// plan's row coordinate space) into a rewritten version of self.
    ///
    /// Returns `Some(new_plan)` when the rewrite is possible. The
    /// caller (`FilterPlan` or a future pushdown rule) drops its
    /// wrapping mask and uses `new_plan` directly. Returns `None`
    /// when the plan can't absorb a mask itself; the caller must
    /// fall back to the wrapping `FilterPlan`.
    ///
    /// `mask_plan`'s row space must match `self`'s row space and its
    /// schema must be `DType::Bool(_)`. The rewritten plan must emit
    /// the *filtered* rows — same observable result as
    /// `FilterPlan(self, mask_plan)`.
    ///
    /// Default returns `None` so most plan nodes opt out; nodes that
    /// can usefully exploit the mask (`FlatPlan` reading with a
    /// selective mask, `ChunkedPlan` slicing the mask per chunk and
    /// pushing further down, etc.) override.
    ///
    /// See `LAYOUT_PLAN.md` § FilterPlan and its pushdown.
    fn try_pushdown_mask(self: Arc<Self>, _mask_plan: LayoutPlanRef) -> Option<LayoutPlanRef> {
        None
    }

    /// Lower this plan into the experimental single-scheduler model.
    ///
    /// This is deliberately not wired into DataFusion. It records plan
    /// metadata, lets leaves enqueue their initial scheduler work, and
    /// leaves execution to [`LayoutLoweringCtx::drive_to_completion`].
    ///
    /// The default lowering assumes every child shares the same row
    /// coordinate space as its parent. Plans that translate row
    /// coordinates, hide children from [`Self::children`], or are both
    /// an operator and a leaf override this method.
    fn lower_to_scheduler(
        &self,
        row_range: Range<u64>,
        ctx: &mut LayoutLoweringCtx,
    ) -> VortexResult<()> {
        let subplan =
            ctx.register_plan_node(row_range.clone(), self.schema(), self.children().len());
        if self.children().is_empty() {
            return ctx.register_leaf_work(subplan, row_range, self.schema());
        }

        for child in self.children() {
            child.lower_to_scheduler(row_range.clone(), ctx)?;
        }
        Ok(())
    }

    /// Read the rows in `row_range` from this plan, in this plan's
    /// row coordinate space. Returns a stream of arrays whose total
    /// row count is `row_range.len()`.
    ///
    /// `demand` is a coordinate-aware view onto the partition's
    /// [`RowDemand`] in the *same* coord system as `row_range`. Plans
    /// that don't change row domain pass it to children unchanged;
    /// plans that translate (`ChunkedPlan`, etc.) call
    /// [`RowDemand::scope`] before delegating. Subtrees in unrelated
    /// row spaces (e.g. a stats child) should pass
    /// [`RowDemand::empty`] instead.
    ///
    /// `frontier` is the matching output-production frontier for this
    /// row domain. The default implementation used today grants every
    /// requested row, preserving current stream behavior. Future
    /// schedulers can wrap it to expose different visible frontiers to
    /// different sub-plans while leaf operators keep asking the same
    /// local "may I produce the next N rows?" question.
    ///
    /// `ctx` is the per-scan execution context (see
    /// [`crate::v2::scan_ctx::ScanCtx`]). It carries the
    /// [`vortex_session::VortexSession`] for this scan plus a typed
    /// key/value map for state plans need to share across `execute`
    /// calls. The plan struct itself must remain a pure description.
    ///
    /// Cross-execute sharing of derived values (e.g. dict values) is
    /// expressed via [`crate::v2::let_use::LetPlan`], which publishes
    /// the value into `ctx` and consumers look it up by `LetId`.
    fn execute(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,
        frontier: &OutputFrontier,
        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream>;
}

// `dyn LayoutPlan` is `PartialEq + Eq + Hash` via the dyn-safe
// `DynEq` / `DynHash` helpers from `vortex-utils`. Concrete plan
// nodes need only impl regular `PartialEq` / `Eq` / `Hash` (manually
// — the typical fields like `Arc<dyn SegmentSource>` don't derive)
// to opt in. `Arc<dyn LayoutPlan>` then gets `Hash + PartialEq` for
// free via the std blanket impl, so `LayoutPlanRef` is usable as a
// `HashMap` key in the CSE pass.
impl PartialEq for dyn LayoutPlan {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other as &dyn Any)
    }
}

impl Eq for dyn LayoutPlan {}

impl Hash for dyn LayoutPlan {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dyn_hash(state);
    }
}

/// Lower one plan range into the experimental single-scheduler model.
///
/// This helper is intentionally detached from the DataFusion scan
/// path. It gives tests and local experiments a one-call way to build
/// the scheduler prototype, inspect queued leaf work, and drive it to
/// completion.
pub fn lower_to_single_scheduler(
    plan: &dyn LayoutPlan,
    row_range: Range<u64>,
) -> VortexResult<LayoutLoweringCtx> {
    let mut ctx = LayoutLoweringCtx::for_single_scheduler(row_range.end);
    ctx.with_global_range(row_range.clone(), |ctx| {
        plan.lower_to_scheduler(row_range, ctx)
    })?;
    Ok(ctx)
}

/// Compare two [`LayoutPlanRef`]s structurally. Uses `Arc::ptr_eq` as
/// a fast path; falls back to the trait's `dyn_eq`. Concrete plan
/// `PartialEq` impls should call this when comparing child plans.
pub fn plans_eq(a: &LayoutPlanRef, b: &LayoutPlanRef) -> bool {
    Arc::ptr_eq(a, b) || (**a).eq(&**b)
}

/// `plans_eq` over a `&[LayoutPlanRef]`.
pub fn plan_slices_eq(a: &[LayoutPlanRef], b: &[LayoutPlanRef]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| plans_eq(x, y))
}

/// Hash a [`LayoutPlanRef`]'s structural content. Concrete plan
/// `Hash` impls should call this when hashing child plans.
///
/// When called inside [`with_hash_cache`] (during CSE), the result
/// is memoised by `Arc` pointer identity. Plans share the same
/// `Arc` after CSE-driven pushdown (e.g. one mask referenced from N
/// FilteredFlatPlans), so without this cache CSE would re-hash the
/// same subtree once per occurrence — quadratic in the number of
/// shared references.
pub fn hash_plan<H: Hasher>(plan: &LayoutPlanRef, state: &mut H) {
    let ptr = Arc::as_ptr(plan) as *const () as usize;

    // Fast path: cache hit (only when `with_hash_cache` is active).
    let cached = HASH_CACHE.with(|c| c.borrow().as_ref().and_then(|m| m.get(&ptr).copied()));
    if let Some(h) = cached {
        state.write_u64(h);
        return;
    }

    // No cache or miss. If a cache is active, compute the hash via
    // the trait (recursive — but every nested `hash_plan` call
    // re-enters this function and benefits from the same cache),
    // store it, then write to the caller's hasher.
    let cache_active = HASH_CACHE.with(|c| c.borrow().is_some());
    if cache_active {
        let mut sub = rustc_hash::FxHasher::default();
        (**plan).hash(&mut sub);
        let h = sub.finish();
        HASH_CACHE.with(|c| {
            if let Some(m) = c.borrow_mut().as_mut() {
                m.insert(ptr, h);
            }
        });
        state.write_u64(h);
    } else {
        (**plan).hash(state);
    }
}

/// `hash_plan` over a `&[LayoutPlanRef]`.
pub fn hash_plan_slice<H: Hasher>(plans: &[LayoutPlanRef], state: &mut H) {
    plans.len().hash(state);
    for p in plans {
        hash_plan(p, state);
    }
}

thread_local! {
    /// Per-Arc structural hash memoisation, used by [`hash_plan`].
    /// Active inside [`with_hash_cache`] (i.e. during the CSE walk);
    /// outside it, [`hash_plan`] degrades to the uncached recursive
    /// hash so non-CSE callers see no behaviour change.
    static HASH_CACHE: std::cell::RefCell<Option<rustc_hash::FxHashMap<usize, u64>>> =
        const { std::cell::RefCell::new(None) };
}

/// Run `f` with a thread-local hash cache active. `hash_plan` calls
/// inside `f` are memoised by `Arc` pointer identity; outer
/// (non-cached) callers are unaffected. Re-entrant calls share the
/// same cache.
pub fn with_hash_cache<R>(f: impl FnOnce() -> R) -> R {
    let outer_active = HASH_CACHE.with(|c| c.borrow().is_some());
    if !outer_active {
        HASH_CACHE.with(|c| *c.borrow_mut() = Some(rustc_hash::FxHashMap::default()));
    }
    let result = f();
    if !outer_active {
        HASH_CACHE.with(|c| *c.borrow_mut() = None);
    }
    result
}

/// Arguments passed to [`crate::Layout::plan`]. Carries the consumer's
/// row selection, the expression to evaluate against the layout, and
/// a [`PlanCtx`] with cross-cutting handles.
///
/// Note: there is intentionally no `row_range` field. Engines express
/// partial scans via partition selection (FileOpener) or
/// `LayoutPlan::repartition` (ExecutionPlan boundary) — see
/// `LAYOUT_PLAN.md` § Partial scans.
#[derive(Clone)]
pub struct PlanArguments {
    pub selection: Selection,
    pub expr: Expression,
    pub ctx: PlanCtx,
}

impl PlanArguments {
    /// Replace the expression while keeping selection and context.
    /// Used by layouts that rewrite the expression on the way down
    /// (e.g., `Struct` field routing, `Dict` predicate rewrite).
    pub fn with_expr(self, expr: Expression) -> Self {
        Self { expr, ..self }
    }
}

/// Cross-cutting context threaded through [`crate::Layout::plan`].
///
/// Carries the [`SegmentSource`] used to fetch on-disk bytes at
/// execute time, the [`VortexSession`] used for plan-time setup
/// that touches the array context, and a [`ResourceCollector`] into
/// which layouts push any [`Resource`] / [`DemandSource`] handles
/// they need active at execute time.
///
/// Notably does *not* carry [`RowDemand`] — that's positional
/// (per-row coordinates), so it travels alongside `row_range` as an
/// explicit parameter to [`LayoutPlan::execute`] rather than as a
/// shared `ScanCtx` slot.
#[derive(Clone)]
pub struct PlanCtx {
    pub segment_source: Arc<dyn SegmentSource>,
    pub session: VortexSession,
    /// Collector for resources discovered during plan-building.
    /// Cloned alongside `PlanCtx`; all clones share one underlying
    /// `Arc<Mutex<...>>`, so layouts pushing into a clone are visible
    /// to the original. `Scan::build` drains it after planning and
    /// registers contents on the resulting `ScanPlan`.
    pub resources: ResourceCollector,
}

impl PlanCtx {
    /// Construct a plan context with the given segment source and
    /// session. Starts with an empty resource collector.
    pub fn new(segment_source: Arc<dyn SegmentSource>, session: VortexSession) -> Self {
        Self {
            segment_source,
            session,
            resources: ResourceCollector::default(),
        }
    }
}

/// Plan-time accumulator for [`DemandSource`] handles.
///
/// Layouts that need a per-scan [`DemandSource`] (e.g. a
/// `ZoneMapResource`) call [`Self::push_demand_source`] during their
/// `Layout::plan` implementation. The same `Arc` is also held by
/// whatever plan node consumes it. After planning,
/// [`crate::v2::scan::Scan::build`] drains the collector via
/// [`Self::take`] and registers the contents on the resulting
/// `ScanPlan` so they appear in the per-partition `RowDemand`.
#[derive(Clone, Default)]
pub struct ResourceCollector {
    inner: Arc<Mutex<ResourceCollectorInner>>,
}

#[derive(Default)]
struct ResourceCollectorInner {
    demand_sources: Vec<Arc<dyn DemandSource>>,
}

impl ResourceCollector {
    /// Register a demand source. The source's `ensure_ready` is
    /// awaited lazily on first pull (consumers that never query
    /// demand pay no init cost).
    pub fn push_demand_source(&self, source: Arc<dyn DemandSource>) {
        self.inner.lock().demand_sources.push(source);
    }

    /// Drain the collector and return the accumulated demand sources.
    pub fn take(&self) -> CollectedResources {
        let mut inner = self.inner.lock();
        CollectedResources {
            demand_sources: std::mem::take(&mut inner.demand_sources),
        }
    }
}

/// Result of [`ResourceCollector::take`].
pub struct CollectedResources {
    pub demand_sources: Vec<Arc<dyn DemandSource>>,
}

/// Stats a partition can vouch for. The row range is mandatory —
/// every plan knows what range its `i`-th partition covers, since
/// that's what an aligned `execute` would ask for. Other stats are
/// `Option<_>`.
///
/// Per-column stats (min/max/null counts) will be added in a later
/// PR when there's a real consumer.
#[derive(Clone, Debug)]
pub struct PartitionStats {
    pub row_range: Range<u64>,
    pub byte_size_estimate: Option<u64>,
}

impl PartitionStats {
    pub fn for_range(row_range: Range<u64>) -> Self {
        Self {
            row_range,
            byte_size_estimate: None,
        }
    }

    pub fn with_byte_size_estimate(mut self, n: u64) -> Self {
        self.byte_size_estimate = Some(n);
        self
    }

    /// Row count derived from the partition's row range.
    pub fn row_count(&self) -> u64 {
        self.row_range.end.saturating_sub(self.row_range.start)
    }
}
