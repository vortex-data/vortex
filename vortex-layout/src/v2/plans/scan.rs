// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`Scan`] — entrypoint for the LayoutPlan v2 path.
//!
//! [`Scan::build`] turns a layout, projection, and selection into a
//! [`LayoutPlanRef`] tree by recursing through `Layout::plan`.
//! Filtered scans decompose `filter` into top-level AND conjuncts,
//! plan each as a bool-stream, AND them, and wrap the projection
//! plan with `FilterPlan`. See `LAYOUT_PLAN.md` § Scan construction.

use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::split_conjunction;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::segments::SegmentSource;
use crate::v2::demand::DemandSource;
use crate::v2::demand::RowDemand;
use crate::v2::experiment::trace_flow;
use crate::v2::plans::LayoutPlan;
use crate::v2::plans::LayoutPlanRef;
use crate::v2::plans::PartitionStats;
use crate::v2::plans::PlanArguments;
use crate::v2::plans::PlanCtx;
use crate::v2::plans::and_bool::ConjunctInfo;
use crate::v2::plans::and_bool::ConjunctPlan;
use crate::v2::plans::cse::cse;
use crate::v2::plans::filter::FilterPlan;
use crate::v2::scan_ctx::ScanCtx;

/// Scan request for the LayoutPlan v2 path. Mirrors the inputs to
/// `vortex_layout::scan::scan_builder::ScanBuilder` but produces a
/// [`LayoutPlanRef`] rather than driving execution itself.
///
/// `Scan` always builds a plan covering the layout's full row space.
/// Engines that want to read only part of the file pick the relevant
/// partitions out of the resulting plan — see `LAYOUT_PLAN.md`
/// § Partial scans.
pub struct Scan {
    /// Root layout being scanned.
    pub layout: LayoutRef,
    /// Source of segment bytes for the file under scan.
    pub segment_source: Arc<dyn SegmentSource>,
    /// Session used at plan-construction time.
    pub session: VortexSession,
    /// Projection expression — defaults to `root()` for "all columns".
    pub projection: Expression,
    /// Optional filter expression. Decomposed into top-level AND
    /// conjuncts and combined into a mask plan; the projection plan
    /// is wrapped with [`crate::v2::plans::filter::FilterPlan`].
    pub filter: Option<Expression>,
    /// Pre-mask selection over the layout's row space.
    pub selection: Selection,
}

impl Scan {
    /// Build the layout plan tree for this scan.
    ///
    /// Projection-only and filtered scans are both supported.
    /// Selection (row sub-set hints) other than `Selection::All` is
    /// not yet plumbed through; callers (e.g. the DataFusion opener)
    /// must fall back to v1 in that case.
    pub fn build(&self) -> VortexResult<LayoutPlanRef> {
        if !matches!(self.selection, Selection::All) {
            vortex_bail!(
                "Scan::build does not yet support non-`All` selections; \
                 fall back to the v1 ScanBuilder path"
            );
        }

        let ctx = PlanCtx::new(Arc::clone(&self.segment_source), self.session.clone());

        let projection_plan = self.layout.plan(PlanArguments {
            selection: self.selection.clone(),
            expr: self.projection.clone(),
            ctx: ctx.clone(),
        })?;

        let row_count = self.layout.row_count();

        let Some(filter) = self.filter.as_ref() else {
            // Projection-only — still worth CSE in case the projection
            // expression itself produced duplicate subtrees (e.g. a
            // pack referring to the same field twice).
            // No filter → no demand machinery needed; skip ScanPlan.
            return cse(projection_plan);
        };

        // Decompose the filter into top-level AND conjuncts, plan
        // each as a bool-stream against the layout, AND them together,
        // and wrap projection with FilterPlan. See `LAYOUT_PLAN.md`
        // § Scan construction.
        let conjuncts = split_conjunction(filter);
        let mut conjunct_plans: Vec<(usize, u8, Expression, LayoutPlanRef)> = conjuncts
            .into_iter()
            .enumerate()
            .map(|(idx, expr)| {
                let cost = conjunct_order_cost(&expr);
                let plan = self.layout.plan(PlanArguments {
                    selection: self.selection.clone(),
                    expr: expr.clone(),
                    ctx: ctx.clone(),
                })?;
                Ok((idx, cost, expr, plan))
            })
            .collect::<VortexResult<_>>()?;
        conjunct_plans.sort_by_key(|(idx, cost, ..)| (*cost, *idx));
        tracing::debug!(
            order = ?conjunct_plans
                .iter()
                .map(|(idx, cost, expr, _)| format!("#{idx} cost={cost} {expr}"))
                .collect::<Vec<_>>(),
            "v2 conjunct order"
        );
        let conjunct_infos: Vec<ConjunctInfo> = conjunct_plans
            .iter()
            .map(|(idx, cost, expr, _)| ConjunctInfo {
                original_idx: *idx,
                cost: *cost,
                expr: expr.to_string(),
            })
            .collect();
        let conjunct_plans: Vec<LayoutPlanRef> = conjunct_plans
            .into_iter()
            .map(|(_, _, _, plan)| plan)
            .collect();

        let mask_plan: LayoutPlanRef = match conjunct_plans.len() {
            0 => return cse(projection_plan),
            1 => conjunct_plans
                .into_iter()
                .next()
                .ok_or_else(|| vortex_err!("len-1 conjunct_plans was empty"))?,
            _ => Arc::new(ConjunctPlan::with_conjuncts(
                conjunct_plans,
                conjunct_infos,
                row_count,
            )),
        };

        // Wrap projection with FilterPlan (or pushed-down equivalent),
        // CSE-collapse, then wrap the whole thing in ScanPlan so the
        // partition's RowDemand gets installed at execute-start.
        let body = cse(FilterPlan::new_or_pushdown(projection_plan, mask_plan))?;

        // Drain demand sources that layouts registered during
        // planning (e.g. `ZoneMapResource` from `ZonedLayout::plan`)
        // and attach them to the ScanPlan. Resource init is lazy:
        // sources `ensure_ready` only when first pulled.
        let collected = ctx.resources.take();
        let mut scan_plan = ScanPlan::new(body, row_count);
        for s in collected.demand_sources {
            scan_plan = scan_plan.with_demand_source(s);
        }
        Ok(Arc::new(scan_plan))
    }
}

/// Static first-pass ordering for AND conjuncts.
///
/// V2's conjunct streams can use an earlier conjunct's mask to reduce
/// later predicate work. Without any history, running the SQL order can
/// spend early batches on expensive low-value predicates. This heuristic
/// is intentionally simple: cheap scalar comparisons first, then likely
/// selective LIKEs, then negated/wide LIKEs. Runtime selectivity feedback
/// can refine this later.
fn conjunct_order_cost(expr: &Expression) -> u8 {
    if let Some(op) = expr.as_opt::<Binary>() {
        return match op {
            Operator::Gt | Operator::Gte | Operator::Lt | Operator::Lte => 10,
            Operator::Eq => 20,
            Operator::NotEq if compares_empty_string_literal(expr) => 25,
            Operator::NotEq => 30,
            Operator::And | Operator::Or => 90,
            Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => 100,
        };
    }

    if let Some(options) = expr.as_opt::<Like>() {
        return like_order_cost(expr, *options);
    }

    50
}

fn compares_empty_string_literal(expr: &Expression) -> bool {
    expr.children()
        .iter()
        .any(|child| utf8_literal(child).is_some_and(|value| value.is_empty()))
}

fn like_order_cost(expr: &Expression, options: LikeOptions) -> u8 {
    let pattern = expr.child(1);
    let pattern = utf8_literal(pattern);
    let leading_wildcard = pattern.is_some_and(|p| p.starts_with(['%', '_']));
    let mut cost = if leading_wildcard { 60 } else { 40 };
    if options.negated {
        cost += 20;
    }
    if options.case_insensitive {
        cost += 10;
    }
    cost
}

fn utf8_literal(expr: &Expression) -> Option<&str> {
    expr.as_opt::<Literal>()?
        .as_utf8_opt()?
        .value()
        .map(|s| s.as_str())
}

/// Top-level wrapper installed by [`Scan::build`] for filtered scans.
/// At execute time it constructs a fresh [`RowDemand`] from
/// `demand_sources` and threads it into the body via the `demand`
/// parameter. Resource init is fully lazy — sources `ensure_ready`
/// only when first pulled, so consumers that never query demand pay
/// nothing.
///
/// The demand inherited from the parent (typically `RowDemand::empty`,
/// since `ScanPlan` is the top-level entry point) is dropped —
/// `ScanPlan` is the boundary at which a new per-scan demand is
/// introduced.
///
/// Pure passthrough otherwise — schema, partition stats, children,
/// pushdown all delegate to the body.
pub struct ScanPlan {
    body: LayoutPlanRef,
    total_rows: u64,
    /// Sources whose pulled mask contributes to the per-partition
    /// `RowDemand`. Each source is also a `Resource` (lazy init via
    /// `ensure_ready` on first pull).
    demand_sources: Vec<Arc<dyn DemandSource>>,
}

impl ScanPlan {
    pub fn new(body: LayoutPlanRef, total_rows: u64) -> Self {
        Self {
            body,
            total_rows,
            demand_sources: Vec::new(),
        }
    }

    /// Register a [`DemandSource`] whose mask is intersected into the
    /// per-partition `RowDemand`.
    pub fn with_demand_source(mut self, source: Arc<dyn DemandSource>) -> Self {
        self.demand_sources.push(source);
        self
    }
}

impl PartialEq for ScanPlan {
    fn eq(&self, other: &Self) -> bool {
        // Demand sources participate by `Arc::ptr_eq` — two ScanPlans
        // built from different `Scan::build` calls never share source
        // Arcs, which matches CSE semantics.
        crate::v2::plans::plans_eq(&self.body, &other.body)
            && self.total_rows == other.total_rows
            && self.demand_sources.len() == other.demand_sources.len()
            && self
                .demand_sources
                .iter()
                .zip(&other.demand_sources)
                .all(|(a, b)| Arc::ptr_eq(a, b))
    }
}

impl Eq for ScanPlan {}

impl std::hash::Hash for ScanPlan {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        crate::v2::plans::hash_plan(&self.body, state);
        self.total_rows.hash(state);
        for s in &self.demand_sources {
            (Arc::as_ptr(s) as *const () as usize).hash(state);
        }
    }
}

impl LayoutPlan for ScanPlan {
    fn schema(&self) -> &DType {
        self.body.schema()
    }

    fn partition_count(&self) -> usize {
        self.body.partition_count()
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        self.body.partition_stats(partition)
    }

    fn output_ordered(&self) -> bool {
        self.body.output_ordered()
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        self.body.required_input_ordered()
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        self.body.maintains_input_order()
    }

    fn children(&self) -> &[LayoutPlanRef] {
        std::slice::from_ref(&self.body)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if children.len() != 1 {
            vortex_bail!(
                "ScanPlan::with_new_children expected 1 child (body), got {}",
                children.len()
            );
        }
        let body = children
            .into_iter()
            .next()
            .ok_or_else(|| vortex_err!("ScanPlan::with_new_children: empty vec"))?;
        Ok(Arc::new(Self {
            body,
            total_rows: self.total_rows,
            demand_sources: self.demand_sources.clone(),
        }))
    }

    fn execute(
        &self,
        row_range: std::ops::Range<u64>,
        _demand: &RowDemand,

        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        // Construct the per-partition RowDemand from our declared
        // sources and hand it to the body. `_demand` (from whoever
        // called us) is intentionally ignored — `ScanPlan` is the
        // boundary that introduces a fresh demand.
        let demand = RowDemand::new(self.demand_sources.clone(), self.total_rows);
        if trace_flow() {
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                row_start = row_range.start,
                row_end = row_range.end,
                total_rows = self.total_rows,
                demand_sources = self.demand_sources.len(),
                "scan execute"
            );
        }

        // Spawn each source's `ensure_ready` so init happens
        // concurrently with the body. Pulls inside the body lazily
        // await the same future, so they observe a ready (or quickly-
        // becoming-ready) resource without an up-front block.
        // Sources that nothing pulls still complete in the background
        // — the work is bounded and idempotent.
        let handle = ctx.session().handle();
        for src in &self.demand_sources {
            let src = Arc::clone(src);
            handle
                .spawn(async move {
                    drop(src.ensure_ready().await);
                })
                .detach();
        }

        self.body.execute(row_range, &demand, ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::Arc;

    use futures::StreamExt;
    use futures::stream;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::expr::root;
    use vortex_array::scalar_fn::session::ScalarFnSession;
    use vortex_array::session::ArraySession;
    use vortex_array::stream::ArrayStream as _;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::Handle;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_scan::selection::Selection;
    use vortex_session::VortexSession;

    use super::Scan;
    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::table::TableStrategy;
    use crate::scan::scan_builder::ScanBuilder;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialStreamAdapter;
    use crate::sequence::SequentialStreamExt as _;
    use crate::session::LayoutSession;
    use crate::test::SESSION;

    fn session_with_handle(handle: Handle) -> VortexSession {
        VortexSession::empty()
            .with::<ArraySession>()
            .with::<LayoutSession>()
            .with::<ScalarFnSession>()
            .with::<RuntimeSession>()
            .with_handle(handle)
    }

    /// Build a `Chunked(Struct(Flat, Flat))` layout with two chunks.
    /// Returns the segment source, the layout, and the array we wrote.
    fn build_chunked_struct_layout() -> (Arc<dyn SegmentSource>, LayoutRef, ArrayRef) {
        let chunk1 = StructArray::from_fields(
            [
                ("a", buffer![1i32, 2, 3].into_array()),
                ("b", buffer![10i32, 20, 30].into_array()),
            ]
            .as_slice(),
        )
        .unwrap()
        .into_array();
        let chunk2 = StructArray::from_fields(
            [
                ("a", buffer![4i32, 5].into_array()),
                ("b", buffer![40i32, 50].into_array()),
            ]
            .as_slice(),
        )
        .unwrap()
        .into_array();
        let dtype = chunk1.dtype().clone();

        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let segments_for_strategy = Arc::<TestSegments>::clone(&segments);
        let strategy = ChunkedLayoutStrategy::new(TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        ));

        let (mut sequence_id, eof) = SequenceId::root().split();
        let chunk1_for_write = chunk1.clone();
        let chunk2_for_write = chunk2.clone();
        let dtype_for_write = dtype.clone();

        let layout = block_on(|handle| async move {
            let session = session_with_handle(handle);
            strategy
                .write_stream(
                    ctx,
                    segments_for_strategy,
                    SequentialStreamAdapter::new(
                        dtype_for_write,
                        stream::iter([
                            Ok((sequence_id.advance(), chunk1_for_write)),
                            Ok((sequence_id.advance(), chunk2_for_write)),
                        ]),
                    )
                    .sendable(),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        let combined = ChunkedArray::try_new(vec![chunk1, chunk2], dtype)
            .unwrap()
            .into_array();
        (segments, layout, combined)
    }

    /// Drive the legacy [`ScanBuilder`] path to read the layout into a single array.
    fn read_v1(segments: Arc<dyn SegmentSource>, layout: &LayoutRef) -> VortexResult<ArrayRef> {
        read_v1_with(segments, layout, root(), None)
    }

    fn read_v1_with(
        segments: Arc<dyn SegmentSource>,
        layout: &LayoutRef,
        projection: vortex_array::expr::Expression,
        filter: Option<vortex_array::expr::Expression>,
    ) -> VortexResult<ArrayRef> {
        let (chunks, dtype) = block_on(|handle| async move {
            let session = session_with_handle(handle);
            let reader = layout.new_reader("v1".into(), segments, &session)?;
            let stream = ScanBuilder::new(session, reader)
                .with_projection(projection)
                .with_some_filter(filter)
                .into_array_stream()?;
            let dtype = stream.dtype().clone();
            let mut stream = stream;
            let mut chunks = Vec::new();
            while let Some(chunk) = stream.next().await {
                chunks.push(chunk?);
            }
            VortexResult::Ok((chunks, dtype))
        })?;
        Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
    }

    /// Drive the v2 [`Scan`] / [`crate::v2::plans::LayoutPlan`] path.
    fn read_v2(segments: Arc<dyn SegmentSource>, layout: &LayoutRef) -> VortexResult<ArrayRef> {
        read_v2_with(segments, layout, root(), None)
    }

    fn read_v2_with(
        segments: Arc<dyn SegmentSource>,
        layout: &LayoutRef,
        projection: vortex_array::expr::Expression,
        filter: Option<vortex_array::expr::Expression>,
    ) -> VortexResult<ArrayRef> {
        let layout = Arc::clone(layout);
        let row_count = layout.row_count();
        let (chunks, plan_dtype) = block_on(|handle| async move {
            let session = session_with_handle(handle);
            let scan = Scan {
                layout,
                segment_source: segments,
                session: session.clone(),
                projection,
                filter,
                selection: Selection::All,
            };
            let plan = scan.build()?;
            let plan_dtype = plan.schema().clone();
            let scan_ctx = crate::v2::scan_ctx::ScanCtx::new(session);

            let mut chunks = Vec::new();
            // Single execute call covering the entire layout's row
            // range. The plan internally handles whatever
            // chunking/slicing it needs. ScanPlan installs a fresh
            // demand internally; the parent demand we pass is detached.
            let parent_demand = crate::v2::demand::RowDemand::empty(row_count);
            let mut stream = plan.execute(0..row_count, &parent_demand, &scan_ctx)?;
            while let Some(chunk) = stream.next().await {
                chunks.push(chunk?);
            }
            VortexResult::Ok((chunks, plan_dtype))
        })?;
        Ok(ChunkedArray::try_new(chunks, plan_dtype)?.into_array())
    }

    fn read_v2_ranges_with(
        segments: Arc<dyn SegmentSource>,
        layout: &LayoutRef,
        projection: vortex_array::expr::Expression,
        filter: Option<vortex_array::expr::Expression>,
        ranges: Vec<Range<u64>>,
    ) -> VortexResult<ArrayRef> {
        let layout = Arc::clone(layout);
        let row_count = layout.row_count();
        let (chunks, plan_dtype) = block_on(|handle| async move {
            let session = session_with_handle(handle);
            let scan = Scan {
                layout,
                segment_source: segments,
                session: session.clone(),
                projection,
                filter,
                selection: Selection::All,
            };
            let plan = scan.build()?;
            let plan_dtype = plan.schema().clone();
            let parent_demand = crate::v2::demand::RowDemand::empty(row_count);

            let mut chunks = Vec::new();
            for range in ranges {
                let scan_ctx = crate::v2::scan_ctx::ScanCtx::new(session.clone());
                let mut stream = plan.execute(range, &parent_demand, &scan_ctx)?;
                while let Some(chunk) = stream.next().await {
                    chunks.push(chunk?);
                }
            }
            VortexResult::Ok((chunks, plan_dtype))
        })?;
        Ok(ChunkedArray::try_new(chunks, plan_dtype)?.into_array())
    }

    /// Build a single-chunk `Flat` layout backed by a primitive array.
    fn build_flat_layout() -> (Arc<dyn SegmentSource>, LayoutRef, ArrayRef) {
        let array = buffer![1i32, 2, 3, 4, 5].into_array();
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let segments_for_strategy = Arc::<TestSegments>::clone(&segments);
        let (ptr, eof) = SequenceId::root().split();
        let array_for_write = array.clone();
        let layout = block_on(|handle| async move {
            let session = session_with_handle(handle);
            FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    segments_for_strategy,
                    crate::sequence::SequentialArrayStreamExt::sequenced(
                        array_for_write.to_array_stream(),
                        ptr,
                    ),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();
        (segments, layout, array)
    }

    /// V1 and V2 must produce identical output for a projection-only scan.
    #[test]
    fn diff_v1_v2_projection_only_chunked_struct() -> VortexResult<()> {
        let (segments, layout, expected) = build_chunked_struct_layout();

        let v1 = read_v1(Arc::clone(&segments), &layout)?;
        let v2 = read_v2(Arc::clone(&segments), &layout)?;

        // V1 and V2 must agree.
        assert_arrays_eq!(v1, v2);
        // And both must agree with what we wrote.
        assert_arrays_eq!(v2, expected);

        Ok(())
    }

    /// Two `Layout::plan` calls on the same file-backed layout
    /// produce structurally-identical plan trees. CSE keys on
    /// `Hash + PartialEq` of `Arc<dyn LayoutPlan>`, so this is the
    /// invariant the pass relies on. The current `ViewedLayoutChildren`
    /// impl re-builds `LayoutRef` per `child(idx)` call, so
    /// `Arc::ptr_eq` would say the children are different — only
    /// structural equality saves us here.
    #[test]
    fn plans_are_structurally_eq_across_layout_plan_calls() -> VortexResult<()> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        use crate::segments::SegmentSource;
        use crate::v2::plans::PlanArguments;
        use crate::v2::plans::PlanCtx;

        let (segments, layout, _) = build_chunked_struct_layout();
        let session = SESSION.clone();
        let segment_source: Arc<dyn SegmentSource> = Arc::clone(&segments) as _;
        let ctx = PlanCtx::new(segment_source, session);
        let args = PlanArguments {
            selection: Selection::All,
            expr: root(),
            ctx,
        };
        let plan_a = layout.plan(args.clone())?;
        let plan_b = layout.plan(args)?;

        // Different `Arc` instances...
        assert!(
            !Arc::ptr_eq(&plan_a, &plan_b),
            "expected separate Arc instances",
        );
        // ...but structurally equal under `dyn LayoutPlan: Eq`.
        assert!(
            crate::v2::plans::plans_eq(&plan_a, &plan_b),
            "two Layout::plan calls should produce structurally-equal plans",
        );
        // And produce the same hash.
        let mut h_a = DefaultHasher::new();
        let mut h_b = DefaultHasher::new();
        crate::v2::plans::hash_plan(&plan_a, &mut h_a);
        crate::v2::plans::hash_plan(&plan_b, &mut h_b);
        assert_eq!(h_a.finish(), h_b.finish());
        Ok(())
    }

    /// Same diff but for a single-chunk `Flat` — exercises the bare
    /// `FlatLayout::plan` path with no `Chunked`/`Struct` wrappers.
    #[test]
    fn diff_v1_v2_projection_only_flat() -> VortexResult<()> {
        let (segments, layout, expected) = build_flat_layout();

        let v1 = read_v1(Arc::clone(&segments), &layout)?;
        let v2 = read_v2(Arc::clone(&segments), &layout)?;

        assert_arrays_eq!(v1, v2);
        assert_arrays_eq!(v2, expected);

        Ok(())
    }

    /// Project a single field out of the chunked struct via
    /// `get_item("a", root())`. Exercises `StructLayout::plan`'s
    /// partition-based field routing.
    #[test]
    fn diff_v1_v2_field_projection_chunked_struct() -> VortexResult<()> {
        use vortex_array::expr::get_item;
        let (segments, layout, _) = build_chunked_struct_layout();

        let proj = get_item("a", root());
        let v1 = read_v1_with(Arc::clone(&segments), &layout, proj.clone(), None)?;
        let v2 = read_v2_with(Arc::clone(&segments), &layout, proj, None)?;

        assert_arrays_eq!(v1, v2);
        Ok(())
    }

    /// Project a `pack(get_item, get_item)` re-ordering of struct
    /// fields. Exercises the partition + re-assembly `ProjectPlan`
    /// path on top of `StructLayout::plan`.
    #[test]
    fn diff_v1_v2_pack_projection_chunked_struct() -> VortexResult<()> {
        use vortex_array::dtype::Nullability;
        use vortex_array::expr::get_item;
        use vortex_array::expr::pack;
        let (segments, layout, _) = build_chunked_struct_layout();

        let proj = pack(
            [
                ("b_out", get_item("b", root())),
                ("a_out", get_item("a", root())),
            ],
            Nullability::NonNullable,
        );
        let v1 = read_v1_with(Arc::clone(&segments), &layout, proj.clone(), None)?;
        let v2 = read_v2_with(Arc::clone(&segments), &layout, proj, None)?;

        assert_arrays_eq!(v1, v2);
        Ok(())
    }

    /// Build a `Chunked(Dict(values=Flat, codes=Chunked(Flat)))` layout
    /// — strings with dictionary compression. Exercises
    /// `DictLayout::plan` + `DictDecodePlan`.
    fn build_dict_chunked_layout() -> (Arc<dyn SegmentSource>, LayoutRef, ArrayRef) {
        use vortex_array::arrays::VarBinArray;
        use vortex_array::dtype::Nullability::NonNullable;

        // Two chunks of repeated strings — high duplication so the
        // writer should reach for the dict strategy.
        let chunk1 = VarBinArray::from_iter(
            ["alpha", "beta", "alpha", "alpha"].into_iter().map(Some),
            DType::Utf8(NonNullable),
        )
        .into_array();
        let chunk2 = VarBinArray::from_iter(
            ["beta", "gamma", "alpha", "gamma"].into_iter().map(Some),
            DType::Utf8(NonNullable),
        )
        .into_array();
        let dtype = chunk1.dtype().clone();

        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let segments_for_strategy = Arc::<TestSegments>::clone(&segments);
        let strategy = ChunkedLayoutStrategy::new(crate::layouts::dict::writer::DictStrategy::new(
            FlatLayoutStrategy::default(),
            FlatLayoutStrategy::default(),
            FlatLayoutStrategy::default(),
            crate::layouts::dict::writer::DictLayoutOptions::default(),
        ));
        let (mut sequence_id, eof) = SequenceId::root().split();
        let chunk1_for_write = chunk1.clone();
        let chunk2_for_write = chunk2.clone();
        let dtype_for_write = dtype.clone();
        let layout = block_on(|handle| async move {
            let session = session_with_handle(handle);
            strategy
                .write_stream(
                    ctx,
                    segments_for_strategy,
                    SequentialStreamAdapter::new(
                        dtype_for_write,
                        stream::iter([
                            Ok((sequence_id.advance(), chunk1_for_write)),
                            Ok((sequence_id.advance(), chunk2_for_write)),
                        ]),
                    )
                    .sendable(),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        let combined = ChunkedArray::try_new(vec![chunk1, chunk2], dtype)
            .unwrap()
            .into_array();
        (segments, layout, combined)
    }

    /// V1/V2 must agree for a dict-encoded chunked Utf8 column with
    /// `root()` projection.
    #[test]
    fn diff_v1_v2_projection_only_dict() -> VortexResult<()> {
        let (segments, layout, expected) = build_dict_chunked_layout();

        let v1 = read_v1(Arc::clone(&segments), &layout)?;
        let v2 = read_v2(Arc::clone(&segments), &layout)?;

        assert_arrays_eq!(v1, v2);
        assert_arrays_eq!(v2, expected);
        Ok(())
    }

    /// `pack([], …)` projection: no field references. V2 must build a
    /// plan that does not recurse into any field's `Layout::plan` and
    /// emit a stream of empty structs of the layout's row count. The
    /// pre-fix behaviour would still plan every field through
    /// `plan_full_struct_with_projection`, paying per-field setup cost
    /// (and surfacing as the `COUNT(*)` regression on wide schemas).
    #[test]
    fn diff_v1_v2_empty_pack_projection_chunked_struct() -> VortexResult<()> {
        use vortex_array::dtype::Nullability;
        use vortex_array::expr::pack;
        let (segments, layout, _) = build_chunked_struct_layout();

        let proj = pack(
            Vec::<(&str, vortex_array::expr::Expression)>::new(),
            Nullability::NonNullable,
        );
        let v1 = read_v1_with(Arc::clone(&segments), &layout, proj.clone(), None)?;
        let v2 = read_v2_with(Arc::clone(&segments), &layout, proj, None)?;

        assert_arrays_eq!(v1, v2);
        // Five rows of empty struct.
        assert_eq!(v2.len(), 5);
        Ok(())
    }

    /// V1/V2 must agree on a single-conjunct filtered scan.
    #[test]
    fn diff_v1_v2_filtered_chunked_struct_single_conjunct() -> VortexResult<()> {
        use vortex_array::expr::eq;
        use vortex_array::expr::get_item;
        use vortex_array::expr::lit;
        let (segments, layout, _) = build_chunked_struct_layout();

        let projection = root();
        let filter = eq(get_item("a", root()), lit(2i32));

        let v1 = read_v1_with(
            Arc::clone(&segments),
            &layout,
            projection.clone(),
            Some(filter.clone()),
        )?;
        let v2 = read_v2_with(Arc::clone(&segments), &layout, projection, Some(filter))?;

        assert_arrays_eq!(v1, v2);
        Ok(())
    }

    /// The scheduler-backed partition driver should produce the same
    /// filtered scan output while returning arrays through its sink
    /// queue.
    #[test]
    fn scheduler_execute_filtered_chunked_struct_single_conjunct() -> VortexResult<()> {
        use vortex_array::expr::eq;
        use vortex_array::expr::get_item;
        use vortex_array::expr::lit;
        let (segments, layout, _) = build_chunked_struct_layout();

        let projection = root();
        let filter = eq(get_item("a", root()), lit(2i32));

        let v1 = read_v1_with(
            Arc::clone(&segments),
            &layout,
            projection.clone(),
            Some(filter.clone()),
        )?;
        let v2 = read_v2_with(Arc::clone(&segments), &layout, projection, Some(filter))?;

        assert_arrays_eq!(v1, v2);
        Ok(())
    }

    /// Build a `Zoned(Chunked(Flat))` layout — three chunks of 3
    /// rows each, with min/max stats per zone. Used to exercise
    /// `ZonedPruningPlan`.
    fn build_zoned_chunked_layout() -> (Arc<dyn SegmentSource>, LayoutRef, ArrayRef) {
        use vortex_array::arrays::ChunkedArray as ChunkedArrayInner;

        use crate::layouts::zoned::writer::ZonedLayoutOptions;
        use crate::layouts::zoned::writer::ZonedStrategy;
        use crate::sequence::SequentialArrayStreamExt;

        let chunks = vec![
            buffer![1i32, 2, 3].into_array(),
            buffer![4i32, 5, 6].into_array(),
            buffer![7i32, 8, 9].into_array(),
        ];
        let combined = ChunkedArrayInner::try_new(chunks.clone(), chunks[0].dtype().clone())
            .unwrap()
            .into_array();

        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let segments_for_strategy = Arc::<TestSegments>::clone(&segments);
        let strategy = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: 3,
                ..Default::default()
            },
        );
        let (ptr, eof) = SequenceId::root().split();
        let combined_for_write = combined.clone();
        let layout = block_on(|handle| async move {
            let session = session_with_handle(handle);
            let stream = combined_for_write.to_array_stream().sequenced(ptr);
            strategy
                .write_stream(ctx, segments_for_strategy, stream, eof, &session)
                .await
        })
        .unwrap();
        (segments, layout, combined)
    }

    /// V1/V2 must agree on a selective filter against a zoned layout.
    /// `> 7` prunes the first two zones; `< 4` prunes the last two.
    /// Both branches are tested.
    #[test]
    fn diff_v1_v2_zoned_pruning_high() -> VortexResult<()> {
        use vortex_array::expr::gt;
        use vortex_array::expr::lit;
        let (segments, layout, _) = build_zoned_chunked_layout();
        let projection = root();
        let filter = gt(root(), lit(7i32));
        let v1 = read_v1_with(
            Arc::clone(&segments),
            &layout,
            projection.clone(),
            Some(filter.clone()),
        )?;
        let v2 = read_v2_with(Arc::clone(&segments), &layout, projection, Some(filter))?;
        assert_arrays_eq!(v1, v2);
        Ok(())
    }

    #[test]
    fn diff_v1_v2_zoned_pruning_low() -> VortexResult<()> {
        use vortex_array::expr::lit;
        use vortex_array::expr::lt;
        let (segments, layout, _) = build_zoned_chunked_layout();
        let projection = root();
        let filter = lt(root(), lit(4i32));
        let v1 = read_v1_with(
            Arc::clone(&segments),
            &layout,
            projection.clone(),
            Some(filter.clone()),
        )?;
        let v2 = read_v2_with(Arc::clone(&segments), &layout, projection, Some(filter))?;
        assert_arrays_eq!(v1, v2);
        Ok(())
    }

    #[test]
    fn diff_v1_v2_zoned_pruning_partitioned_mid_chunk() -> VortexResult<()> {
        use vortex_array::expr::lit;
        use vortex_array::expr::lt;
        let (segments, layout, _) = build_zoned_chunked_layout();
        let projection = root();
        let filter = lt(root(), lit(4i32));
        let v1 = read_v1_with(
            Arc::clone(&segments),
            &layout,
            projection.clone(),
            Some(filter.clone()),
        )?;
        let row_count = layout.row_count();
        let v2 = read_v2_ranges_with(
            Arc::clone(&segments),
            &layout,
            projection,
            Some(filter),
            vec![0..2, 2..row_count],
        )?;
        assert_arrays_eq!(v1, v2);
        Ok(())
    }

    /// Filter that prunes nothing — every zone overlaps the predicate.
    /// Confirms ZonedPruningPlan still produces correct output when
    /// no zone is pruneable.
    #[test]
    fn diff_v1_v2_zoned_pruning_no_op() -> VortexResult<()> {
        use vortex_array::expr::gt;
        use vortex_array::expr::lit;
        let (segments, layout, _) = build_zoned_chunked_layout();
        let projection = root();
        // `> 0` keeps everything.
        let filter = gt(root(), lit(0i32));
        let v1 = read_v1_with(
            Arc::clone(&segments),
            &layout,
            projection.clone(),
            Some(filter.clone()),
        )?;
        let v2 = read_v2_with(Arc::clone(&segments), &layout, projection, Some(filter))?;
        assert_arrays_eq!(v1, v2);
        Ok(())
    }

    /// V1/V2 must agree when the filter has multiple AND conjuncts —
    /// `Scan::build` decomposes via `split_conjunction` and combines
    /// with `ConjunctPlan`.
    #[test]
    fn diff_v1_v2_filtered_chunked_struct_two_conjuncts() -> VortexResult<()> {
        use vortex_array::expr::and;
        use vortex_array::expr::get_item;
        use vortex_array::expr::gt;
        use vortex_array::expr::lit;
        use vortex_array::expr::lt;
        let (segments, layout, _) = build_chunked_struct_layout();

        let projection = root();
        // a > 1 AND b < 50 — touches both fields, ensures the AND
        // path runs (vs the single-mask short-circuit).
        let filter = and(
            gt(get_item("a", root()), lit(1i32)),
            lt(get_item("b", root()), lit(50i32)),
        );

        let v1 = read_v1_with(
            Arc::clone(&segments),
            &layout,
            projection.clone(),
            Some(filter.clone()),
        )?;
        let v2 = read_v2_with(Arc::clone(&segments), &layout, projection, Some(filter))?;

        assert_arrays_eq!(v1, v2);
        Ok(())
    }
}
