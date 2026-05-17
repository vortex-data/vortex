// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`DictDecodePlan`] — wraps a child plan that produces dict codes,
//! materialises a [`DictArray`] per chunk against the dict values,
//! and applies the caller's expression.
//!
//! Holds two child plans (codes + values) in fully-lowered form — no
//! raw `LayoutRef`. The values plan is built once at
//! `DictLayout::plan` time and read at the start of every `execute`
//! call; within one `execute` call the values are awaited once and
//! reused across every codes chunk.
//!
//! See `LAYOUT_PLAN.md` § Per-layout `plan` walkthrough / `DictLayout::plan`.

use std::ops::Range;
use std::sync::Arc;

use async_stream::try_stream;
use futures::StreamExt;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::SharedArray;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::v2::demand::RowDemand;
use crate::v2::experiment::trace_flow;
use crate::v2::plans::LayoutPlan;
use crate::v2::plans::LayoutPlanRef;
use crate::v2::plans::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;
use crate::v2::scheduler::LayoutLoweringCtx;
use crate::v2::scheduler::OutputFrontier;

/// Per-execute call: take codes from `codes_plan`, await the dict
/// values from `values_plan` once at the start of the output stream,
/// then materialise a [`DictArray`] per chunk and apply `expr`.
pub struct DictDecodePlan {
    codes_plan: LayoutPlanRef,
    values_plan: LayoutPlanRef,
    expr: Expression,
    output_dtype: DType,
    all_values_referenced: bool,
}

impl DictDecodePlan {
    pub fn new(
        codes_plan: LayoutPlanRef,
        values_plan: LayoutPlanRef,
        expr: Expression,
        output_dtype: DType,
        all_values_referenced: bool,
    ) -> Self {
        Self {
            codes_plan,
            values_plan,
            expr,
            output_dtype,
            all_values_referenced,
        }
    }

    /// The lowered values plan. Reads the dict's values table.
    pub fn values_plan(&self) -> &LayoutPlanRef {
        &self.values_plan
    }
}

impl PartialEq for DictDecodePlan {
    fn eq(&self, other: &Self) -> bool {
        crate::v2::plans::plans_eq(&self.codes_plan, &other.codes_plan)
            && crate::v2::plans::plans_eq(&self.values_plan, &other.values_plan)
            && self.expr == other.expr
            && self.output_dtype == other.output_dtype
            && self.all_values_referenced == other.all_values_referenced
    }
}

impl Eq for DictDecodePlan {}

impl std::hash::Hash for DictDecodePlan {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        crate::v2::plans::hash_plan(&self.codes_plan, state);
        crate::v2::plans::hash_plan(&self.values_plan, state);
        self.expr.hash(state);
        self.output_dtype.hash(state);
        self.all_values_referenced.hash(state);
    }
}

impl LayoutPlan for DictDecodePlan {
    fn schema(&self) -> &DType {
        &self.output_dtype
    }

    fn partition_count(&self) -> usize {
        self.codes_plan.partition_count()
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        // Decoding is row-preserving; row range passes through.
        self.codes_plan.partition_stats(partition)
    }

    fn output_ordered(&self) -> bool {
        self.codes_plan.output_ordered()
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        // Children order: [codes]. Values is row-shape-independent
        // (one materialised dict array per scan).
        vec![true]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn children(&self) -> &[LayoutPlanRef] {
        // Expose the codes child for the typical pushdown walker.
        // `values_plan` is reachable via `DictDecodePlan::values_plan`.
        std::slice::from_ref(&self.codes_plan)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if children.len() != 1 {
            vortex_bail!(
                "DictDecodePlan::with_new_children expected 1 child, got {}",
                children.len()
            );
        }
        let codes_plan = children
            .into_iter()
            .next()
            .ok_or_else(|| vortex_err!("DictDecodePlan::with_new_children: empty vec"))?;
        Ok(Arc::new(Self {
            codes_plan,
            values_plan: Arc::clone(&self.values_plan),
            expr: self.expr.clone(),
            output_dtype: self.output_dtype.clone(),
            all_values_referenced: self.all_values_referenced,
        }))
    }

    fn try_pushdown_mask(self: Arc<Self>, mask_plan: LayoutPlanRef) -> Option<LayoutPlanRef> {
        if !matches!(mask_plan.schema(), DType::Bool(_)) {
            if trace_flow() {
                tracing::debug!(
                    target: "vortex_layout::v2::flow",
                    output_dtype = %self.output_dtype,
                    "dict pushdown failed non-bool mask"
                );
            }
            return None;
        }

        let pushed_codes = Arc::clone(&self.codes_plan).try_pushdown_mask(mask_plan)?;

        if trace_flow() {
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                output_dtype = %self.output_dtype,
                "dict pushdown succeeded"
            );
        }

        Some(Arc::new(Self {
            codes_plan: pushed_codes,
            values_plan: Arc::clone(&self.values_plan),
            expr: self.expr.clone(),
            output_dtype: self.output_dtype.clone(),
            // Filtering codes can make previously referenced dictionary
            // values unreachable, so drop the stronger hint after pushdown.
            all_values_referenced: false,
        }))
    }

    fn lower_to_scheduler(
        &self,
        row_range: Range<u64>,
        ctx: &mut LayoutLoweringCtx,
    ) -> VortexResult<()> {
        ctx.register_plan_node(row_range.clone(), self.schema(), 2);

        let values_total = self.values_plan.partition_stats(0)?.row_count();
        let global_range = ctx.current_global_range();
        ctx.with_global_range(global_range, |ctx| {
            self.values_plan.lower_to_scheduler(0..values_total, ctx)
        })?;

        self.codes_plan.lower_to_scheduler(row_range, ctx)
    }

    fn execute(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,
        frontier: &OutputFrontier,

        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        // Read the entire values table. The values plan covers its
        // own row space (the dict's distinct-value count), not the
        // codes' row space — execute it over its full range with a
        // detached demand (the dict-values coord system is unrelated
        // to ours).
        let values_total = self
            .values_plan
            .partition_stats(0)
            .map(|s| s.row_count())
            .unwrap_or(0);

        let codes_plan = Arc::clone(&self.codes_plan);
        let values_plan = Arc::clone(&self.values_plan);
        let demand = demand.clone();
        let frontier = frontier.clone();
        let ctx_for_stream = ctx.clone();
        let expr = self.expr.clone();
        let dtype = self.output_dtype.clone();
        let all_values_referenced = self.all_values_referenced;
        let stream = try_stream! {
            let mut codes_stream = codes_plan.execute(row_range, &demand, &frontier, &ctx_for_stream)?;
            let Some(first_codes) = codes_stream.next().await else {
                return;
            };

            let values_demand = RowDemand::empty(values_total);
            let values_frontier = OutputFrontier::unbounded(values_total);
            let values_stream = values_plan.execute(
                0..values_total,
                &values_demand,
                &values_frontier,
                &ctx_for_stream,
            )?;

            // Materialise the values into a single shared array. Wrap
            // in `SharedArray` so each chunk's `DictArray::new_unchecked`
            // gets a cheap Arc-clone rather than re-canonicalising.
            let values = SharedArray::new(values_stream.read_all().await?).into_array();
            let mut pending = Some(first_codes?);
            loop {
                let codes = if let Some(codes) = pending.take() {
                    codes
                } else if let Some(codes_res) = codes_stream.next().await {
                    codes_res?
                } else {
                    break;
                };
                // SAFETY: matches the v1 `DictReader::projection_evaluation`
                // contract (`vortex-layout/src/layouts/dict/reader.rs:243`):
                // codes dtype is enforced by the codes child reader, and
                // `all_values_referenced` is purely a correctness hint.
                let array = unsafe {
                    DictArray::new_unchecked(codes, values.clone())
                        .set_all_values_referenced(all_values_referenced)
                }
                .into_array()
                .optimize()?;
                yield array.apply(&expr)?;
            }
        };
        Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)))
    }
}
