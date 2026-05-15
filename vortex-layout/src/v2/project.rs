// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ProjectPlan`] — wraps a child plan, evaluates an [`Expression`] on
//! every batch it produces.
//!
//! ProjectPlan is a temporary helper for the PR 3 V2 entrypoint. The
//! full design (`LAYOUT_PLAN.md` § Per-layout `plan` walkthrough) has
//! `StructLayout::plan` route per field via `referenced_field_paths`,
//! which avoids reading the unreferenced fields. Until that lands,
//! `StructLayout::plan` reads every field and `ProjectPlan` applies the
//! caller's expression at the top level.

use std::sync::Arc;

use futures::StreamExt;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::v2::demand::RowDemand;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;

/// Applies a Vortex [`Expression`] to every batch produced by `child`.
pub struct ProjectPlan {
    child: LayoutPlanRef,
    expr: Expression,
    output_dtype: DType,
}

impl ProjectPlan {
    pub fn new(child: LayoutPlanRef, expr: Expression, output_dtype: DType) -> Self {
        Self {
            child,
            expr,
            output_dtype,
        }
    }
}

impl PartialEq for ProjectPlan {
    fn eq(&self, other: &Self) -> bool {
        crate::v2::plan::plans_eq(&self.child, &other.child)
            && self.expr == other.expr
            && self.output_dtype == other.output_dtype
    }
}

impl Eq for ProjectPlan {}

impl std::hash::Hash for ProjectPlan {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        crate::v2::plan::hash_plan(&self.child, state);
        self.expr.hash(state);
        self.output_dtype.hash(state);
    }
}

impl LayoutPlan for ProjectPlan {
    fn schema(&self) -> &DType {
        &self.output_dtype
    }

    fn partition_count(&self) -> usize {
        self.child.partition_count()
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        // Projection is row-preserving; row counts pass through.
        self.child.partition_stats(partition)
    }

    fn output_ordered(&self) -> bool {
        self.child.output_ordered()
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        vec![true]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn children(&self) -> &[LayoutPlanRef] {
        std::slice::from_ref(&self.child)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if children.len() != 1 {
            vortex_bail!(
                "ProjectPlan::with_new_children expected 1 child, got {}",
                children.len()
            );
        }
        let child = children
            .into_iter()
            .next()
            .vortex_expect("ProjectPlan: len-1 children vec was empty after the length check");
        Ok(Arc::new(Self {
            child,
            expr: self.expr.clone(),
            output_dtype: self.output_dtype.clone(),
        }))
    }

    fn try_pushdown_mask(self: Arc<Self>, mask_plan: LayoutPlanRef) -> Option<LayoutPlanRef> {
        // Projection is row-preserving: the mask covers our output
        // row space, which equals our child's row space. Push the
        // mask straight through to the child; if it absorbs we wrap
        // the rewritten child back in `ProjectPlan` and the
        // expression evaluates over the filtered rows.
        if !matches!(mask_plan.schema(), DType::Bool(_)) {
            return None;
        }
        let new_child = Arc::clone(&self.child).try_pushdown_mask(mask_plan)?;
        Some(Arc::new(Self {
            child: new_child,
            expr: self.expr.clone(),
            output_dtype: self.output_dtype.clone(),
        }))
    }

    fn execute(
        &self,
        row_range: std::ops::Range<u64>,
        demand: &RowDemand,
        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        let inner = self.child.execute(row_range, demand, ctx)?;
        let expr = self.expr.clone();
        let dtype = self.output_dtype.clone();
        let mapped = inner.map(move |chunk_res| chunk_res.and_then(|chunk| chunk.apply(&expr)));
        Ok(Box::pin(ArrayStreamAdapter::new(dtype, mapped)))
    }
}
