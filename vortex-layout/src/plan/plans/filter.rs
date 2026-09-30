// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::ops::Range;

use vortex_array::EmptyMetadata;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

use crate::plan::Plan;
use crate::plan::PlanChildren;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::SegmentScan;
use crate::plan::check_child_count;
use crate::plan::exec::ExecContext;
use crate::plan::exec::ExecNode;
use crate::plan::exec::FilterNode;
use crate::plan::exec::SegmentScanNode;
use crate::plan::exec::Selection;

/// Keeps only the selected rows of its child.
///
/// A filter holds no predicate. The rows it keeps are the selection it is executed with; its child
/// produces every row of the same row domain, and values outside the selection are unspecified.
/// The filter's dtype and row domain are its child's.
#[derive(Clone, Debug)]
pub struct Filter;

/// Operator-specific data for a [`Filter`] plan.
#[derive(Clone, Debug)]
pub struct FilterData;

/// A plan that keeps only the selected rows of its child.
pub type FilterPlan = Plan<Filter>;

impl FilterPlan {
    /// Creates a filter over `child`.
    pub fn new(child: PlanRef) -> Self {
        PlanParts {
            vtable: Filter,
            dtype: child.dtype().clone(),
            row_count: child.row_count(),
            children: vec![child].into(),
            data: FilterData,
        }
        .into_typed()
    }

    /// Returns the plan whose rows are filtered.
    pub fn child_plan(&self) -> VortexResult<PlanRef> {
        self.child_required(0)
    }
}

impl PlanVTable for Filter {
    type PlanData = FilterData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.filter");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        Some(EmptyMetadata)
    }

    fn with_children(
        plan: &Plan<Self>,
        children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        check_child_count("Filter", children, 1)?;
        let child = children
            .get(0)?
            .ok_or_else(|| vortex_err!("Filter child is absent"))?;
        if child.dtype() != plan.dtype() || child.row_count() != plan.row_count() {
            vortex_bail!("Filter child does not match the filter's dtype and row count");
        }
        Ok(())
    }

    fn child_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        if index == 0 {
            Cow::Borrowed("child")
        } else {
            Cow::Owned(format!("child[{index}]"))
        }
    }

    /// Runs fused with a segment-scan child, as one [`SegmentScanNode`] that keeps the selected
    /// rows itself; over any other child, as a [`FilterNode`] that filters the child's whole
    /// pieces.
    fn exec(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: Mask,
        ctx: &ExecContext,
    ) -> VortexResult<Box<dyn ExecNode>> {
        let child = plan.child_plan()?;
        let Some(scan) = child.as_opt::<SegmentScan>() else {
            return Ok(Box::new(FilterNode::new(
                plan.clone(),
                Selection::try_new(rows, mask)?,
                ctx.session().clone(),
            )));
        };
        let filter = Some(mask.clone());
        Ok(Box::new(SegmentScanNode::try_new(
            scan.clone(),
            Selection::try_new(rows, mask)?,
            filter,
            ctx.clone(),
        )?))
    }
}
