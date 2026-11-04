// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::ops::Range;

use vortex_array::EmptyMetadata;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

use crate::plan::physical::Plan;
use crate::plan::physical::PlanId;
use crate::plan::physical::PlanParts;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanSlots;
use crate::plan::physical::PlanVTable;
use crate::plan::physical::SegmentScan;
use crate::plan::physical::operators::MaskStage;
use crate::plan::physical::pipeline::Chain;
use crate::plan::physical::pipeline::Compiler;

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
            slots: [child].into(),
            data: FilterData,
        }
        .into_typed()
    }

    /// Returns the plan whose rows are filtered.
    pub fn child_plan(&self) -> VortexResult<PlanRef> {
        self.slot_required(0)
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

    fn with_slots(
        plan: &Plan<Self>,
        slots: &PlanSlots,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        slots.check_len("Filter", 1)?;
        let child = slots.required(0)?;
        if child.dtype() != plan.dtype() || child.row_count() != plan.row_count() {
            vortex_bail!("Filter child does not match the filter's dtype and row count");
        }
        Ok(())
    }

    fn slot_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        if index == 0 {
            Cow::Borrowed("child")
        } else {
            Cow::Owned(format!("slot[{index}]"))
        }
    }

    /// Fuses with a segment-scan child, whose source keeps the selected rows itself; over any
    /// other child, compiles the child over every row and keeps the selected rows of each batch.
    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        if mask.all_false() {
            return Ok(None);
        }
        let child = plan.child_plan()?;
        if let Some(scan) = child.as_opt::<SegmentScan>() {
            return compiler.scan(scan, rows, Some(mask.clone()));
        }
        let len = usize::try_from(rows.end - rows.start)?;
        let predicate = plan.dtype().is_boolean();
        let chain = compiler.compile(&child, rows, &Mask::new_true(len))?;
        Ok(chain.map(|chain| {
            if mask.all_true() {
                chain
            } else {
                chain.with(MaskStage::new(mask.clone(), predicate))
            }
        }))
    }
}
