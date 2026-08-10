// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;

use vortex_array::ProstMetadata;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::plan::Plan;
use crate::plan::PlanChildren;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::check_child_count;

/// Adds row-index support to its child, offsetting row numbers into the file's row domain.
#[derive(Clone, Debug)]
pub struct RowIdx;

/// The row offset applied to the child domain.
#[derive(Clone, Debug)]
pub struct RowIdxData {
    row_offset: u64,
}

/// A plan that adds row-index support to its child.
pub type RowIdxPlan = Plan<RowIdx>;

impl RowIdxPlan {
    /// Creates a row-index plan with `row_offset` applied to its child domain.
    pub fn new(row_offset: u64, child: PlanRef) -> Self {
        PlanParts {
            vtable: RowIdx,
            dtype: child.dtype().clone(),
            row_count: child.row_count(),
            children: vec![child].into(),
            data: RowIdxData { row_offset },
        }
        .into_typed()
    }

    /// Returns the row offset applied to the child domain.
    pub fn row_offset(&self) -> u64 {
        self.data().row_offset
    }

    /// Returns the child plan.
    pub fn child_plan(&self) -> VortexResult<PlanRef> {
        self.child_required(0)
    }
}

impl PlanVTable for RowIdx {
    type PlanData = RowIdxData;
    type Metadata = ProstMetadata<RowIdxPlanMetadata>;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.row_idx");
        *ID
    }

    fn metadata(plan: &Plan<Self>) -> Option<Self::Metadata> {
        Some(ProstMetadata(RowIdxPlanMetadata {
            row_offset: plan.data().row_offset,
        }))
    }

    fn with_children(
        plan: &Plan<Self>,
        children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        check_child_count("RowIdx", children, 1)?;
        let child = children
            .get(0)?
            .ok_or_else(|| vortex_error::vortex_err!("RowIdx child is absent"))?;
        if child.dtype() != plan.dtype() || child.row_count() != plan.row_count() {
            vortex_error::vortex_bail!(
                "RowIdx child shape changed from ({}, {}) to ({}, {})",
                plan.dtype(),
                plan.row_count(),
                child.dtype(),
                child.row_count()
            );
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
}

/// Serialized metadata for a [`RowIdx`] plan.
#[derive(Clone, PartialEq, Eq, ::prost::Message)]
pub struct RowIdxPlanMetadata {
    /// The row offset applied to the child domain.
    #[prost(uint64, tag = "1")]
    pub row_offset: u64,
}
