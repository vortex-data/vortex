// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::sync::Arc;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use crate::plan::Plan;
use crate::plan::PlanChildren;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;

/// Concatenates its children row-wise.
#[derive(Clone, Debug)]
pub struct Concat;

/// Row offsets of each concatenated child.
#[derive(Clone, Debug)]
pub struct ConcatData {
    row_offsets: Arc<[u64]>,
}

/// A plan that concatenates its children row-wise.
pub type ConcatPlan = Plan<Concat>;

impl ConcatPlan {
    /// Creates a concatenation from potentially unresolved children without validation.
    ///
    /// # Safety
    ///
    /// Every child must have `dtype`; `row_offsets` must contain the cumulative row offset of
    /// every child; and the sum of all child row counts must equal `row_count` without overflow.
    pub(crate) unsafe fn from_children_unchecked(
        dtype: DType,
        row_count: u64,
        row_offsets: Arc<[u64]>,
        children: PlanChildren,
    ) -> Self {
        PlanParts {
            vtable: Concat,
            dtype,
            row_count,
            children,
            data: ConcatData { row_offsets },
        }
        .into_typed()
    }

    /// Creates a concatenation over `children`.
    ///
    /// Every child must produce `dtype`, and the row domain is the sum of the child row counts.
    pub fn try_new(dtype: DType, children: Vec<PlanRef>) -> VortexResult<Self> {
        let mut row_offsets = Vec::with_capacity(children.len());
        let mut row_count = 0u64;
        for child in &children {
            if child.dtype() != &dtype {
                vortex_bail!(
                    "Concat child dtype {} does not match {dtype}",
                    child.dtype()
                );
            }
            row_offsets.push(row_count);
            row_count = row_count
                .checked_add(child.row_count())
                .ok_or_else(|| vortex_error::vortex_err!("Concat row count overflow"))?;
        }
        // SAFETY: Child dtypes and the checked cumulative row metadata were validated above.
        Ok(unsafe {
            Self::from_children_unchecked(dtype, row_count, row_offsets.into(), children.into())
        })
    }

    /// Returns the first row of each child within this plan's row domain.
    pub fn row_offsets(&self) -> &[u64] {
        &self.data().row_offsets
    }
}

impl PlanVTable for Concat {
    type PlanData = ConcatData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.concat");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        // Row offsets are derived from the children, so nothing needs storing.
        Some(EmptyMetadata)
    }

    fn with_children(
        plan: &Plan<Self>,
        children: &PlanChildren,
        data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        if children.len() != plan.children().len() {
            vortex_bail!(
                "Concat expects {} children but got {}",
                plan.children().len(),
                children.len()
            );
        }

        let mut row_offsets = Vec::with_capacity(children.len());
        let mut row_count = 0u64;
        for child in children.iter() {
            let child = child?;
            if child.dtype() != plan.dtype() {
                vortex_bail!(
                    "Concat child dtype {} does not match {}",
                    child.dtype(),
                    plan.dtype()
                );
            }
            row_offsets.push(row_count);
            row_count = row_count
                .checked_add(child.row_count())
                .ok_or_else(|| vortex_error::vortex_err!("Concat row count overflow"))?;
        }
        if row_count != plan.row_count() {
            vortex_bail!(
                "Concat children have {row_count} rows but the plan has {}",
                plan.row_count()
            );
        }
        data.row_offsets = row_offsets.into();
        Ok(())
    }

    fn child_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        Cow::Owned(format!("chunks[{index}]"))
    }
}
