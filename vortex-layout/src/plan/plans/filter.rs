// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use crate::plan::Plan;
use crate::plan::PlanChildren;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::check_child_count;

const CHILD: usize = 0;
const MASK: usize = 1;

/// Keeps the rows of `child` where `mask` is true, with children ordered as `[child, mask]`.
///
/// Both children share one row domain, which is also the filter's. `mask` is a non-nullable
/// boolean plan: a predicate evaluated over the same rows, or a [`Selection`](crate::plan::Selection)
/// bound to the rows the plan is executed with. Values of `child` outside the mask are
/// unspecified. The filter's dtype is its child's.
#[derive(Clone, Debug)]
pub struct Filter;

/// A plan that keeps the rows of one child selected by another.
pub type FilterPlan = Plan<Filter>;

impl FilterPlan {
    /// Creates a filter keeping the rows of `child` where `mask` is true.
    pub fn try_new(child: PlanRef, mask: PlanRef) -> VortexResult<Self> {
        check_mask(&child, &mask)?;
        Ok(PlanParts {
            vtable: Filter,
            dtype: child.dtype().clone(),
            row_count: child.row_count(),
            children: vec![child, mask].into(),
            data: (),
        }
        .into_typed())
    }

    /// Returns the plan whose rows are filtered.
    pub fn child_plan(&self) -> VortexResult<PlanRef> {
        self.child_required(CHILD)
    }

    /// Returns the boolean plan selecting the rows to keep.
    pub fn mask(&self) -> VortexResult<PlanRef> {
        self.child_required(MASK)
    }
}

fn check_mask(child: &PlanRef, mask: &PlanRef) -> VortexResult<()> {
    if mask.dtype() != &selection_dtype() {
        vortex_bail!(
            "Filter mask must be a non-nullable bool, got {}",
            mask.dtype()
        );
    }
    if mask.row_count() != child.row_count() {
        vortex_bail!(
            "Filter mask has {} rows but its child has {}",
            mask.row_count(),
            child.row_count()
        );
    }
    Ok(())
}

impl PlanVTable for Filter {
    type PlanData = ();
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
        check_child_count("Filter", children, 2)?;
        let child = children
            .get(CHILD)?
            .ok_or_else(|| vortex_err!("Filter child is absent"))?;
        let mask = children
            .get(MASK)?
            .ok_or_else(|| vortex_err!("Filter mask is absent"))?;
        if child.dtype() != plan.dtype() || child.row_count() != plan.row_count() {
            vortex_bail!("Filter child does not match the filter's dtype and row count");
        }
        check_mask(&child, &mask)
    }

    fn child_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        match index {
            CHILD => Cow::Borrowed("child"),
            MASK => Cow::Borrowed("mask"),
            _ => Cow::Owned(format!("child[{index}]")),
        }
    }
}

/// The rows a plan is executed with, as a non-nullable boolean over its row domain.
///
/// A selection is a leaf. The executor binds it to the row mask supplied for each execution
/// range, so a [`Filter`] whose mask is a selection keeps exactly the rows the caller asked for.
#[derive(Clone, Debug)]
pub struct Selection;

/// A childless source of the execution row mask.
pub type SelectionPlan = Plan<Selection>;

impl SelectionPlan {
    /// Creates a selection covering `row_count` rows.
    pub fn new(row_count: u64) -> Self {
        PlanParts {
            vtable: Selection,
            dtype: selection_dtype(),
            row_count,
            children: PlanChildren::default(),
            data: (),
        }
        .into_typed()
    }
}

/// Returns the dtype of a selection, and of every [`Filter`] mask.
pub fn selection_dtype() -> DType {
    DType::Bool(Nullability::NonNullable)
}

impl PlanVTable for Selection {
    type PlanData = ();
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.selection");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        Some(EmptyMetadata)
    }

    fn with_children(
        _plan: &Plan<Self>,
        children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        check_child_count("Selection", children, 0)
    }
}
