// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::ops::Range;

use vortex_array::EmptyMetadata;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::plan::Plan;
use crate::plan::PlanArrayFuture;
use crate::plan::PlanChildren;
use crate::plan::PlanExecutionContext;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::check_child_count;

pub(crate) const DATA: usize = 0;
pub(crate) const ZONES: usize = 1;

/// Reads data alongside the zone statistics summarising it.
///
/// This operator covers both `vortex.zoned` layouts and legacy `vortex.stats` layouts, which have
/// the same physical child shape.
#[derive(Clone, Debug)]
pub struct Zoned;

/// A plan that pairs data with its zone statistics.
pub type ZonedPlan = Plan<Zoned>;

impl ZonedPlan {
    pub(crate) fn from_children(dtype: DType, row_count: u64, children: PlanChildren) -> Self {
        PlanParts {
            vtable: Zoned,
            dtype,
            row_count,
            children,
            data: (),
        }
        .into_typed()
    }

    /// Creates a zoned plan over `data` summarised by `zones`.
    pub fn new(data: PlanRef, zones: PlanRef) -> Self {
        let dtype: DType = data.dtype().clone();
        let row_count = data.row_count();
        Self::from_children(dtype, row_count, vec![data, zones].into())
    }

    /// Returns the plan producing the summarised data.
    pub fn data_plan(&self) -> VortexResult<PlanRef> {
        self.child_required(DATA)
    }

    /// Returns the plan producing the zone statistics.
    pub fn zones_plan(&self) -> VortexResult<PlanRef> {
        self.child_required(ZONES)
    }
}

impl PlanVTable for Zoned {
    type PlanData = ();
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.zoned");
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
        check_child_count("Zoned", children, 2)?;
        let data = children
            .get(DATA)?
            .ok_or_else(|| vortex_error::vortex_err!("Zoned data child is absent"))?;
        if data.dtype() != plan.dtype() || data.row_count() != plan.row_count() {
            vortex_error::vortex_bail!("Zoned data child shape does not match the plan output");
        }
        Ok(())
    }

    fn execute(
        plan: &Plan<Self>,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        plan.data_plan()?.execute(ctx, row_range, mask)
    }

    fn child_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        match index {
            DATA => Cow::Borrowed("data"),
            ZONES => Cow::Borrowed("zones"),
            _ => Cow::Owned(format!("child[{index}]")),
        }
    }
}
