// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selection-specific planning without changing the pipeline or planner contracts.

use std::ops::Range;

use vortex_array::EmptyMetadata;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

use super::Plan;
use super::PlanId;
use super::PlanParts;
use super::PlanRef;
use super::PlanVTable;
use super::pipeline::Chain;
use super::pipeline::Compiler;

#[derive(Clone, Debug)]
struct Selected;

#[derive(Clone, Debug)]
struct SelectedData {
    rows: Range<u64>,
    mask: Mask,
}

/// Binds a projection's selection before segment discovery can initialize unselected children.
/// Whole-value dependencies retain their original plans and cache identities.
pub(crate) fn selected(plan: PlanRef, rows: Range<u64>, mask: Mask) -> PlanRef {
    if mask.all_true() {
        return plan;
    }
    PlanParts {
        vtable: Selected,
        dtype: plan.dtype().clone(),
        row_count: plan.row_count(),
        children: vec![plan].into(),
        data: SelectedData { rows, mask },
    }
    .into_typed()
    .into_plan()
}

impl PlanVTable for Selected {
    type PlanData = SelectedData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.selected");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        None
    }

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        let selected = plan.data().mask(&rows)?;
        let mask = mask & &selected;
        if mask.all_false() {
            return Ok(None);
        }
        compiler.compile_rows(&plan.child_required(0)?, rows, &mask)
    }
}

impl SelectedData {
    fn mask(&self, rows: &Range<u64>) -> VortexResult<Mask> {
        vortex_ensure!(
            rows.start >= self.rows.start && rows.end <= self.rows.end,
            "Selected plan requested outside its bound row range"
        );
        Ok(self.mask.slice(
            usize::try_from(rows.start - self.rows.start)?
                ..usize::try_from(rows.end - self.rows.start)?,
        ))
    }
}

/// Trims a child range to selected rows, without allocating a mask for an empty or contiguous
/// selection. The returned range remains relative to the input mask.
pub(crate) fn selected_slice(mask: &Mask, rows: Range<usize>) -> Option<(Range<usize>, Mask)> {
    if rows.is_empty() {
        return None;
    }
    match mask {
        Mask::AllFalse(_) => None,
        Mask::AllTrue(_) => {
            let len = rows.end - rows.start;
            Some((rows, Mask::new_true(len)))
        }
        Mask::Values(values) => {
            let bits = values.bit_buffer().slice(rows.clone());
            let first = bits.set_indices().next()?;
            let end = bits.last_set_index()? + 1;
            if rows.start + first == 0 && rows.start + end == mask.len() {
                // Keep the derived indices shared across fields when no rows were trimmed.
                return Some((0..mask.len(), mask.clone()));
            }
            let len = end - first;
            let selected = if bits.count_range(first, end) == len {
                Mask::new_true(len)
            } else {
                Mask::from_buffer(bits.slice(first..end))
            };
            Some((rows.start + first..rows.start + end, selected))
        }
    }
}
