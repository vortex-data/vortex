// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lowering native layouts into scan plans.

mod legacy_stats;
mod segment_scan;

use std::sync::Arc;

pub use legacy_stats::LegacyStats;
pub use legacy_stats::LegacyStatsData;
pub use legacy_stats::LegacyStatsPlan;
pub use segment_scan::SegmentScan;
pub use segment_scan::SegmentScanData;
pub use segment_scan::SegmentScanPlan;
use vortex_error::vortex_err;
use vortex_scan::plan::LazyPlanRef;
use vortex_scan::plan::PlanSlots;

use crate::LayoutRef;

/// One lazy plan slot per slot of `layout`, at the same position.
pub(crate) fn slots_from_layout(layout: &LayoutRef) -> PlanSlots {
    slots_from_layout_slots(layout, 0..layout.nslots())
}

/// One lazy plan slot per entry of `layout_slots`: plan slot `i` plans layout slot
/// `layout_slots[i]`.
///
/// This is where every layout maps its slots to plan slots. An absent layout slot is an
/// absent plan slot; presence comes from the layout's metadata, so nothing is read or planned
/// until a plan slot is first accessed.
pub(crate) fn slots_from_layout_slots(
    layout: &LayoutRef,
    layout_slots: impl IntoIterator<Item = usize>,
) -> PlanSlots {
    layout_slots
        .into_iter()
        .map(|slot| {
            layout.slot_type(slot)?;
            let layout = Arc::clone(layout);
            Some(LazyPlanRef::lazy(move || {
                let child = layout
                    .slot(slot)?
                    .ok_or_else(|| vortex_err!("Layout child slot {slot} is absent"))?;
                child.new_plan()
            }))
        })
        .collect()
}

#[cfg(test)]
mod tests;
