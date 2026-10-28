// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conservative last-consumer rows for the file's segment reads.

#[cfg(test)]
mod tests;

use std::ops::Range;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::request::IoTarget;
use vortex_scan::planning::driver::ReadLifetime;
use vortex_scan::planning::driver::ReadLifetimes;
use vortex_utils::aliases::hash_map::HashMap;

use super::SegmentLocation;
use crate::LayoutRef;

/// Index the stored layout once, independently of the scan's selection and optimized plans.
/// Auxiliary descendants all inherit their containing root-row region: dictionary value row
/// numbers and zone-table row numbers cannot be compared to the execution watermark.
pub(crate) fn read_lifetimes(
    layout: &LayoutRef,
    locations: &[SegmentLocation],
) -> VortexResult<ReadLifetimes> {
    let mut lifetimes = HashMap::default();
    visit(
        layout,
        0..layout.row_count(),
        true,
        locations,
        &mut lifetimes,
    )?;
    Ok(Arc::new(lifetimes))
}

fn visit(
    layout: &LayoutRef,
    rows: Range<u64>,
    mapped: bool,
    locations: &[SegmentLocation],
    lifetimes: &mut HashMap<IoTarget, ReadLifetime>,
) -> VortexResult<()> {
    for segment in layout.segment_ids() {
        let location = locations
            .get(*segment as usize)
            .ok_or_else(|| vortex_err!("segment {segment} has no known location"))?;
        let lifetime = ReadLifetime::ThroughRow(rows.end);
        lifetimes
            .entry(location.target())
            .and_modify(|current| *current = (*current).max(lifetime))
            .or_insert(lifetime);
    }
    for slot in 0..layout.nslots() {
        let Some(child) = layout.slot(slot)? else {
            continue;
        };
        let offset = layout.slot_type(slot).and_then(|kind| kind.row_offset());
        let (child_rows, child_mapped) = match (mapped, offset) {
            (true, Some(offset)) => {
                let start = rows
                    .start
                    .checked_add(offset)
                    .ok_or_else(|| vortex_err!("layout row offset overflow"))?;
                let end = start
                    .checked_add(child.row_count())
                    .ok_or_else(|| vortex_err!("layout row end overflow"))?;
                (start..end, true)
            }
            _ => (rows.clone(), false),
        };
        visit(&child, child_rows, child_mapped, locations, lifetimes)?;
    }
    Ok(())
}
