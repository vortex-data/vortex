// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conservative consumer row ranges for the file's segment reads.

#[cfg(test)]
mod tests;

use std::ops::Range;
use std::sync::Arc;

use vortex_array::dtype::FieldName;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::request::IoTarget;
use vortex_scan::planning::driver::ReadLifetime;
use vortex_scan::planning::driver::ReadLifetimes;
use vortex_utils::aliases::hash_map::HashMap;

use super::SegmentLocation;
use crate::LayoutChildType;
use crate::LayoutRef;

/// Index the stored layout, independently of the scan's selection and optimized plans.
/// Auxiliary descendants all inherit their containing root-row region: dictionary value row
/// numbers and zone-table row numbers cannot be compared to root-row claims.
///
/// With `fields`, only those fields of the root struct are indexed. Visiting a child layout
/// materializes it, so indexing every field of a wide file costs far more than a scan of a few of
/// them. A read the index leaves out lives until the file execution ends.
pub(crate) fn read_lifetimes(
    layout: &LayoutRef,
    locations: &[SegmentLocation],
    fields: Option<&[FieldName]>,
) -> VortexResult<ReadLifetimes> {
    let mut lifetimes = HashMap::default();
    visit(
        layout,
        0..layout.row_count(),
        true,
        fields,
        locations,
        &mut lifetimes,
    )?;
    Ok(Arc::new(lifetimes))
}

fn visit(
    layout: &LayoutRef,
    rows: Range<u64>,
    mapped: bool,
    fields: Option<&[FieldName]>,
    locations: &[SegmentLocation],
    lifetimes: &mut HashMap<IoTarget, ReadLifetime>,
) -> VortexResult<()> {
    for segment in layout.segment_ids() {
        let location = locations
            .get(*segment as usize)
            .ok_or_else(|| vortex_err!("segment {segment} has no known location"))?;
        lifetimes
            .entry(location.target())
            .and_modify(|current| {
                if let ReadLifetime::Rows(current) = current {
                    current.start = current.start.min(rows.start);
                    current.end = current.end.max(rows.end);
                }
            })
            .or_insert_with(|| ReadLifetime::Rows(rows.clone()));
    }
    for slot in 0..layout.nslots() {
        let kind = layout.slot_type(slot);
        // Below the root struct's fields, every descendant is indexed.
        let child_fields = match (&kind, fields) {
            (Some(LayoutChildType::Field(name)), Some(fields)) => {
                if !fields.contains(name) {
                    continue;
                }
                None
            }
            (Some(LayoutChildType::Field(_)), None) => None,
            _ => fields,
        };
        let Some(child) = layout.slot(slot)? else {
            continue;
        };
        let offset = kind.and_then(|kind| kind.row_offset());
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
        visit(
            &child,
            child_rows,
            child_mapped,
            child_fields,
            locations,
            lifetimes,
        )?;
    }
    Ok(())
}
