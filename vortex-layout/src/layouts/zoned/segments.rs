// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Finds the segments that hold a layout tree's zone maps.

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_utils::aliases::hash_set::HashSet;

use crate::LayoutRef;
use crate::VTable;
use crate::layouts::zoned::LegacyStats;
use crate::layouts::zoned::Zoned;
use crate::segments::SegmentId;

/// The slot of a zoned layout that holds its zone map. Slot 0 holds its rows.
const ZONES_SLOT: usize = 1;

/// The segments of every zone map in the layout tree of `root`: each segment under the zones
/// child of a [`Zoned`] or [`LegacyStats`] layout.
///
/// Zone maps hold statistics, never row values, so a caller can keep these segments to prune
/// later scans of the file without keeping its data.
pub fn zone_map_segments(root: &LayoutRef) -> VortexResult<HashSet<SegmentId>> {
    let (zoned, legacy) = (Zoned.id(), LegacyStats.id());
    let mut segments = HashSet::default();
    let mut pending = vec![(Arc::clone(root), false)];

    while let Some((layout, in_zone_map)) = pending.pop() {
        if in_zone_map {
            segments.extend(layout.segment_ids());
        }

        let is_zoned = layout.encoding_id() == zoned || layout.encoding_id() == legacy;
        for slot in 0..layout.nslots() {
            if let Some(child) = layout.slot(slot)? {
                pending.push((child, in_zone_map || (is_zoned && slot == ZONES_SLOT)));
            }
        }
    }

    Ok(segments)
}
