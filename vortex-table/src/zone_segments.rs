// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Where a file's zone maps are: the segments of the `zones` child of each zoned layout.
//!
//! A scan prunes a file's zones on these segments before it reads any data. The manifest records
//! their byte ranges, by top-level column, so a planner can fetch the zone maps of every file it
//! keeps in one round of IO, for only the columns a filter reads.

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::Footer;
use vortex_layout::LayoutRef;
use vortex_layout::layouts::struct_::Struct;
use vortex_layout::layouts::zoned::LegacyStats;
use vortex_layout::layouts::zoned::Zoned;

/// One segment of a file's zone maps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZoneSegment {
    /// The top-level column whose zone maps the segment holds, empty when the file is not a
    /// struct.
    pub column: String,
    /// The segment's id in the file's segment map.
    pub segment: u32,
    /// The segment's first byte in the file.
    pub offset: u64,
    /// The segment's length in bytes.
    pub length: u32,
    /// The exponent of the alignment the segment's buffer needs.
    pub alignment: u8,
}

/// The zone-map segments of the file `footer` describes, in layout order.
pub fn zone_segments(footer: &Footer) -> VortexResult<Vec<ZoneSegment>> {
    let root = footer.layout();
    let mut columns: Vec<(String, LayoutRef)> = Vec::new();
    if root.is::<Struct>() {
        for (name, child) in root.child_names().zip(root.children()?) {
            columns.push((name.to_string(), child));
        }
    } else {
        columns.push((String::new(), Arc::clone(root)));
    }
    let segment_map = footer.segment_map();
    let mut segments = Vec::new();
    for (column, layout) in columns {
        for node in layout.depth_first_traversal() {
            let node = node?;
            if !(node.is::<Zoned>() || node.is::<LegacyStats>()) {
                continue;
            }
            let Some(zones) = node.slot(1)? else {
                continue;
            };
            for zone_node in zones.depth_first_traversal() {
                for id in zone_node?.segment_ids() {
                    let segment = *id;
                    let spec = segment_map.get(segment as usize).ok_or_else(|| {
                        vortex_err!("segment {segment} is not in the segment map")
                    })?;
                    segments.push(ZoneSegment {
                        column: column.clone(),
                        segment,
                        offset: spec.offset,
                        length: spec.length,
                        alignment: spec.alignment.exponent(),
                    });
                }
            }
        }
    }
    Ok(segments)
}
