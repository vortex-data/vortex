// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared segment tree collection logic used by both the TUI browse view and the CLI segments command.

use std::sync::Arc;

use vortex::dtype::FieldName;
use vortex::error::VortexResult;
use vortex::file::SegmentSpec;
use vortex::layout::Layout;
use vortex::layout::LayoutChildType;
use vortex::utils::aliases::hash_map::HashMap;

/// Information about a single segment for display purposes.
pub struct SegmentDisplay {
    /// Name of the segment (e.g., "data", `[0]`, "zones")
    pub name: FieldName,
    /// The underlying segment specification
    pub spec: SegmentSpec,
    /// Row offset within the file
    pub row_offset: u64,
    /// Number of rows in this segment
    pub row_count: u64,
}

/// A tree of segments organized by field name.
pub struct SegmentTree {
    /// Map from field name to list of segments for that field
    pub segments: HashMap<FieldName, Vec<SegmentDisplay>>,
    /// Ordered list of field names (columns) in display order
    pub segment_ordering: Vec<FieldName>,
}

/// Collect segment tree from a layout and segment map.
pub fn collect_segment_tree(
    root_layout: &dyn Layout,
    segments: &Arc<[SegmentSpec]>,
) -> SegmentTree {
    let mut tree = SegmentTree {
        segments: HashMap::new(),
        segment_ordering: Vec::new(),
    };
    // Ignore errors during traversal - we want to collect as much as possible
    drop(segments_by_name_impl(
        root_layout,
        None,
        None,
        Some(0),
        segments,
        &mut tree,
    ));
    tree
}

fn segments_by_name_impl(
    root: &dyn Layout,
    group_name: Option<FieldName>,
    name: Option<FieldName>,
    row_offset: Option<u64>,
    segments: &Arc<[SegmentSpec]>,
    segment_tree: &mut SegmentTree,
) -> VortexResult<()> {
    // Recurse into children
    for (child, child_type) in root.children()?.into_iter().zip(root.child_types()) {
        match child_type {
            LayoutChildType::Transparent(sub_name) => segments_by_name_impl(
                child.as_ref(),
                group_name.clone(),
                Some(
                    name.as_ref()
                        .map(|n| format!("{n}.{sub_name}").into())
                        .unwrap_or_else(|| sub_name.into()),
                ),
                row_offset,
                segments,
                segment_tree,
            )?,
            LayoutChildType::Auxiliary(aux_name) => segments_by_name_impl(
                child.as_ref(),
                group_name.clone(),
                Some(
                    name.as_ref()
                        .map(|n| format!("{n}.{aux_name}").into())
                        .unwrap_or_else(|| aux_name.into()),
                ),
                Some(0),
                segments,
                segment_tree,
            )?,
            LayoutChildType::Chunk((idx, chunk_row_offset)) => segments_by_name_impl(
                child.as_ref(),
                group_name.clone(),
                Some(
                    name.as_ref()
                        .map(|n| format!("{n}.[{idx}]"))
                        .unwrap_or_else(|| format!("[{idx}]"))
                        .into(),
                ),
                // Compute absolute row offset.
                Some(chunk_row_offset + row_offset.unwrap_or(0)),
                segments,
                segment_tree,
            )?,
            LayoutChildType::Field(field_name) => {
                // Step into a new group name
                let new_group_name = group_name
                    .as_ref()
                    .map(|n| format!("{n}.{field_name}").into())
                    .unwrap_or_else(|| field_name);
                segment_tree.segment_ordering.push(new_group_name.clone());

                segments_by_name_impl(
                    child.as_ref(),
                    Some(new_group_name),
                    None,
                    row_offset,
                    segments,
                    segment_tree,
                )?
            }
        }
    }

    let current_segments = segment_tree
        .segments
        .entry(group_name.unwrap_or_else(|| FieldName::from("root")))
        .or_default();

    for segment_id in root.segment_ids() {
        let segment_spec = segments[*segment_id as usize].clone();
        current_segments.push(SegmentDisplay {
            name: name.clone().unwrap_or_else(|| "<unnamed>".into()),
            spec: segment_spec,
            row_count: root.row_count(),
            row_offset: row_offset.unwrap_or(0),
        })
    }

    Ok(())
}
