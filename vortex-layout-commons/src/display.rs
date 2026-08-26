// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::future::try_join_all;
use termtree::Tree;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::LayoutRef;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// Display the layout as a tree, fetching buffer sizes from the segment source.
///
/// # Warning
///
/// This function performs IO to fetch each referenced segment.
pub(crate) async fn display_tree_with_segment_sizes(
    layout: LayoutRef,
    segment_source: Arc<dyn SegmentSource>,
) -> VortexResult<DisplayLayoutTree> {
    let mut segment_ids = Vec::new();
    let mut segment_buffer_sizes = HashMap::new();
    collect_segment_ids(&layout, &mut segment_ids, &mut segment_buffer_sizes)?;
    segment_ids.sort_unstable();
    segment_ids.dedup();

    let fetches = segment_ids.into_iter().map(|segment_id| {
        let segment_source = Arc::clone(&segment_source);
        async move {
            let buffer = segment_source.request(segment_id).await?;
            let parts = SerializedArray::try_from(buffer)?;
            VortexResult::Ok((segment_id, parts.buffer_lengths()))
        }
    });
    segment_buffer_sizes.extend(try_join_all(fetches).await?);

    Ok(DisplayLayoutTree {
        layout,
        segment_buffer_sizes: Some(segment_buffer_sizes),
        verbose: true,
    })
}

fn collect_segment_ids(
    layout: &LayoutRef,
    segment_ids: &mut Vec<SegmentId>,
    segment_buffer_sizes: &mut HashMap<SegmentId, Vec<usize>>,
) -> VortexResult<()> {
    let inlined_sizes = layout.inlined_segment_buffer_sizes();
    segment_buffer_sizes.extend(inlined_sizes.iter().cloned());
    segment_ids.extend(
        layout
            .segment_ids()
            .into_iter()
            .filter(|id| !inlined_sizes.iter().any(|(inlined_id, _)| inlined_id == id)),
    );
    for child in layout.children()? {
        collect_segment_ids(&child, segment_ids, segment_buffer_sizes)?;
    }
    Ok(())
}

/// Display wrapper for a layout tree.
pub struct DisplayLayoutTree {
    layout: LayoutRef,
    segment_buffer_sizes: Option<HashMap<SegmentId, Vec<usize>>>,
    verbose: bool,
}

impl DisplayLayoutTree {
    /// Create a layout tree display without fetching segment data.
    pub fn new(layout: LayoutRef, verbose: bool) -> Self {
        Self {
            layout,
            segment_buffer_sizes: None,
            verbose,
        }
    }

    fn make_tree(&self, layout: LayoutRef) -> VortexResult<Tree<String>> {
        let mut node_parts = vec![
            layout.encoding_id().to_string(),
            format!("dtype: {}", layout.dtype()),
        ];

        if layout.nchildren() > 0 {
            node_parts.push(format!("children: {}", layout.nchildren()));
        }

        if self.verbose {
            let metadata = layout.metadata();
            if !metadata.is_empty() {
                node_parts.push(format!("metadata: {} bytes", metadata.len()));
            }
            node_parts.push(format!("rows: {}", layout.row_count()));
        }

        let segments = layout.segment_ids();
        if segments.len() == 1 {
            let segment_id = segments[0];
            let inlined_sizes = layout.inlined_segment_buffer_sizes();
            if let Some(buffer_sizes) = self
                .segment_buffer_sizes
                .as_ref()
                .and_then(|sizes| sizes.get(&segment_id))
                .or_else(|| {
                    inlined_sizes
                        .iter()
                        .find_map(|(id, sizes)| (*id == segment_id).then_some(sizes))
                })
            {
                node_parts.push(format_buffer_sizes(buffer_sizes, *segment_id));
            } else if !self.verbose {
                node_parts.push(format!("segment: {}", *segment_id));
            } else {
                node_parts.push(format!("segments: [{}]", *segment_id));
            }
        } else if !segments.is_empty() && self.verbose {
            node_parts.push(format!(
                "segments: [{}]",
                segments
                    .iter()
                    .map(|segment| (**segment).to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        let children = layout.children()?;
        let child_names = layout.child_names().collect::<Vec<_>>();
        let child_trees = if child_names.len() == children.len() {
            children
                .into_iter()
                .zip(child_names)
                .map(|(child, name)| {
                    let child_tree = self.make_tree(child)?;
                    Ok(Tree::new(format!("{name}: {}", child_tree.root))
                        .with_leaves(child_tree.leaves))
                })
                .collect::<VortexResult<Vec<_>>>()?
        } else {
            children
                .into_iter()
                .map(|child| self.make_tree(child))
                .collect::<VortexResult<Vec<_>>>()?
        };

        Ok(Tree::new(node_parts.join(", ")).with_leaves(child_trees))
    }
}

fn format_buffer_sizes(buffer_sizes: &[usize], segment_id: u32) -> String {
    let sizes = buffer_sizes
        .iter()
        .map(|size| format!("{size}B"))
        .collect::<Vec<_>>()
        .join(", ");
    let total = buffer_sizes.iter().sum::<usize>();
    format!("segment {segment_id}, buffers=[{sizes}], total={total}B")
}

impl std::fmt::Display for DisplayLayoutTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.make_tree(Arc::clone(&self.layout)) {
            Ok(tree) => write!(f, "{tree}"),
            Err(error) => write!(f, "Error building layout tree: {error}"),
        }
    }
}
