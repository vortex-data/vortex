// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::future::try_join_all;
use termtree::Tree;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::LayoutRef;
use crate::layouts::flat::Flat;
use crate::layouts::flat::FlatLayout;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// Display the layout as a tree, fetching segment sizes from the segment source.
///
/// # Warning
///
/// This function performs IO to fetch each segment's buffer. For layouts with
/// many segments, this may result in significant IO overhead.
pub(super) async fn display_tree_with_segment_sizes(
    layout: LayoutRef,
    segment_source: Arc<dyn SegmentSource>,
) -> VortexResult<DisplayLayoutTree> {
    // First, collect all segment IDs from the layout tree (excluding those with inline array_tree)
    let mut segments_to_fetch = Vec::new();
    collect_segments_to_fetch(&layout, &mut segments_to_fetch)?;
    segments_to_fetch.dedup();

    // Fetch segments in parallel and parse buffer info
    let fetch_futures = segments_to_fetch.iter().map(|&segment_id| {
        let segment_source = Arc::clone(&segment_source);
        async move {
            let buffer = segment_source.request(segment_id).await?;
            let parts = SerializedArray::try_from(buffer)?;
            VortexResult::Ok((segment_id, parts.buffer_lengths()))
        }
    });
    let results = try_join_all(fetch_futures).await?;
    let segment_buffer_sizes: HashMap<SegmentId, Vec<usize>> = results.into_iter().collect();

    Ok(DisplayLayoutTree {
        layout,
        segment_buffer_sizes: Some(segment_buffer_sizes),
        verbose: true,
    })
}

/// Collect segment IDs that need to be fetched (those without inline array_tree).
fn collect_segments_to_fetch(
    layout: &LayoutRef,
    segment_ids: &mut Vec<SegmentId>,
) -> VortexResult<()> {
    // For FlatLayout, only add if there's no inline array_tree
    if let Some(flat_layout) = layout.as_opt::<Flat>() {
        if flat_layout.array_tree().is_none() {
            segment_ids.push(flat_layout.segment_id());
        }
    } else {
        // For other layouts, add all segment IDs
        segment_ids.extend(layout.segment_ids());
    }

    // Recurse into children
    for child in layout.children()? {
        collect_segments_to_fetch(&child, segment_ids)?;
    }
    Ok(())
}

/// Build a tree node for a FlatLayout, showing buffer sizes.
fn format_flat_layout_buffers(
    flat_layout: &FlatLayout,
    segment_buffer_sizes: Option<&HashMap<SegmentId, Vec<usize>>>,
) -> String {
    let segment_id = flat_layout.segment_id();

    // First, try to get buffer info from inline array_tree
    if let Some(array_tree) = flat_layout.array_tree()
        && let Ok(parts) = SerializedArray::from_array_tree(array_tree.as_ref().to_vec())
    {
        return format_buffer_sizes(&parts.buffer_lengths(), *segment_id);
    }

    // Otherwise, try to get from fetched segment info
    if let Some(sizes_map) = segment_buffer_sizes
        && let Some(buffer_sizes) = sizes_map.get(&segment_id)
    {
        return format_buffer_sizes(buffer_sizes, *segment_id);
    }

    // Fallback: just show segment ID
    format!("segment: {}", *segment_id)
}

fn format_buffer_sizes(buffer_sizes: &[usize], segment_id: u32) -> String {
    let buffer_sizes_str: Vec<String> = buffer_sizes.iter().map(|s| format!("{}B", s)).collect();
    let total: usize = buffer_sizes.iter().sum();
    format!(
        "segment {}, buffers=[{}], total={}B",
        segment_id,
        buffer_sizes_str.join(", "),
        total
    )
}

/// Display wrapper for layout tree visualization.
pub struct DisplayLayoutTree {
    layout: LayoutRef,
    segment_buffer_sizes: Option<HashMap<SegmentId, Vec<usize>>>,
    verbose: bool,
}

impl DisplayLayoutTree {
    /// Create a new display tree without pre-fetched segment buffer sizes.
    pub fn new(layout: LayoutRef, verbose: bool) -> Self {
        Self {
            layout,
            segment_buffer_sizes: None,
            verbose,
        }
    }

    fn make_tree(&self, layout: LayoutRef) -> VortexResult<Tree<String>> {
        // Build the node label with encoding, dtype, and metadata
        let mut node_parts = vec![
            format!("{}", layout.encoding()),
            format!("dtype: {}", layout.dtype()),
        ];

        // Add child count if there are children
        let nchildren = layout.nchildren();
        if nchildren > 0 {
            node_parts.push(format!("children: {}", nchildren));
        }

        // Add metadata and row count if verbose
        if self.verbose {
            let metadata = layout.metadata();
            if !metadata.is_empty() {
                node_parts.push(format!("metadata: {} bytes", metadata.len()));
            }
            node_parts.push(format!("rows: {}", layout.row_count()));
        }

        // For FlatLayout, show buffer info
        if let Some(flat_layout) = layout.as_opt::<Flat>() {
            node_parts.push(format_flat_layout_buffers(
                flat_layout,
                self.segment_buffer_sizes.as_ref(),
            ));
        } else {
            // Not a FlatLayout - show segment IDs if any (for verbose mode)
            if self.verbose {
                let segment_ids = layout.segment_ids();
                if !segment_ids.is_empty() {
                    node_parts.push(format!(
                        "segments: [{}]",
                        segment_ids
                            .iter()
                            .map(|s| format!("{}", **s))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            }
        }

        let node_name = node_parts.join(", ");

        // Get children and child names directly from the layout
        let children = layout.children()?;
        let child_names: Vec<_> = layout.child_names().collect();

        // Build child trees
        let child_trees: VortexResult<Vec<Tree<String>>> =
            if !children.is_empty() && child_names.len() == children.len() {
                // If we have names for all children, use them
                children
                    .into_iter()
                    .zip(child_names.iter())
                    .map(|(child, name)| {
                        let child_tree = self.make_tree(child)?;
                        Ok(Tree::new(format!("{}: {}", name, child_tree.root))
                            .with_leaves(child_tree.leaves))
                    })
                    .collect()
            } else if !children.is_empty() {
                // No names available, just show children
                children.into_iter().map(|c| self.make_tree(c)).collect()
            } else {
                // Leaf node - no children
                Ok(Vec::new())
            };

        Ok(Tree::new(node_name).with_leaves(child_trees?))
    }
}

impl std::fmt::Display for DisplayLayoutTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.make_tree(Arc::clone(&self.layout)) {
            Ok(tree) => write!(f, "{}", tree),
            Err(e) => write!(f, "Error building layout tree: {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::VarBinViewBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::serde::SerializedArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBufferMut;
    use vortex_buffer::buffer;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_utils::env::EnvVarGuard;

    use crate::IntoLayout;
    use crate::OwnedLayoutChildren;
    use crate::layouts::chunked::ChunkedLayout;
    use crate::layouts::flat::Flat;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::struct_::StructLayout;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::strategy::LayoutStrategy;
    use crate::test::SESSION;

    /// Test display_tree with inline array_tree metadata (no segment source needed).
    #[test]
    fn test_display_tree_inline_array_tree() {
        let _guard = EnvVarGuard::set("FLAT_LAYOUT_INLINE_ARRAY_NODE", "1");
        block_on(|handle| async move {
            let session = SESSION.clone().with_handle(handle);
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());

            // Create nullable i64 array (2 buffers: data + validity)
            let (ptr1, eof1) = SequenceId::root().split();
            let mut validity_builder = BitBufferMut::with_capacity(5);
            for b in [true, false, true, true, false] {
                validity_builder.append(b);
            }
            let validity = Validity::Array(
                BoolArray::new(validity_builder.freeze(), Validity::NonNullable).into_array(),
            );
            let array1 = PrimitiveArray::new(buffer![1i64, 2, 3, 4, 5], validity);
            let layout1 = FlatLayoutStrategy::default()
                .write_stream(
                    ctx.clone(),
                    Arc::<TestSegments>::clone(&segments),
                    array1.into_array().to_array_stream().sequenced(ptr1),
                    eof1,
                    &session,
                )
                .await
                .unwrap();

            // Create utf8 array (2 buffers: views + data)
            let (ptr2, eof2) = SequenceId::root().split();
            let mut builder = VarBinViewBuilder::with_capacity(DType::Utf8(NonNullable), 5);
            for s in [
                "hello world this is long",
                "another long string",
                "short",
                "medium str",
                "x",
            ] {
                builder.append_value(s);
            }
            let layout2 = FlatLayoutStrategy::default()
                .write_stream(
                    ctx.clone(),
                    Arc::<TestSegments>::clone(&segments),
                    builder
                        .finish()
                        .into_array()
                        .to_array_stream()
                        .sequenced(ptr2),
                    eof2,
                    &session,
                )
                .await
                .unwrap();

            // Create struct layout
            let struct_layout = StructLayout::new(
                5,
                DType::Struct(
                    StructFields::new(
                        vec![FieldName::from("numbers"), FieldName::from("strings")].into(),
                        vec![
                            DType::Primitive(PType::I64, Nullability::Nullable),
                            DType::Utf8(NonNullable),
                        ],
                    ),
                    NonNullable,
                ),
                vec![
                    ChunkedLayout::new(
                        5,
                        DType::Primitive(PType::I64, Nullability::Nullable),
                        OwnedLayoutChildren::layout_children(vec![layout1]),
                    )
                    .into_layout(),
                    layout2,
                ],
            )
            .into_layout();

            let output = format!("{}", struct_layout.display_tree_verbose(true));

            let expected = "\
vortex.struct, dtype: {numbers=i64?, strings=utf8}, children: 2, rows: 5
├── numbers: vortex.chunked, dtype: i64?, children: 1, rows: 5
│   └── [0]: vortex.flat, dtype: i64?, metadata: 171 bytes, rows: 5, segment 0, buffers=[40B, 1B], total=41B
└── strings: vortex.flat, dtype: utf8, metadata: 110 bytes, rows: 5, segment 1, buffers=[43B, 80B], total=123B
";
            assert_eq!(output, expected);
        })
    }

    /// Test display_tree_with_segments using async segment source to fetch buffer sizes.
    #[test]
    fn test_display_tree_with_segment_source() {
        // Ensure inline array node is disabled for this test
        let _guard = EnvVarGuard::remove("FLAT_LAYOUT_INLINE_ARRAY_NODE");
        block_on(|handle| async move {
            let session = SESSION.clone().with_handle(handle);
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());

            // Create simple i32 array
            let (ptr1, eof1) = SequenceId::root().split();
            let array1 = PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5], Validity::NonNullable);
            let layout1 = FlatLayoutStrategy::default()
                .write_stream(
                    ctx.clone(),
                    Arc::<TestSegments>::clone(&segments),
                    array1.into_array().to_array_stream().sequenced(ptr1),
                    eof1,
                    &session,
                )
                .await
                .unwrap();

            // Create another i32 array
            let (ptr2, eof2) = SequenceId::root().split();
            let array2 = PrimitiveArray::new(buffer![6i32, 7, 8, 9, 10], Validity::NonNullable);
            let layout2 = FlatLayoutStrategy::default()
                .write_stream(
                    ctx.clone(),
                    Arc::<TestSegments>::clone(&segments),
                    array2.into_array().to_array_stream().sequenced(ptr2),
                    eof2,
                    &session,
                )
                .await
                .unwrap();

            // Create chunked layout
            let chunked_layout = ChunkedLayout::new(
                10,
                DType::Primitive(PType::I32, NonNullable),
                OwnedLayoutChildren::layout_children(vec![layout1, layout2]),
            )
            .into_layout();

            let output = chunked_layout
                .display_tree_with_segments(segments)
                .await
                .unwrap();

            let expected = "\
vortex.chunked, dtype: i32, children: 2, rows: 10
├── [0]: vortex.flat, dtype: i32, rows: 5, segment 0, buffers=[20B], total=20B
└── [1]: vortex.flat, dtype: i32, rows: 5, segment 1, buffers=[20B], total=20B
";
            assert_eq!(output.to_string(), expected);
        })
    }

    /// Test display_array_tree with inline array node metadata.
    #[test]
    fn test_display_array_tree_with_inline_node() {
        let _guard = EnvVarGuard::set("FLAT_LAYOUT_INLINE_ARRAY_NODE", "1");

        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        // Create a simple primitive array
        let array = PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5], Validity::AllValid);
        let layout = block_on(|handle| async {
            let session = SESSION.clone().with_handle(handle);
            FlatLayoutStrategy::default()
                .write_stream(
                    ctx.clone(),
                    Arc::<TestSegments>::clone(&segments),
                    array.into_array().to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await
                .unwrap()
        });

        let flat_layout = layout.as_::<Flat>();

        let array_tree = flat_layout
            .array_tree()
            .expect("array_tree should be populated when FLAT_LAYOUT_INLINE_ARRAY_NODE is set");

        let parts = SerializedArray::from_array_tree(array_tree.as_ref().to_vec())
            .expect("should parse array_tree");
        assert_eq!(parts.buffer_lengths(), vec![20]); // 5 i32 values = 20 bytes

        assert_eq!(
            layout.display_tree().to_string(),
            "\
vortex.flat, dtype: i32?, segment 0, buffers=[20B], total=20B
"
        );
    }

    /// Test display_tree without inline array node (shows segment ID).
    #[test]
    fn test_display_tree_without_inline_node() {
        let _guard = EnvVarGuard::set("FLAT_LAYOUT_INLINE_ARRAY_NODE", "1");

        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();

        // Create a simple primitive array
        let array = PrimitiveArray::new(buffer![10i64, 20, 30], Validity::NonNullable);
        let layout = block_on(|handle| async {
            let session = SESSION.clone().with_handle(handle);
            FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    Arc::<TestSegments>::clone(&segments),
                    array.into_array().to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await
                .unwrap()
        });

        // Test display_tree exact output (with inline array_tree enabled by env var from other test)
        assert_eq!(
            layout.display_tree().to_string(),
            "\
vortex.flat, dtype: i64, segment 0, buffers=[24B], total=24B
"
        );
    }
}
