// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::future::try_join_all;
use vortex_array::display::tree_model::{Attr, AttrValue, DisplayTreeNode};
use vortex_array::serde::ArrayParts;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::LayoutRef;
use crate::layouts::flat::FlatLayout;
use crate::layouts::flat::FlatVTable;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// Options controlling which attributes are included in tree display.
#[derive(Debug, Clone, Default)]
pub struct TreeDisplayOptions {
    /// Include the dtype attribute.
    pub dtype: bool,
    /// Include the child count attribute.
    pub children_count: bool,
    /// Include metadata size in bytes.
    pub metadata_bytes: bool,
    /// Include row count.
    pub row_count: bool,
    /// Include segment IDs.
    pub segment_ids: bool,
    /// Include buffer sizes for flat layouts.
    pub buffer_sizes: bool,
}

impl TreeDisplayOptions {
    /// Create options that include all attributes.
    pub fn all() -> Self {
        Self {
            dtype: true,
            children_count: true,
            metadata_bytes: true,
            row_count: true,
            segment_ids: true,
            buffer_sizes: true,
        }
    }

    /// Create minimal options (just encoding name).
    pub fn minimal() -> Self {
        Self::default()
    }

    /// Create options suitable for verbose output.
    pub fn verbose() -> Self {
        Self::all()
    }

    /// Create options suitable for concise output.
    pub fn concise() -> Self {
        Self {
            dtype: true,
            children_count: true,
            metadata_bytes: false,
            row_count: false,
            segment_ids: true,
            buffer_sizes: true,
        }
    }

    /// Builder: include dtype.
    #[must_use]
    pub fn with_dtype(mut self) -> Self {
        self.dtype = true;
        self
    }

    /// Builder: include children count.
    #[must_use]
    pub fn with_children_count(mut self) -> Self {
        self.children_count = true;
        self
    }

    /// Builder: include metadata bytes.
    #[must_use]
    pub fn with_metadata_bytes(mut self) -> Self {
        self.metadata_bytes = true;
        self
    }

    /// Builder: include row count.
    #[must_use]
    pub fn with_row_count(mut self) -> Self {
        self.row_count = true;
        self
    }

    /// Builder: include segment IDs.
    #[must_use]
    pub fn with_segment_ids(mut self) -> Self {
        self.segment_ids = true;
        self
    }

    /// Builder: include buffer sizes.
    #[must_use]
    pub fn with_buffer_sizes(mut self) -> Self {
        self.buffer_sizes = true;
        self
    }
}

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
        let segment_source = segment_source.clone();
        async move {
            let buffer = segment_source.request(segment_id).await?;
            let parts = ArrayParts::try_from(buffer)?;
            VortexResult::Ok((segment_id, parts.buffer_lengths()))
        }
    });
    let results = try_join_all(fetch_futures).await?;
    let segment_buffer_sizes: HashMap<SegmentId, Vec<usize>> = results.into_iter().collect();

    Ok(DisplayLayoutTree {
        layout,
        segment_buffer_sizes: Some(segment_buffer_sizes),
        options: TreeDisplayOptions::verbose(),
    })
}

/// Collect segment IDs that need to be fetched (those without inline array_tree).
fn collect_segments_to_fetch(
    layout: &LayoutRef,
    segment_ids: &mut Vec<SegmentId>,
) -> VortexResult<()> {
    // For FlatLayout, only add if there's no inline array_tree
    if let Some(flat_layout) = layout.as_opt::<FlatVTable>() {
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

/// Add buffer size attributes to a tree node for a FlatLayout.
fn add_flat_layout_attrs(
    node: &mut DisplayTreeNode,
    flat_layout: &FlatLayout,
    segment_buffer_sizes: Option<&HashMap<SegmentId, Vec<usize>>>,
    options: &TreeDisplayOptions,
) {
    let segment_id = flat_layout.segment_id();

    // First, try to get buffer info from inline array_tree
    if let Some(array_tree) = flat_layout.array_tree()
        && let Ok(parts) = ArrayParts::from_array_tree(array_tree.as_ref().to_vec())
    {
        add_buffer_attrs(node, &parts.buffer_lengths(), *segment_id, options);
        return;
    }

    // Otherwise, try to get from fetched segment info
    if let Some(sizes_map) = segment_buffer_sizes
        && let Some(buffer_sizes) = sizes_map.get(&segment_id)
    {
        add_buffer_attrs(node, buffer_sizes, *segment_id, options);
        return;
    }

    // Fallback: just show segment ID
    if options.segment_ids {
        node.attrs.push(Attr::new("segment", *segment_id as u64));
    }
}

fn add_buffer_attrs(
    node: &mut DisplayTreeNode,
    buffer_sizes: &[usize],
    segment_id: u32,
    options: &TreeDisplayOptions,
) {
    if options.segment_ids {
        node.attrs.push(Attr::new("segment", segment_id as u64));
    }

    if options.buffer_sizes {
        // Format buffer sizes with "B" suffix for human-readable display
        let buffer_str = format!(
            "[{}]",
            buffer_sizes
                .iter()
                .map(|s| format!("{}B", s))
                .collect::<Vec<_>>()
                .join(", ")
        );
        node.attrs.push(Attr::new("buffers", buffer_str));

        let total: usize = buffer_sizes.iter().sum();
        node.attrs.push(Attr::new("total", format!("{}B", total)));
    }
}

/// Display wrapper for layout tree visualization.
///
/// This type provides both lazy text display (via [`Display`]) and eager JSON
/// serialization (via [`to_tree_node()`] + [`serde::Serialize`]).
///
/// - For text output, use `format!("{}", tree)` or `tree.to_string()`. The tree
///   is walked lazily during rendering.
/// - For JSON output, call `tree.to_tree_node()` to build an owned [`DisplayTreeNode`],
///   then serialize that.
///
/// # Example
///
/// ```ignore
/// let layout = /* ... */;
///
/// // Concise text display
/// println!("{}", layout.display_tree());
///
/// // Verbose text display
/// println!("{}", layout.display_tree_verbose(true));
///
/// // Custom options
/// let options = TreeDisplayOptions::minimal().with_dtype().with_row_count();
/// let tree = DisplayLayoutTree::with_options(layout, options);
/// println!("{}", tree);
///
/// // JSON output
/// let json = serde_json::to_string(&tree.to_tree_node()?)?;
/// ```
pub struct DisplayLayoutTree {
    layout: LayoutRef,
    segment_buffer_sizes: Option<HashMap<SegmentId, Vec<usize>>>,
    options: TreeDisplayOptions,
}

impl DisplayLayoutTree {
    /// Create a new display tree with default (concise) options.
    pub fn new(layout: LayoutRef, verbose: bool) -> Self {
        let options = if verbose {
            TreeDisplayOptions::verbose()
        } else {
            TreeDisplayOptions::concise()
        };
        Self {
            layout,
            segment_buffer_sizes: None,
            options,
        }
    }

    /// Create a new display tree with custom options.
    pub fn with_options(layout: LayoutRef, options: TreeDisplayOptions) -> Self {
        Self {
            layout,
            segment_buffer_sizes: None,
            options,
        }
    }

    /// Convert the layout tree to a [`DisplayTreeNode`] (eager).
    ///
    /// This walks the entire tree and builds an owned representation suitable
    /// for JSON serialization. For text display, prefer using `Display` directly
    /// as it walks the tree lazily.
    pub fn to_tree_node(&self) -> VortexResult<DisplayTreeNode> {
        self.make_tree_node(self.layout.clone())
    }

    fn make_tree_node(&self, layout: LayoutRef) -> VortexResult<DisplayTreeNode> {
        let mut node = DisplayTreeNode::new(layout.encoding().to_string());

        // Add inline attributes
        self.add_inline_attrs(&mut node, &layout);

        // For FlatLayout, show buffer info as inline attributes
        if let Some(flat_layout) = layout.as_opt::<FlatVTable>() {
            add_flat_layout_attrs(&mut node, flat_layout, self.segment_buffer_sizes.as_ref(), &self.options);
        } else if self.options.segment_ids {
            // Not a FlatLayout - show segment IDs if any
            let segment_ids = layout.segment_ids();
            if !segment_ids.is_empty() {
                let ids: Vec<AttrValue> =
                    segment_ids.iter().map(|s| AttrValue::UInt(**s as u64)).collect();
                node.nested_attrs.push(Attr::new("segments", AttrValue::List(ids)));
            }
        }

        // Build child nodes
        let children = layout.children()?;
        let child_names: Vec<_> = layout.child_names().collect();

        if !children.is_empty() && child_names.len() == children.len() {
            for (child, name) in children.into_iter().zip(child_names.iter()) {
                let child_node = self.make_tree_node(child)?;
                node.children.insert(name.to_string(), child_node);
            }
        } else if !children.is_empty() {
            for (i, child) in children.into_iter().enumerate() {
                let child_node = self.make_tree_node(child)?;
                node.children.insert(format!("[{}]", i), child_node);
            }
        }

        Ok(node)
    }

    /// Add inline attributes to a node for a layout.
    fn add_inline_attrs(&self, node: &mut DisplayTreeNode, layout: &LayoutRef) {
        if self.options.dtype {
            node.attrs.push(Attr::new("dtype", layout.dtype().to_string()));
        }

        if self.options.children_count {
            let nchildren = layout.nchildren();
            if nchildren > 0 {
                node.attrs.push(Attr::new("children", nchildren as u64));
            }
        }

        if self.options.metadata_bytes {
            let metadata = layout.metadata();
            if !metadata.is_empty() {
                node.attrs.push(Attr::new("metadata", format!("{} bytes", metadata.len())));
            }
        }

        if self.options.row_count {
            node.attrs.push(Attr::new("rows", layout.row_count()));
        }
    }

    /// Write a layout node and its children lazily during display.
    fn write_layout(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        layout: &LayoutRef,
        prefix: &str,
        child_name: Option<&str>,
    ) -> std::fmt::Result {
        // Write the node line
        if let Some(name) = child_name {
            write!(f, "{}: ", name)?;
        }
        write!(f, "{}", layout.encoding())?;

        // Write inline attrs based on options
        if self.options.dtype {
            write!(f, ", dtype: {}", layout.dtype())?;
        }

        if self.options.children_count {
            let nchildren = layout.nchildren();
            if nchildren > 0 {
                write!(f, ", children: {}", nchildren)?;
            }
        }

        if self.options.metadata_bytes {
            let metadata = layout.metadata();
            if !metadata.is_empty() {
                write!(f, ", metadata: {} bytes", metadata.len())?;
            }
        }

        if self.options.row_count {
            write!(f, ", rows: {}", layout.row_count())?;
        }

        // For FlatLayout, show buffer info
        if let Some(flat_layout) = layout.as_opt::<FlatVTable>() {
            self.write_flat_layout_attrs(f, flat_layout)?;
        }

        writeln!(f)?;

        // Write nested attrs for non-flat layouts (segment IDs)
        if layout.as_opt::<FlatVTable>().is_none() && self.options.segment_ids {
            let segment_ids = layout.segment_ids();
            if !segment_ids.is_empty() {
                let ids_str = segment_ids
                    .iter()
                    .map(|s| format!("{}", **s))
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(f, "{}    segments: [{}]", prefix, ids_str)?;
            }
        }

        // Write children
        let children = layout.children().map_err(|_| std::fmt::Error)?;
        let child_names: Vec<_> = layout.child_names().collect();
        let num_children = children.len();

        for (i, child) in children.into_iter().enumerate() {
            let is_last = i == num_children - 1;
            let connector = if is_last { "└── " } else { "├── " };
            let new_prefix = if is_last {
                format!("{}    ", prefix)
            } else {
                format!("{}│   ", prefix)
            };

            let name = if child_names.len() == num_children {
                child_names[i].to_string()
            } else {
                format!("[{}]", i)
            };

            write!(f, "{}{}", prefix, connector)?;
            self.write_layout(f, &child, &new_prefix, Some(&name))?;
        }

        Ok(())
    }

    /// Write flat layout buffer attributes.
    fn write_flat_layout_attrs(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        flat_layout: &FlatLayout,
    ) -> std::fmt::Result {
        let segment_id = flat_layout.segment_id();

        // Try inline array_tree first
        if let Some(array_tree) = flat_layout.array_tree()
            && let Ok(parts) = ArrayParts::from_array_tree(array_tree.as_ref().to_vec())
        {
            return self.write_buffer_info(f, &parts.buffer_lengths(), *segment_id);
        }

        // Try fetched segment info
        if let Some(sizes_map) = &self.segment_buffer_sizes
            && let Some(buffer_sizes) = sizes_map.get(&segment_id)
        {
            return self.write_buffer_info(f, buffer_sizes, *segment_id);
        }

        // Fallback: just segment ID
        if self.options.segment_ids {
            write!(f, ", segment: {}", *segment_id)?;
        }
        Ok(())
    }

    /// Write buffer size info.
    fn write_buffer_info(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        buffer_sizes: &[usize],
        segment_id: u32,
    ) -> std::fmt::Result {
        if self.options.segment_ids {
            write!(f, ", segment: {}", segment_id)?;
        }

        if self.options.buffer_sizes {
            let buffers_str = buffer_sizes
                .iter()
                .map(|s| format!("{}B", s))
                .collect::<Vec<_>>()
                .join(", ");
            let total: usize = buffer_sizes.iter().sum();
            write!(f, ", buffers: [{}], total: {}B", buffers_str, total)?;
        }

        Ok(())
    }
}

impl std::fmt::Display for DisplayLayoutTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_layout(f, &self.layout, "", None)
    }
}

/// Serialize the layout tree to JSON.
///
/// This implementation is gated behind the `serde` feature on `vortex-array`.
#[cfg(feature = "serde")]
impl serde::Serialize for DisplayLayoutTree {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self.to_tree_node() {
            Ok(node) => node.serialize(serializer),
            Err(e) => Err(serde::ser::Error::custom(format!(
                "Error building layout tree: {}",
                e
            ))),
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
    use vortex_array::serde::ArrayParts;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBufferMut;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::FieldName;
    use vortex_dtype::Nullability;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType;
    use vortex_dtype::StructFields;
    use vortex_io::runtime::single::block_on;
    use vortex_utils::env::EnvVarGuard;

    use crate::IntoLayout;
    use crate::OwnedLayoutChildren;
    use crate::layouts::chunked::ChunkedLayout;
    use crate::layouts::flat::FlatVTable;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::struct_::StructLayout;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::strategy::LayoutStrategy;

    /// Test display_tree with inline array_tree metadata (no segment source needed).
    #[test]
    fn test_display_tree_inline_array_tree() {
        let _guard = EnvVarGuard::set("FLAT_LAYOUT_INLINE_ARRAY_NODE", "1");
        block_on(|handle| async move {
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());

            // Create nullable i64 array (2 buffers: data + validity)
            let (ptr1, eof1) = SequenceId::root().split();
            let mut validity_builder = BitBufferMut::with_capacity(5);
            for b in [true, false, true, true, false] {
                validity_builder.append(b);
            }
            let validity = Validity::Array(
                BoolArray::from_bit_buffer(validity_builder.freeze(), Validity::NonNullable)
                    .into_array(),
            );
            let array1 = PrimitiveArray::new(buffer![1i64, 2, 3, 4, 5], validity);
            let layout1 = FlatLayoutStrategy::default()
                .write_stream(
                    ctx.clone(),
                    segments.clone(),
                    array1.to_array_stream().sequenced(ptr1),
                    eof1,
                    handle.clone(),
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
                    segments.clone(),
                    builder.finish().to_array_stream().sequenced(ptr2),
                    eof2,
                    handle.clone(),
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
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());

            // Create simple i32 array
            let (ptr1, eof1) = SequenceId::root().split();
            let array1 = PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5], Validity::NonNullable);
            let layout1 = FlatLayoutStrategy::default()
                .write_stream(
                    ctx.clone(),
                    segments.clone(),
                    array1.to_array_stream().sequenced(ptr1),
                    eof1,
                    handle.clone(),
                )
                .await
                .unwrap();

            // Create another i32 array
            let (ptr2, eof2) = SequenceId::root().split();
            let array2 = PrimitiveArray::new(buffer![6i32, 7, 8, 9, 10], Validity::NonNullable);
            let layout2 = FlatLayoutStrategy::default()
                .write_stream(
                    ctx.clone(),
                    segments.clone(),
                    array2.to_array_stream().sequenced(ptr2),
                    eof2,
                    handle.clone(),
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
            FlatLayoutStrategy::default()
                .write_stream(
                    ctx.clone(),
                    segments.clone(),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    handle,
                )
                .await
                .unwrap()
        });

        let flat_layout = layout.as_::<FlatVTable>();

        let array_tree = flat_layout
            .array_tree()
            .expect("array_tree should be populated when FLAT_LAYOUT_INLINE_ARRAY_NODE is set");

        let parts = ArrayParts::from_array_tree(array_tree.as_ref().to_vec())
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
            FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    segments.clone(),
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    handle,
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
