// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use crate::ArrayRef;
use crate::arrays::Chunked;
use crate::display::DisplayOptions;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;
use crate::display::extractors::BufferExtractor;
use crate::display::extractors::MetadataExtractor;
use crate::display::extractors::NbytesExtractor;
use crate::display::extractors::StatsExtractor;
use crate::display::node::DisplayNode;

/// Composable tree display builder.
///
/// Use [`tree_display()`][crate::DynArray::tree_display] for the default display with all
/// built-in extractors, or [`tree_display_builder()`][crate::DynArray::tree_display_builder]
/// to start with a blank slate and compose your own:
///
/// ```
/// # use vortex_array::IntoArray;
/// # use vortex_buffer::buffer;
/// use vortex_array::display::{NbytesExtractor, MetadataExtractor, BufferExtractor};
///
/// let array = buffer![0_i16, 1, 2, 3, 4].into_array();
///
/// // Default: all built-in extractors
/// let full = array.tree_display();
///
/// // Custom: pick only what you need
/// let custom = array.tree_display_builder()
///     .with(NbytesExtractor)
///     .with(MetadataExtractor);
/// ```
pub struct TreeDisplay {
    array: ArrayRef,
    extractors: Vec<Box<dyn TreeExtractor>>,
}

impl TreeDisplay {
    /// Create a new tree display for the given array with no extractors.
    ///
    /// With no extractors, only encoding headers and the tree structure are shown.
    /// Use [`Self::default_display`] for the standard set of all built-in extractors.
    pub fn new(array: ArrayRef) -> Self {
        Self {
            array,
            extractors: Vec::new(),
        }
    }

    /// Create a tree display with all built-in extractors: nbytes, stats, metadata, and buffers.
    pub fn default_display(array: ArrayRef) -> Self {
        Self::new(array)
            .with(NbytesExtractor)
            .with(StatsExtractor)
            .with(MetadataExtractor)
            .with(BufferExtractor { show_percent: true })
    }

    /// Add an extractor to the display pipeline.
    pub fn with<E: TreeExtractor + 'static>(mut self, extractor: E) -> Self {
        self.extractors.push(Box::new(extractor));
        self
    }

    /// Add a pre-boxed extractor to the display pipeline.
    pub fn with_boxed(mut self, extractor: Box<dyn TreeExtractor>) -> Self {
        self.extractors.push(extractor);
        self
    }

    /// Recursively build the display node tree.
    fn build_node(&self, name: &str, array: &ArrayRef, ctx: &mut TreeContext) -> DisplayNode {
        // Collect header annotations from all extractors
        let header_annotations: Vec<String> = self
            .extractors
            .iter()
            .flat_map(|e| e.header_annotations(array.as_ref(), ctx))
            .collect();

        // Collect detail lines from all extractors
        let detail_lines: Vec<String> = self
            .extractors
            .iter()
            .flat_map(|e| e.detail_lines(array.as_ref(), ctx))
            .collect();

        // Push context for children: chunked arrays reset the percentage root
        let child_size = if array.is::<Chunked>() {
            None
        } else {
            Some(array.nbytes())
        };
        ctx.push(child_size);

        // Recurse into children
        let children: Vec<DisplayNode> = array
            .children_names()
            .into_iter()
            .zip(array.children())
            .map(|(child_name, child)| self.build_node(&child_name, &child, ctx))
            .collect();

        ctx.pop();

        DisplayNode {
            name: name.to_string(),
            encoding_summary: format!("{}", array.display_as(DisplayOptions::MetadataOnly)),
            header_annotations,
            detail_lines,
            children,
        }
    }
}

impl fmt::Display for TreeDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut ctx = TreeContext::new();
        let root = self.build_node("root", &self.array, &mut ctx);
        root.render(f, "")
    }
}
