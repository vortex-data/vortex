// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use vortex_utils::tree::TreeDisplayAdapter;
use vortex_utils::tree::write_indented_tree;

use crate::ArrayRef;
use crate::display::extractor::IndentedFormatter;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;
use crate::display::extractors::BufferExtractor;
use crate::display::extractors::EncodingSummaryExtractor;
use crate::display::extractors::MetadataExtractor;
use crate::display::extractors::NbytesExtractor;
use crate::display::extractors::StatsExtractor;

/// Composable tree display builder.
///
/// Use `tree_display()` for the default display with all built-in extractors,
/// or `tree_display_builder()` to start with a blank slate and compose your own:
///
/// ```
/// # use vortex_array::IntoArray;
/// # use vortex_buffer::buffer;
/// use vortex_array::display::{EncodingSummaryExtractor, NbytesExtractor, MetadataExtractor, BufferExtractor};
///
/// let array = buffer![0_i16, 1, 2, 3, 4].into_array();
///
/// // Default: all built-in extractors
/// let full = array.tree_display();
///
/// // Custom: pick only what you need
/// let custom = array.tree_display_builder()
///     .with(EncodingSummaryExtractor)
///     .with(NbytesExtractor)
///     .with(MetadataExtractor);
/// ```
pub struct TreeDisplay {
    array: ArrayRef,
    extractors: Vec<Box<dyn TreeExtractor<ArrayRef, TreeContext>>>,
}

impl TreeDisplay {
    /// Create a new tree display for the given array with no extractors.
    ///
    /// With no extractors, only node names and the tree structure are shown.
    /// Use [`Self::default_display`] for the standard set of all built-in extractors.
    pub fn new(array: ArrayRef) -> Self {
        Self {
            array,
            extractors: Vec::new(),
        }
    }

    /// Create a tree display with all built-in extractors: encoding summary, nbytes, stats,
    /// metadata, and buffers.
    pub fn default_display(array: ArrayRef) -> Self {
        Self::new(array)
            .with(EncodingSummaryExtractor)
            .with(NbytesExtractor)
            .with(StatsExtractor)
            .with(MetadataExtractor)
            .with(BufferExtractor { show_percent: true })
    }

    /// Add an extractor to the display pipeline.
    pub fn with<E: TreeExtractor<ArrayRef, TreeContext> + 'static>(mut self, extractor: E) -> Self {
        self.extractors.push(Box::new(extractor));
        self
    }

    /// Add a pre-boxed extractor to the display pipeline.
    pub fn with_boxed(mut self, extractor: Box<dyn TreeExtractor<ArrayRef, TreeContext>>) -> Self {
        self.extractors.push(extractor);
        self
    }
}

impl TreeDisplayAdapter for TreeDisplay {
    type Context = TreeContext;
    type Node = ArrayRef;

    fn write_node(
        &self,
        array: &ArrayRef,
        ctx: &TreeContext,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        for extractor in &self.extractors {
            extractor.write_header(array, ctx, f)?;
        }
        Ok(())
    }

    fn write_details(
        &self,
        array: &ArrayRef,
        ctx: &TreeContext,
        f: &mut IndentedFormatter<'_, '_>,
    ) -> fmt::Result {
        for extractor in &self.extractors {
            extractor.write_details(array, ctx, f)?;
        }
        Ok(())
    }

    fn visit_children(
        &self,
        array: &ArrayRef,
        visit: &mut dyn FnMut(&str, &ArrayRef, bool) -> fmt::Result,
    ) -> fmt::Result {
        let mut children = array
            .children_names()
            .into_iter()
            .zip(array.children())
            .peekable();
        while let Some((child_name, child)) = children.next() {
            let is_last = children.peek().is_none();
            visit(&child_name, &child, is_last)?;
        }
        Ok(())
    }
}

impl fmt::Display for TreeDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut ctx = TreeContext::new();
        write_indented_tree(self, "root", &self.array, &mut ctx, f)
    }
}
