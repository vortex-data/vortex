// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use crate::ArrayRef;
use crate::arrays::Chunked;
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
    extractors: Vec<Box<dyn TreeExtractor>>,
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
    pub fn with<E: TreeExtractor + 'static>(mut self, extractor: E) -> Self {
        self.extractors.push(Box::new(extractor));
        self
    }

    /// Add a pre-boxed extractor to the display pipeline.
    pub fn with_boxed(mut self, extractor: Box<dyn TreeExtractor>) -> Self {
        self.extractors.push(extractor);
        self
    }

    /// Recursively write a node and all its descendants directly to the formatter.
    fn write_node(
        &self,
        name: &str,
        array: &ArrayRef,
        ctx: &mut TreeContext,
        indent: &str,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        // Header line: "{indent}{name}:{annotations...}\n"
        write!(f, "{indent}{name}:")?;
        for extractor in &self.extractors {
            extractor.write_header(array, ctx, f)?;
        }
        writeln!(f)?;

        // Detail lines
        let child_indent = format!("{indent}  ");
        {
            let mut indented = IndentedFormatter::new(f, &child_indent);
            for extractor in &self.extractors {
                extractor.write_details(array, ctx, &mut indented)?;
            }
        }

        // Push context for children: chunked arrays reset the percentage root
        let child_size = if array.is::<Chunked>() {
            None
        } else {
            Some(array.nbytes())
        };
        ctx.push(child_size);

        // Recurse into children
        for (child_name, child) in array.children_names().into_iter().zip(array.children()) {
            self.write_node(&child_name, &child, ctx, &child_indent, f)?;
        }

        ctx.pop();

        Ok(())
    }
}

impl fmt::Display for TreeDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut ctx = TreeContext::new();
        self.write_node("root", &self.array, &mut ctx, "", f)
    }
}
