// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

use crate::ArrayRef;

/// Context threaded through tree traversal for percentage calculations etc.
pub struct TreeContext {
    /// Stack of ancestor nbytes values. `None` entries reset the percentage root
    /// (e.g. for chunked arrays where each chunk is its own root).
    pub(crate) ancestor_sizes: Vec<Option<u64>>,
}

impl TreeContext {
    pub(crate) fn new() -> Self {
        Self {
            ancestor_sizes: Vec::new(),
        }
    }

    /// The total size used as the denominator for percentage calculations.
    /// Returns `None` if there is no ancestor (i.e., this node is the root or
    /// a chunk boundary reset the percentage root).
    pub fn parent_total_size(&self) -> Option<u64> {
        self.ancestor_sizes.last().cloned().flatten()
    }

    pub(crate) fn push(&mut self, size: Option<u64>) {
        self.ancestor_sizes.push(size);
    }

    pub(crate) fn pop(&mut self) {
        self.ancestor_sizes.pop();
    }
}

/// Wrapper providing access to a [`fmt::Formatter`] and the current indentation string.
pub struct IndentedFormatter<'a, 'b> {
    inner: &'a mut fmt::Formatter<'b>,
    indent: &'a str,
}

impl<'a, 'b> IndentedFormatter<'a, 'b> {
    pub(crate) fn new(f: &'a mut fmt::Formatter<'b>, indent: &'a str) -> Self {
        Self { inner: f, indent }
    }

    /// Access the indent string and underlying [`fmt::Formatter`] together.
    pub fn parts(&mut self) -> (&str, &mut fmt::Formatter<'b>) {
        (self.indent, self.inner)
    }

    /// The current indentation string.
    pub fn indent(&self) -> &str {
        self.indent
    }

    /// Access the underlying [`fmt::Formatter`].
    pub fn formatter(&mut self) -> &mut fmt::Formatter<'b> {
        self.inner
    }
}

/// Trait for contributing display information to tree nodes.
///
/// Each extractor represents one "dimension" of display (e.g., nbytes, stats, metadata, buffers).
/// Extractors are composable: you can combine any number of them via [`TreeDisplay::with`].
///
/// [`TreeDisplay::with`]: super::TreeDisplay::with
pub trait TreeExtractor: Send + Sync {
    /// Write header annotations (space-prefixed) to the formatter.
    fn write_header(
        &self,
        array: &ArrayRef,
        ctx: &TreeContext,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let _ = (array, ctx, f);
        Ok(())
    }

    /// Write detail lines below the header.
    ///
    /// Content written through `f` is automatically indented. Use
    /// [`f.formatter()`](IndentedFormatter::formatter) to access the underlying
    /// [`fmt::Formatter`] for formatting flags.
    fn write_details(
        &self,
        array: &ArrayRef,
        ctx: &TreeContext,
        f: &mut IndentedFormatter<'_, '_>,
    ) -> fmt::Result {
        let _ = (array, ctx, f);
        Ok(())
    }
}
