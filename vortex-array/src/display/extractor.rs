// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Write;

use crate::DynArray;

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

/// A formatter wrapper that automatically prepends indentation at the start of each line.
pub(crate) struct IndentedFormatter<'a, 'b> {
    inner: &'a mut fmt::Formatter<'b>,
    indent: &'a str,
    at_line_start: bool,
}

impl<'a, 'b> IndentedFormatter<'a, 'b> {
    pub(crate) fn new(f: &'a mut fmt::Formatter<'b>, indent: &'a str) -> Self {
        Self {
            inner: f,
            indent,
            at_line_start: true,
        }
    }
}

impl Write for IndentedFormatter<'_, '_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let mut parts = s.split('\n');

        if let Some(first) = parts.next()
            && !first.is_empty()
        {
            if self.at_line_start {
                self.inner.write_str(self.indent)?;
                self.at_line_start = false;
            }
            self.inner.write_str(first)?;
        }

        for part in parts {
            self.inner.write_char('\n')?;
            self.at_line_start = true;
            if !part.is_empty() {
                self.inner.write_str(self.indent)?;
                self.at_line_start = false;
                self.inner.write_str(part)?;
            }
        }

        Ok(())
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
        array: &dyn DynArray,
        ctx: &TreeContext,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let _ = (array, ctx, f);
        Ok(())
    }

    /// Write detail lines below the header.
    ///
    /// The caller handles indentation — extractors just write their content directly.
    fn write_details(
        &self,
        array: &dyn DynArray,
        ctx: &TreeContext,
        f: &mut dyn Write,
    ) -> fmt::Result {
        let _ = (array, ctx, f);
        Ok(())
    }
}
