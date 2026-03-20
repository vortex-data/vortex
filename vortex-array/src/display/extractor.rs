// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

/// Trait for contributing display information to tree nodes.
///
/// Each extractor represents one "dimension" of display (e.g., nbytes, stats, metadata, buffers).
/// Extractors are composable: you can combine any number of them via [`TreeDisplay::with`].
///
/// [`TreeDisplay::with`]: super::TreeDisplay::with
pub trait TreeExtractor: Send + Sync {
    /// Annotations appended to the header line (e.g., `nbytes=10 B (100.00%)`).
    fn header_annotations(&self, array: &dyn DynArray, ctx: &TreeContext) -> Vec<String> {
        let _ = (array, ctx);
        vec![]
    }

    /// Additional detail lines shown below the header (e.g., `metadata: EmptyMetadata`).
    fn detail_lines(&self, array: &dyn DynArray, ctx: &TreeContext) -> Vec<String> {
        let _ = (array, ctx);
        vec![]
    }
}
