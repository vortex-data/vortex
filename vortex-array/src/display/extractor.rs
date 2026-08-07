// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use vortex_utils::tree::IndentedFormatter;
use vortex_utils::tree::TreeDisplayContext;
pub use vortex_utils::tree::TreeDisplayExtractor as TreeExtractor;

use crate::ArrayRef;
use crate::arrays::Chunked;

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
}

impl TreeDisplayContext<ArrayRef> for TreeContext {
    fn push_parent(&mut self, parent: &ArrayRef) {
        self.ancestor_sizes.push(if parent.is::<Chunked>() {
            None
        } else {
            Some(parent.nbytes())
        });
    }

    fn pop_parent(&mut self, _parent: &ArrayRef) {
        self.ancestor_sizes.pop();
    }
}
