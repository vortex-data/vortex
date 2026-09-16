// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregation helpers for Split Block Bloom Filters (SBBFs).
//!
//! This module contains implementation details for the operations
//! required by `AggregateFnVTable`.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use super::BloomPartial;

/// Practical implementation to avoid having to share blocks
impl BloomPartial {
    /// Returns true if all the blocks are saturated, in other words,
    /// all bits are `1`.
    #[inline]
    pub(in crate::layouts::zoned) fn is_saturated(&self) -> bool {
        self.blocks.iter().all(|byte| *byte == [u32::MAX; 8])
    }

    /// Merges a compatible partial into this one.
    ///
    /// The merge is a bitwise OR, which represents the union of two split-block
    /// Bloom filters when they use the same block count.
    ///
    /// _Notice_ This method only validates the block count.
    /// Merging a filter created with a different hash function
    /// will produce an invalid filter and introduce false negatives.
    #[inline]
    pub(in crate::layouts::zoned) fn union(&mut self, other: &BloomPartial) -> VortexResult<()> {
        vortex_ensure_eq!(
            self.len(),
            other.len(),
            "bloom partial block count mismatch"
        );

        for (dst_block, src_block) in self.blocks.iter_mut().zip(&other.blocks) {
            for (dst_split, src_split) in dst_block.iter_mut().zip(src_block) {
                *dst_split |= *src_split;
            }
        }

        Ok(())
    }
}
