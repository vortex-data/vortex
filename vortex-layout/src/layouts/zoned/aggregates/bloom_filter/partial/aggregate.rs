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
        self.blocks()
            .as_flattened()
            .iter()
            .all(|split| *split == u32::MAX)
    }

    /// Merges a compatible partial into this one.
    ///
    /// The merge is a bitwise OR, which represents the union of two split-block
    /// Bloom filters when they use the same block count. `other` is only read, so a partial
    /// parsed from a stored filter merges straight out of the scalar's bytes.
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

        // One flat `u32` slice each: the compiler turns the flat OR into wide loads and stores,
        // where a loop per block spends about as many instructions on the block loop as on the
        // OR itself.
        let dst = self.blocks_mut().as_flattened_mut();
        let src = other.blocks().as_flattened();
        for (dst_split, src_split) in dst.iter_mut().zip(src) {
            *dst_split |= *src_split;
        }

        Ok(())
    }
}
