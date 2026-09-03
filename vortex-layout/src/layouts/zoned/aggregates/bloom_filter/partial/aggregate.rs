// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregation helpers for Split Block Bloom Filters (SBBFs).
//!
//! This module contains implementation details for the operations
//! required by `AggregateFnVTable`.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

use super::BLOCK_SIZE;
use super::BYTES_PER_SPLIT;
use super::BloomPartial;

/// Practical implementation to avoid having to share blocks
impl BloomPartial {
    /// Resets and empties all blocks.
    #[inline]
    pub(in crate::layouts::zoned) fn reset(&mut self) {
        self.blocks.fill([0; 8]);
    }

    /// Returns true if all the blocks are saturated, in other words,
    /// all bits are `1`.
    #[inline]
    pub(in crate::layouts::zoned) fn is_saturated(&self) -> bool {
        self.blocks.iter().all(|byte| *byte == [u32::MAX; 8])
    }

    /// Merges a compatible serialized Bloom filter into this partial.
    ///
    /// The merge is a bitwise OR, which represents the union of two split-block
    /// Bloom filters when they use the same block count.
    ///
    /// _Notice_ This method only validates the byte length.
    /// Merging bytes from a filter created with a different hash function
    /// will produce an invalid filter and introduce false negatives.
    #[inline]
    pub(in crate::layouts::zoned) fn merge(&mut self, other: &[u8]) -> VortexResult<()> {
        // Partial returns size in blocks,
        // while bytes contains len in amount of bytes.
        // So blocks * block_size (bytes) = total amount of bytes
        vortex_ensure_eq!(
            self.len() * BLOCK_SIZE,
            other.len(),
            "bloom partial block count mismatch"
        );

        for (dst_block, src_block) in self
            .blocks
            .iter_mut()
            .zip(other.as_chunks::<BLOCK_SIZE>().0)
        {
            for (dst_split, src_split) in dst_block
                .iter_mut()
                .zip(src_block.as_chunks::<BYTES_PER_SPLIT>().0)
            {
                *dst_split |= u32::from_le_bytes(*src_split);
            }
        }

        Ok(())
    }
}
