// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::stats::ArrayStats;

/// A lazily-filtered array that stores the child array and a mask indicating which elements
/// are selected.
///
/// The `offset` and `len` fields allow O(1) slicing by representing a logical view into
/// the filtered elements without materializing the slice.
#[derive(Clone, Debug)]
pub struct FilterArray {
    pub(super) child: ArrayRef,
    pub(super) mask: Mask,
    /// Offset into the logical (filtered) index space.
    pub(super) offset: usize,
    /// Length in the logical (filtered) index space.
    pub(super) len: usize,
    pub(super) stats: ArrayStats,
}

impl FilterArray {
    pub fn new(child: ArrayRef, mask: Mask) -> Self {
        if child.len() != mask.len() {
            vortex_panic!(
                "FilterArray length mismatch: child array has length {} but mask has length {}",
                child.len(),
                mask.len()
            );
        }
        let len = mask.true_count();
        Self {
            child,
            mask,
            offset: 0,
            len,
            stats: ArrayStats::default(),
        }
    }

    /// Create a sliced view of this FilterArray.
    ///
    /// This is O(1) as it only adjusts the offset and length without copying data.
    pub(super) fn slice(&self, offset: usize, len: usize) -> Self {
        if offset + len > self.len {
            vortex_panic!(
                "FilterArray slice out of bounds: offset {} + len {} > array len {}",
                offset,
                len,
                self.len
            );
        }
        Self {
            child: self.child.clone(),
            mask: self.mask.clone(),
            offset: self.offset + offset,
            len,
            stats: ArrayStats::default(),
        }
    }

    /// The mask used to filter the child array.
    pub fn filter_mask(&self) -> &Mask {
        &self.mask
    }

    /// The offset into the logical (filtered) index space.
    pub fn offset(&self) -> usize {
        self.offset
    }
}
