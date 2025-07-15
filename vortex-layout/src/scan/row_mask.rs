// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::{BitAnd, Deref, Range};

use vortex_mask::Mask;

/// A RowMask captures a set of selected rows within a row range.
///
/// The range itself can be [`u64`], but the length of the range must fit into a [`usize`], this
/// allows us to use a `usize` filter mask within a much larger file.
#[derive(Debug, Clone)]
pub(crate) struct RowMask {
    row_offset: u64,
    mask: Mask,
}

impl Deref for RowMask {
    type Target = Mask;

    fn deref(&self) -> &Self::Target {
        &self.mask
    }
}

impl RowMask {
    pub fn new(row_offset: u64, mask: Mask) -> Self {
        Self { row_offset, mask }
    }

    /// The row range of the [`RowMask`].
    pub fn row_range(&self) -> Range<u64> {
        self.row_offset..self.row_offset + self.mask.len() as u64
    }

    /// The mask of the [`RowMask`].
    pub fn mask(&self) -> &Mask {
        &self.mask
    }

    // pub fn into_mask(self) -> Mask {
    //     self.mask
    // }

    pub fn intersect(&self, other: &Mask) -> Self {
        assert_eq!(
            self.mask.len(),
            other.len(),
            "Masks must have the same length"
        );
        let new_mask = self.mask.bitand(other);
        Self {
            row_offset: self.row_offset,
            mask: new_mask,
        }
    }
}
