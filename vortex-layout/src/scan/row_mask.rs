use std::ops::Range;

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
}
