// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Selection;
use crate::row_mask::RowMask;
use std::ops::{BitAnd, Range};
use vortex_error::{VortexExpect, VortexResult};
use vortex_layout::masks::BoxMaskIterator;

/// Given an iterator of masks a range and a selection, returns row masks that intersect the range
/// and satisfy the selection
pub struct SelectionIntersectionMaskIterator {
    iterators: BoxMaskIterator,
    offset: u64,
    selection: Selection,
    range: Range<u64>,
}

impl SelectionIntersectionMaskIterator {
    pub fn new(iterators: BoxMaskIterator, selection: Selection) -> Self {
        Self {
            iterators,
            selection,
            offset: 0,
            range: 0..u64::MAX,
        }
    }

    pub fn with_range(&mut self, range: Range<u64>) {
        self.range = range;
    }
}

impl Iterator for SelectionIntersectionMaskIterator {
    type Item = VortexResult<RowMask>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mask = match self.iterators.next() {
                None => return None,
                Some(Err(e)) => return Some(Err(e)),
                Some(Ok(mask)) => mask,
            };

            let mask_start = self.offset;
            let mask_end = self.offset + mask.len() as u64;
            self.offset = mask_end;

            // Check if mask is completely outside range
            if mask_end <= self.range.start || mask_start >= self.range.end {
                continue;
            }

            let overlap_start = mask_start.max(self.range.start);
            let overlap_end = mask_end.min(self.range.end);

            // If mask is partially outside range, slice it
            let (sliced_mask, slice_offset) =
                if overlap_start > mask_start || overlap_end < mask_end {
                    // Both local_start and length will fit into usize, since they are smaller than
                    // a `mask.len()`
                    let local_start = usize::try_from(overlap_start - mask_start)
                        .vortex_expect("must will fit into usize");
                    let length = usize::try_from(overlap_end - overlap_start)
                        .vortex_expect("must will fit into usize");
                    (mask.slice(local_start, length), overlap_start)
                } else {
                    (mask, mask_start)
                };

            // Fast path for all-false masks
            if sliced_mask.all_false() {
                return Some(Ok(RowMask::new(slice_offset, sliced_mask)));
            }

            // Apply selection mask to the effective range
            let selection_mask = self.selection.mask(slice_offset, sliced_mask.len());
            let final_mask = selection_mask.bitand(&sliced_mask);

            return Some(Ok(RowMask::new(slice_offset, final_mask)));
        }
    }
}
