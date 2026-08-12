// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_scan::selection::Selection;

use crate::scan::IDEAL_SPLIT_SIZE;

/// The maximum number of rows in a single range. This is somewhat arbitrarily chosen.
const MAX_RANGE_SIZE: u64 = IDEAL_SPLIT_SIZE / 25;

/// The minimum gap between ranges. This is somewhat arbitrarily chosen.
const MIN_GAP_BETWEEN_RANGES: u64 = IDEAL_SPLIT_SIZE / 2;

/// The way in which we compute splits for a file.
pub enum Splits {
    /// Natural splits computed by the layout reader (e.g., computing splits across different-sized
    /// column chunks).
    ///
    /// The boundaries are sorted in ascending order and deduplicated.
    Natural(Arc<[u64]>),

    /// Exact split ranges.
    ///
    /// This is an optimization for when we know the exact rows we need to get from a file (which is
    /// common if we just want to select a few (sparse) indices).
    Ranges(Vec<Range<u64>>),
}

/// Attempts to compute split ranges from the given selection.
pub fn attempt_split_ranges(
    selection: &Selection,
    row_range: Option<&Range<u64>>,
) -> Option<Vec<Range<u64>>> {
    let Selection::IncludeByIndex(buffer) = selection else {
        return None;
    };

    let indices = buffer.as_slice();
    let indices = if let Some(row_range) = row_range {
        if row_range.is_empty() {
            return Some(Vec::new());
        }

        let start = indices.partition_point(|&index| index < row_range.start);
        let end = indices.partition_point(|&index| index < row_range.end);
        &indices[start..end]
    } else {
        indices
    };

    if indices.is_empty() {
        return Some(Vec::new());
    }

    debug_assert!(indices.is_sorted());

    // We need to create ranges that will represent splits that cover our indices.
    // We want to make sure that we do not create too many splits. We also want to make sure our
    // splits do not cover too much as they would overlap column chunk boundaries.

    let mut ranges = Vec::with_capacity((indices.len() as u64 / MAX_RANGE_SIZE) as usize);
    let mut curr_start = indices[0];
    let mut curr_end = indices[0] + 1; // Ranges are exclusive at the end.

    // Build the ranges by iterating over the indices and attempting to extend the current range.
    for &idx in &indices[1..] {
        // Check what the new range size would be if we extend the current range.
        let new_range_size = (idx + 1) - curr_start;
        let gap = (idx + 1) - curr_end;

        if new_range_size >= MAX_RANGE_SIZE {
            // If we need to start a new range, check that it is far enough away.
            if gap >= MIN_GAP_BETWEEN_RANGES {
                // Finalize the current range and start a new one.
                ranges.push(curr_start..curr_end);
                curr_start = idx;
                curr_end = idx + 1;
            } else {
                return None;
            }
        } else {
            // Extend the current range to include this index.
            curr_end = idx + 1;
        }
    }

    // Add the last range.
    ranges.push(curr_start..curr_end);

    Some(ranges)
}

#[cfg(test)]
mod tests {
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;

    use super::*;

    fn include(indices: impl IntoIterator<Item = u64>) -> Selection {
        Selection::IncludeByIndex(
            StrictSortedBuffer::try_new(Buffer::from_iter(indices))
                .vortex_expect("test indices must be strictly sorted"),
        )
    }

    #[test]
    fn split_ranges_intersect_row_range() {
        let ranges = attempt_split_ranges(&include([1, 3, 5, 7, 9]), Some(&(3..9)))
            .vortex_expect("sparse split planning should apply");

        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 3..8);
    }

    #[test]
    fn split_ranges_empty_intersection() {
        assert_eq!(
            attempt_split_ranges(&include([1, 3, 5]), Some(&(6..9))),
            Some(Vec::new())
        );
    }

    #[test]
    fn restrictive_row_range_enables_sparse_splits() {
        let indices = (0..MAX_RANGE_SIZE).chain([MAX_RANGE_SIZE + MIN_GAP_BETWEEN_RANGES]);
        let ranges = attempt_split_ranges(&include(indices), Some(&(MAX_RANGE_SIZE..u64::MAX)))
            .vortex_expect("the row range should make sparse split planning applicable");

        assert_eq!(ranges.len(), 1);
        assert_eq!(
            ranges[0],
            MAX_RANGE_SIZE + MIN_GAP_BETWEEN_RANGES..MAX_RANGE_SIZE + MIN_GAP_BETWEEN_RANGES + 1
        );
    }
}
