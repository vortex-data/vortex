// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::dtype::FieldMask;
use vortex_error::VortexResult;
use vortex_layout::LayoutReader;
use vortex_layout::SplitPointIter;

use crate::IDEAL_SPLIT_SIZE;
use crate::selection::Selection;
use crate::split_by::SplitBy;

/// The maximum number of rows in a single range. This is somewhat arbitrarily chosen.
const MAX_RANGE_SIZE: u64 = IDEAL_SPLIT_SIZE / 25;

/// The minimum gap between ranges. This is somewhat arbitrarily chosen.
const MIN_GAP_BETWEEN_RANGES: u64 = IDEAL_SPLIT_SIZE / 2;

/// The way in which we compute splits for a file.
pub(super) enum Splits {
    /// Natural splits computed by the layout reader (e.g., computing splits across different-sized
    /// column chunks).
    Natural {
        split_by: SplitBy,
        field_mask: Vec<FieldMask>,
    },

    /// Exact split ranges.
    ///
    /// This is an optimization for when we know the exact rows we need to get from a file (which is
    /// common if we just want to select a few (sparse) indices).
    Ranges(Vec<Range<u64>>),
}

impl Splits {
    pub(super) fn iter(
        &self,
        layout_reader: &dyn LayoutReader,
        row_range: Range<u64>,
    ) -> VortexResult<Box<dyn Iterator<Item = Range<u64>> + Send>> {
        if row_range.is_empty() {
            return Ok(Box::new(std::iter::empty()));
        }

        match self {
            Splits::Natural {
                split_by,
                field_mask,
            } => {
                let points =
                    split_by.split_points(layout_reader, row_range.clone(), field_mask.clone())?;
                Ok(Box::new(SplitRangeIter::new(row_range.start, points)))
            }
            Splits::Ranges(ranges) => {
                Ok(Box::new(ranges.clone().into_iter().filter_map(
                    move |range| intersect_range(&range, &row_range),
                )))
            }
        }
    }
}

struct SplitRangeIter {
    start: u64,
    split_points: SplitPointIter,
}

impl SplitRangeIter {
    fn new(start: u64, split_points: SplitPointIter) -> Self {
        Self {
            start,
            split_points,
        }
    }
}

impl Iterator for SplitRangeIter {
    type Item = Range<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        let end = self.split_points.next()?;
        let range = self.start..end;
        self.start = end;
        Some(range)
    }
}

fn intersect_range(left: &Range<u64>, right: &Range<u64>) -> Option<Range<u64>> {
    let start = left.start.max(right.start);
    let end = left.end.min(right.end);
    (start < end).then_some(start..end)
}

/// Attempts to compute split ranges from the given selection.
pub(super) fn attempt_split_ranges(
    selection: &Selection,
    row_range: Option<&Range<u64>>,
) -> Option<Vec<Range<u64>>> {
    let Selection::IncludeByIndex(buffer) = selection else {
        return None;
    };

    // TODO(connor): We can be smarter here, as the row range is more restrictive than the
    // selection.
    if row_range.is_some() {
        return None;
    }

    let indices = buffer.as_slice();
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
