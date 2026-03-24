// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;

use vortex_mask::Mask;

use crate::v2::selection::Selection;

/// Identifies a split within a scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SplitId(u32);

impl From<u32> for SplitId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

/// A split with its associated row range.
#[derive(Debug, Clone)]
pub struct SplitRange {
    pub id: SplitId,
    pub row_range: Range<u64>,
    pub mask: Mask,
}

/// Forms splits from boundary points, coalescing small adjacent intervals and subdividing
/// large ones.
///
/// 1. Converts consecutive boundary pairs into intervals.
/// 2. Greedily coalesces adjacent small intervals up to `min_split_rows`.
/// 3. Subdivides intervals exceeding `max_split_rows`.
/// 4. Assigns monotonic [`SplitId`]s.
///
// FIXME(ngates): add Selection to slice out the empty ends of splits and skip large empty
//  sections.
pub fn form_splits(
    boundaries: &BTreeSet<u64>,
    selection: &Selection,
    total_row_count: u64,
    min_split_rows: u64,
    max_split_rows: u64,
) -> Vec<SplitRange> {
    if boundaries.is_empty() || total_row_count == 0 {
        return Vec::new();
    }

    // Convert boundary points to intervals.
    let points: Vec<u64> = boundaries.iter().copied().collect();
    let mut intervals: Vec<Range<u64>> = Vec::new();
    for window in points.windows(2) {
        let start = window[0];
        let end = window[1];
        if start < end && start < total_row_count {
            intervals.push(start..end.min(total_row_count));
        }
    }

    if intervals.is_empty() {
        return Vec::new();
    }

    // Greedily coalesce adjacent small intervals.
    let mut coalesced: Vec<Range<u64>> = Vec::new();
    let mut current = intervals[0].clone();
    for interval in &intervals[1..] {
        let current_len = current.end - current.start;
        let merged_len = interval.end - current.start;
        if current_len < min_split_rows
            && current.end == interval.start
            && merged_len <= max_split_rows
        {
            current.end = interval.end;
        } else {
            coalesced.push(current);
            current = interval.clone();
        }
    }
    coalesced.push(current);

    // Subdivide large intervals and assign split IDs.
    let mut splits = Vec::new();
    let mut split_id = 0u32;
    for interval in coalesced {
        let len = interval.end - interval.start;
        if len <= max_split_rows {
            splits.push(SplitRange {
                id: SplitId(split_id),
                row_range: interval.clone(),
                mask: selection.row_mask(&interval),
            });
            split_id += 1;
        } else {
            let mut start = interval.start;
            while start < interval.end {
                let end = (start + max_split_rows).min(interval.end);
                splits.push(SplitRange {
                    id: SplitId(split_id),
                    row_range: start..end,
                    mask: selection.row_mask(&*start..end),
                });
                split_id += 1;
                start = end;
            }
        }
    }

    splits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_form_splits_coalescing() {
        let mut boundaries = BTreeSet::new();
        for b in [0, 100, 200, 300, 1000] {
            boundaries.insert(b);
        }

        let splits = form_splits(&boundaries, &Selection::All, 1000, 250, 1000);

        // [0..100, 100..200, 200..300] coalesce to [0..300], then [300..1000] stays.
        assert_eq!(splits.len(), 2);
        assert_eq!(splits[0].row_range, 0..300);
        assert_eq!(splits[1].row_range, 300..1000);
        assert_eq!(splits[0].id, SplitId(0));
        assert_eq!(splits[1].id, SplitId(1));
    }

    #[test]
    fn test_form_splits_subdivision() {
        let mut boundaries = BTreeSet::new();
        boundaries.insert(0);
        boundaries.insert(2000);

        let splits = form_splits(&boundaries, &Selection::All, 2000, 100, 500);

        // [0..2000] subdivides into [0..500, 500..1000, 1000..1500, 1500..2000].
        assert_eq!(splits.len(), 4);
        assert_eq!(splits[0].row_range, 0..500);
        assert_eq!(splits[1].row_range, 500..1000);
        assert_eq!(splits[2].row_range, 1000..1500);
        assert_eq!(splits[3].row_range, 1500..2000);
    }

    #[test]
    fn test_form_splits_empty() {
        let boundaries = BTreeSet::new();
        assert!(form_splits(&boundaries, &Selection::All, 1000, 100, 500).is_empty());
        assert!(form_splits(&BTreeSet::from([0, 100]), &Selection::All, 0, 100, 500).is_empty());
    }

    #[test]
    fn test_form_splits_single_interval() {
        let boundaries = BTreeSet::from([0, 500]);
        let splits = form_splits(&boundaries, &Selection::All, 500, 100, 1000);
        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0].row_range, 0..500);
    }
}
