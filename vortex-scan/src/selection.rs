// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;
use std::ops::{Not, Range};
use std::sync::Arc;

#[cfg(feature = "roaring")]
use roaring::RoaringTreemap;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_layout::RowSelection;
use vortex_mask::Mask;

/// A selection identifies a set of rows to include in the scan (in addition to applying any
/// filter predicates).
#[derive(Default, Clone)]
pub enum Selection {
    /// No selection, all rows are included.
    #[default]
    All,
    // TODO(joe): replace this with IncludeRoaring
    /// A selection of rows to include by index.
    IncludeByIndex(Buffer<u64>),
    /// A selection of rows to exclude by index.
    ExcludeByIndex(Buffer<u64>),
    /// A selection of rows to include using a [`roaring::RoaringTreemap`].
    #[cfg(feature = "roaring")]
    IncludeRoaring(Arc<RoaringTreemap>),
    /// A selection of rows to exclude using a [`roaring::RoaringTreemap`].
    #[cfg(feature = "roaring")]
    ExcludeRoaring(Arc<RoaringTreemap>),
}

impl Selection {
    pub fn is_disjoint(&self, range: &Range<u64>) -> bool {
        match &self {
            Selection::All => false,
            Selection::IncludeByIndex(indices) => is_disjoint(indices, range),
            Selection::ExcludeByIndex(indices) => is_disjoint_with_exclusions(indices, range),
            #[cfg(feature = "roaring")]
            Selection::IncludeRoaring(treemap) => non_empty_treemap_range(treemap, range),
            #[cfg(feature = "roaring")]
            Selection::ExcludeRoaring(treemap) => all_empty_treemap_range(treemap, range),
        }
    }

    pub fn mask(&self, offset: u64, length: usize) -> Mask {
        let range = offset..offset + (length as u64);
        match self {
            Selection::All => Mask::new_true(length),
            Selection::IncludeByIndex(include) => indices_range(&range, include)
                .map(|idx_range| {
                    Mask::from_indices(
                        length,
                        include
                            .slice(idx_range)
                            .iter()
                            .map(|idx| *idx - range.start)
                            .map(|idx| {
                                usize::try_from(idx)
                                    .vortex_expect("Index does not fit into a usize")
                            })
                            .collect(),
                    )
                })
                .unwrap_or_else(|| Mask::new_false(length)),
            Selection::ExcludeByIndex(exclude) => {
                let mask = Selection::IncludeByIndex(exclude.clone()).mask(offset, length);
                mask.not()
            }
            #[cfg(feature = "roaring")]
            Selection::IncludeRoaring(roaring) => {
                use std::ops::BitAnd;

                // First we perform a cheap is_disjoint check
                let mut range_treemap = RoaringTreemap::new();
                range_treemap.insert_range(range.clone());

                if roaring.is_disjoint(&range_treemap) {
                    return Mask::new_false(length);
                }

                // Otherwise, intersect with the selected range and shift to relativize.
                let roaring = roaring.as_ref().bitand(range_treemap);

                Mask::from_indices(
                    length,
                    roaring
                        .iter()
                        .map(|idx| idx - range.start)
                        .map(|idx| {
                            usize::try_from(idx).vortex_expect("Index does not fit into a usize")
                        })
                        .collect(),
                )
            }
            #[cfg(feature = "roaring")]
            Selection::ExcludeRoaring(roaring) => {
                use std::ops::BitAnd;

                let mut range_treemap = RoaringTreemap::new();
                range_treemap.insert_range(range.clone());

                // If there are no deletions in the intersection, then we have an all true mask.
                if roaring.intersection_len(&range_treemap) == length as u64 {
                    return Mask::new_true(length);
                }

                // Otherwise, intersect with the selected range and shift to relativize.
                let roaring = roaring.as_ref().bitand(range_treemap);

                Mask::from_excluded_indices(
                    length,
                    roaring.iter().map(|idx| idx - range.start).map(|idx| {
                        usize::try_from(idx).vortex_expect("Index does not fit into a usize")
                    }),
                )
            }
        }
    }
}

#[derive(Clone)]
pub struct RangeSelection {
    range: Option<Range<u64>>,
    selection: Selection,
}

impl RangeSelection {
    pub fn new(range: Option<Range<u64>>, selection: Selection) -> Self {
        Self { range, selection }
    }

    pub fn is_disjoint(&self, range: &Range<u64>) -> bool {
        if let Some(sel_range) = &self.range {
            if range.start >= sel_range.end || range.end <= sel_range.start {
                return true;
            };
        };

        self.selection.is_disjoint(range)
    }
}

#[derive(Clone)]
pub struct SlicedSelection {
    offset: u64,
    length: u64,
    inner: RangeSelection,
}

impl RowSelection for SlicedSelection {
    fn is_disjoint(&self, range: &Range<u64>) -> bool {
        self.inner.is_disjoint(
            &(range.start + self.offset..min(self.offset + self.length, range.end + self.offset)),
        )
    }

    fn slice(&self, range: &Range<u64>) -> Arc<dyn RowSelection + 'static> {
        let mut other = SlicedSelection::new(self.inner.clone());
        other.offset += self.offset + range.start;
        other.length = min(range.end - range.start, self.length);
        Arc::new(other) as Arc<_>
    }
}

impl SlicedSelection {
    pub fn new(range_selection: RangeSelection) -> Self {
        Self {
            offset: 0,
            length: u64::MAX,
            inner: range_selection,
        }
    }
}

fn is_disjoint(indices: &[u64], range: &Range<u64>) -> bool {
    if indices.is_empty() || range.is_empty() {
        return true;
    }

    // Find the first index that could potentially be in range
    match indices.binary_search(&range.start) {
        Ok(_) => false, // Found exact match, not disjoint
        Err(pos) => {
            // Check if the index at pos is within range
            pos >= indices.len() || indices[pos] >= range.end
        }
    }
}

fn is_disjoint_with_exclusions(excluded_indices: &[u64], range: &Range<u64>) -> bool {
    if range.is_empty() {
        return true;
    }

    // Find all indices in the range [offset, offset + length)
    let start_pos = match excluded_indices.binary_search(&range.start) {
        Ok(pos) => pos,
        Err(pos) => pos,
    };

    // Find the first index >= range_end
    let end_pos = match excluded_indices.binary_search(&range.end) {
        Ok(pos) => pos,
        Err(pos) => pos,
    };

    // Count excluded indices within the range
    let excluded_count = end_pos - start_pos;

    // If all indices in the range are excluded, it's disjoint
    excluded_count as u64 == range.end - range.start
}

/// Find the positional range within row_indices that covers all rows in the given range.
fn indices_range(range: &Range<u64>, row_indices: &[u64]) -> Option<Range<usize>> {
    if row_indices.first().is_some_and(|&first| first >= range.end)
        || row_indices.last().is_some_and(|&last| range.start > last)
    {
        return None;
    }

    // For the given row range, find the indices that are within the row_indices.
    let start_idx = row_indices
        .binary_search(&range.start)
        .unwrap_or_else(|x| x);
    let end_idx = row_indices.binary_search(&range.end).unwrap_or_else(|x| x);

    (start_idx != end_idx).then_some(start_idx..end_idx)
}

#[cfg(feature = "roaring")]
fn clamp_range_to_partition(partition_key: u32, range: &Range<u64>) -> Range<u32> {
    let start_partition = (range.start >> 32) as u32;
    let end_partition = (range.end >> 32) as u32;

    if partition_key == start_partition && partition_key == end_partition {
        // Same partition - use exact range
        (range.start & 0xFFFFFFFF) as u32..(range.end & 0xFFFFFFFF) as u32
    } else if partition_key == start_partition {
        // First partition - from start to end of partition
        (range.start & 0xFFFFFFFF) as u32..u32::MAX
    } else if partition_key == end_partition {
        // Last partition - from start of partition to end
        0..(range.end & 0xFFFFFFFF) as u32
    } else {
        // Middle partition - entire range
        0..u32::MAX
    }
}

#[cfg(feature = "roaring")]
fn treemap_range(
    treemap: &RoaringTreemap,
    range: &Range<u64>,
) -> impl Iterator<Item = (u32, impl Iterator<Item = u32>)> {
    let start_partition = (range.start >> 32) as u32;
    let end_partition = (range.end >> 32) as u32;

    treemap
        .bitmaps()
        .skip_while(move |(key, _)| *key < start_partition)
        .take_while(move |(key, _)| *key <= end_partition)
        .map(move |(partition_key, bitmap)| {
            let lower_range = clamp_range_to_partition(partition_key, range);

            // Check all not values match the range
            (partition_key, bitmap.range(lower_range))
        })
}

#[cfg(feature = "roaring")]
fn all_empty_treemap_range(treemap: &RoaringTreemap, range: &Range<u64>) -> bool {
    treemap_range(treemap, range).all(|(_, mut map)| map.next().is_none())
}
#[cfg(feature = "roaring")]
fn non_empty_treemap_range(treemap: &RoaringTreemap, range: &Range<u64>) -> bool {
    treemap_range(treemap, range).any(|(_, mut map)| map.next().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::row_mask::RowMask;
    use crate::selection_intersection::SelectionIntersectionMaskIterator;
    use std::iter::once;
    use vortex_mask::Mask::{AllFalse, AllTrue};

    #[test]
    fn test_contains_range() {
        let sel = SlicedSelection::new(RangeSelection::new(Some(2..101), Selection::All));
        assert!(!sel.is_disjoint(&(0..101)));
        assert!(sel.is_disjoint(&(101..103)));
        assert!(sel.is_disjoint(&(0..2)));
    }

    #[test]
    fn test_subset_basic() {
        let sel = SlicedSelection::new(RangeSelection::new(Some(100..201), Selection::All));

        let subset = sel.slice(&(50..250));
        assert!(subset.is_disjoint(&(0..50)));
        assert!(!subset.is_disjoint(&(50..51)));
        assert!(!subset.is_disjoint(&(49..150)));
        assert!(!subset.is_disjoint(&(150..151)));
        assert!(subset.is_disjoint(&(151..152)));
    }

    #[test]
    fn test_subset_chain() {
        let sel = SlicedSelection::new(RangeSelection::new(Some(100..201), Selection::All));

        let subset = sel.slice(&(50..250));
        let subset = subset.slice(&(10..200));
        assert!(subset.is_disjoint(&(0..40)));
        assert!(!subset.is_disjoint(&(40..41)));
        assert!(!subset.is_disjoint(&(49..150)));
        assert!(subset.is_disjoint(&(142..151)));
    }

    #[cfg(feature = "roaring")]
    #[test]
    fn test_non_empty_range_spans_partitions() {
        let mut treemap = RoaringTreemap::new();

        // Insert values in two different partitions
        // Partition 0 (upper 32 bits = 0)
        treemap.insert(0xFFFFFFFF); // Last value in partition 0

        // Partition 1 (upper 32 bits = 1)
        treemap.insert(0x100000000); // First value in partition 1
        treemap.insert(0x100000001); // Second value in partition 1

        // Partition 2 (upper 32 bits = 2)
        treemap.insert(0x200000000); // First value in partition 2

        let mask = SlicedSelection::new(RangeSelection::new(
            Some(0..0x400000001),
            Selection::IncludeRoaring(Arc::new(treemap)),
        ));
        let mask = mask.slice(&(0..0x300000001));

        // Test range that spans from partition 0 to partition 1
        assert!(!mask.is_disjoint(&(0xFFFFFFF0..0x100000011)));

        // Test range that spans from partition 1 to partition 2
        assert!(!mask.is_disjoint(&(0x100000000..0x200000001)));

        // Test range in middle of partition 1 with no values
        assert!(mask.is_disjoint(&(0x100000010..0x1FFFFFFFE)));

        // Test range that spans all three partitions
        assert!(!mask.is_disjoint(&(0..0x300000001)));

        // Test empty range between values in different partitions
        assert!(mask.is_disjoint(&(0x100000002..0x200000000)));
    }

    fn first_row_mask(range: Range<u64>, selection: Selection) -> RowMask {
        let mut iter = SelectionIntersectionMaskIterator::new(
            Box::new(once(Ok(AllTrue(
                usize::try_from(range.end).vortex_expect("use a smaller range"),
            )))),
            selection,
        );
        iter.with_range(range);
        let result = iter.next();
        assert!(result.is_some());
        assert!(iter.next().is_none());
        result.unwrap().unwrap()
    }

    #[test]
    fn test_row_mask_all() {
        let selection = Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 5, 7]));
        let range = 1..8;
        let result = first_row_mask(range, selection);

        assert_eq!(
            result,
            RowMask::new(1, Mask::from_indices(7, vec![0, 2, 4, 6]))
        );
    }

    #[test]
    fn test_row_mask_slice() {
        let selection = Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 5, 7]));
        let range = 3..6;
        let result = first_row_mask(range, selection);

        assert_eq!(result, RowMask::new(3, Mask::from_indices(3, vec![0, 2])));
    }

    #[test]
    fn test_row_mask_exclusive() {
        let selection = Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 5, 7]));
        let range = 3..5;
        let result = first_row_mask(range, selection);

        assert_eq!(result, RowMask::new(3, Mask::from_indices(2, vec![0])));
    }

    #[test]
    fn test_row_mask_all_false() {
        let selection = Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 5, 7]));
        let range = 8..10;
        let result = first_row_mask(range, selection);

        assert_eq!(result, RowMask::new(8, AllFalse(2)));
    }

    #[test]
    fn test_row_mask_all_true() {
        let selection = Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 4, 5, 6]));
        let range = 3..7;
        let result = first_row_mask(range, selection);
        assert_eq!(result, RowMask::new(3, AllTrue(4)));
    }

    #[test]
    fn test_row_mask_zero() {
        let selection = Selection::IncludeByIndex(Buffer::from_iter(vec![0]));
        let range = 0..5;
        let result = first_row_mask(range, selection);
        assert_eq!(result, RowMask::new(0, Mask::from_indices(5, vec![0])));
    }
}
