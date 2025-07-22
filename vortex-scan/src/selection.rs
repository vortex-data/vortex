// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use roaring::RoaringTreemap;
use std::cmp::min;
use std::ops::{Not, Range};
use std::sync::Arc;
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
    IncludeRoaring(Arc<RoaringTreemap>),
    /// A selection of rows to exclude using a [`roaring::RoaringTreemap`].
    ExcludeRoaring(Arc<RoaringTreemap>),
}

impl Selection {
    pub fn is_disjoint(&self, range: &Range<u64>) -> bool {
        match &self {
            Selection::All => false,
            Selection::IncludeByIndex(indices) => is_disjoint(indices, range),
            Selection::ExcludeByIndex(indices) => is_disjoint_with_exclusions(indices, range),
            Selection::IncludeRoaring(treemap) => non_empty_treemap_range(treemap, range),
            Selection::ExcludeRoaring(treemap) => all_empty_treemap_range(treemap, range),
        }
    }

    pub fn mask(&self, offset: u64, length: usize) -> Mask {
        let range = offset..offset + (length as u64);
        match self {
            Selection::All => Mask::new_true(length),
            Selection::IncludeByIndex(include) => {
                let mask = indices_range(&range, include)
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
                    .unwrap_or_else(|| Mask::new_false(length));

                mask
            }
            Selection::ExcludeByIndex(exclude) => {
                let mask = Selection::IncludeByIndex(exclude.clone())
                    .mask(offset, length)
                    .clone();
                mask.not()
            }
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
                let mask = Mask::from_indices(
                    length,
                    roaring
                        .iter()
                        .map(|idx| idx - range.start)
                        .map(|idx| {
                            usize::try_from(idx).vortex_expect("Index does not fit into a usize")
                        })
                        .collect(),
                );

                mask
            }
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
                let mask = Mask::from_excluded_indices(
                    length,
                    roaring.iter().map(|idx| idx - range.start).map(|idx| {
                        usize::try_from(idx).vortex_expect("Index does not fit into a usize")
                    }),
                );

                mask
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
        other.offset += range.start;
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

fn all_empty_treemap_range(treemap: &RoaringTreemap, range: &Range<u64>) -> bool {
    treemap_range(treemap, range).all(|(_, mut map)| map.next().is_none())
}

fn non_empty_treemap_range(treemap: &RoaringTreemap, range: &Range<u64>) -> bool {
    treemap_range(treemap, range).any(|(_, mut map)| map.next().is_some())
}

// #[cfg(test)]
// mod tests {
//     use itertools::Itertools;
//     use vortex_error::VortexResult;
//     use vortex_mask::Mask;
//
//     use super::*;
//
//     #[test]
//     fn test_contains_range() {
//         let mask = SlicedSelection::all(101).with_range(2..101);
//         assert!(mask.non_empty_range(0..101));
//         assert!(!mask.non_empty_range(101..103));
//         assert!(!mask.non_empty_range(0..2));
//     }
//
//     #[test]
//     fn test_range_masks() {
//         let mask = SlicedSelection::all(41).with_range(10..20);
//         assert_eq!(
//             mask.mask().collect::<VortexResult<Mask>>().unwrap(),
//             Mask::from_indices(41, (10..20).collect_vec())
//         );
//
//         let mask = SlicedSelection::all(10).with_range(0..9);
//         assert_eq!(
//             mask.mask().collect::<VortexResult<Mask>>().unwrap(),
//             Mask::from_indices(10, (0..9).collect_vec())
//         );
//
//         let mask = SlicedSelection::all(10).with_range(2..10);
//         assert_eq!(
//             mask.mask().collect::<VortexResult<Mask>>().unwrap(),
//             Mask::from_indices(10, (2..10).collect_vec())
//         );
//     }
//
//     #[test]
//     fn test_subset_basic() {
//         let mask = SlicedSelection::all(201).with_range(100..201);
//         let subset = mask.clone()._slice(10..51);
//         assert_eq!(
//             subset.mask().collect::<VortexResult<Mask>>().unwrap(),
//             Mask::AllFalse(41)
//         ); // 100 + 10
//         let subset = mask._slice(100..151);
//         assert_eq!(
//             subset.mask().collect::<VortexResult<Mask>>().unwrap(),
//             Mask::AllTrue(51)
//         ); // 100 + 10
//     }
//
//     #[test]
//     fn test_subset_chain() {
//         let mask = SlicedSelection::all(1001);
//         let subset1 = mask._slice(100..501); // Range becomes [100, 501)
//         let subset2 = subset1._slice(50..151); // Range becomes [150, 251)
//
//         assert_eq!(
//             subset2.mask().collect::<VortexResult<Mask>>().unwrap(),
//             Mask::AllTrue(101)
//         );
//     }
//
//     #[test]
//     fn test_range_and() {
//         let mask = SlicedSelection::all(30).with_range(5..15);
//
//         assert!(mask.non_empty_range(0..10));
//         assert!(mask.non_empty_range(10..20));
//         assert!(!mask.non_empty_range(20..30));
//
//         assert_eq!(
//             Mask::from_indices(10, (5..10).collect_vec()),
//             mask.clone()
//                 ._slice(0..10)
//                 .mask()
//                 .collect::<VortexResult<Mask>>()
//                 .unwrap(),
//         );
//
//         assert_eq!(
//             Mask::from_indices(10, (0..5).collect_vec()),
//             mask.clone()
//                 ._slice(10..20)
//                 .mask()
//                 .collect::<VortexResult<Mask>>()
//                 .unwrap(),
//         );
//
//         assert_eq!(
//             Mask::AllFalse(10),
//             mask._slice(20..30)
//                 .mask()
//                 .collect::<VortexResult<Mask>>()
//                 .unwrap(),
//         );
//     }
//
//     #[test]
//     fn test_subset_with_treemap() {
//         let mut treemap = RoaringTreemap::new();
//         // Add some values in the range
//         treemap.insert(105);
//         treemap.insert(120);
//         treemap.insert(180);
//
//         let mask = SlicedSelection::all(1001)
//             .with_range(0..180)
//             .with_treemap(Arc::new(treemap));
//
//         let subset = mask._slice(100..201);
//         let subset = subset._slice(10..21);
//
//         assert_eq!(
//             subset.mask().collect::<VortexResult<Mask>>().unwrap(),
//             Mask::from_indices(11, vec![10])
//         );
//     }
//
//     #[test]
//     fn test_subset_refines_range() {
//         let mut treemap = RoaringTreemap::new();
//         // Original range covers 0-1000
//         for i in (0..1000).step_by(10) {
//             treemap.insert(i);
//         }
//
//         let original_mask = SlicedSelection::all(2001)
//             .with_range(0..1001)
//             .with_treemap(Arc::new(treemap));
//
//         let subset_mask = original_mask._slice(0..1001);
//
//         // Create subset that covers positions 100-200 in the original range
//         let subset_mask = subset_mask._slice(100..201);
//
//         // Test that non_empty_range works correctly on the subset
//         assert!(subset_mask.non_empty_range(10..21)); // Should find value at 110
//         assert!(subset_mask.non_empty_range(50..61)); // Should find value at 150
//         assert!(!subset_mask.non_empty_range(150..161)); // Outside subset range
//     }
//
//     #[test]
//     fn test_subset_with_large_numbers() {
//         let mut treemap = RoaringTreemap::new();
//         // Use large numbers that span multiple partitions
//         let base = 0x123456780000000u64;
//         treemap.insert(base + 1000);
//         treemap.insert(base + 2000);
//         treemap.insert(base + 3000);
//
//         let mask = SlicedSelection::all(base + 5000).with_treemap(Arc::new(treemap));
//         let slice = mask._slice(base..base + 20001);
//         let slice = slice._slice(500..10501);
//         let slice = slice._slice(0..2001);
//
//         assert_eq!(
//             slice.mask().collect::<VortexResult<Mask>>().unwrap(),
//             Mask::from_indices(2001, vec![500, 1500])
//         );
//
//         // match: base + 500 + 1500..base + 500 + 1600 == base + 1900..base + 2100
//         assert!(slice.non_empty_range(1500..2001));
//         // Should not find values outside the subset range
//         assert!(!slice.non_empty_range(1900..2000));
//         assert!(!slice.non_empty_range(2001..2500));
//     }
//
//     #[test]
//     fn test_non_empty_range_edge_cases() {
//         let mut treemap = RoaringTreemap::new();
//         treemap.insert(50);
//         treemap.insert(150);
//
//         let mask = SlicedSelection::all(1001)
//             .with_range(0..1001)
//             .with_treemap(Arc::new(treemap));
//         let mask = mask._slice(0..201);
//
//         // Test exact boundaries
//         assert!(mask.non_empty_range(50..51)); // Exact match
//         assert!(mask.non_empty_range(49..52)); // Contains 50
//         assert!(!mask.non_empty_range(51..150)); // Gap between values
//
//         // Test range completely outside mask bounds
//         assert!(!mask.non_empty_range(300..401));
//         assert!(!mask.non_empty_range(0..1)); // Before mask start (but mask starts at 0)
//     }
//
//     #[test]
//     fn test_all_mask_behavior() {
//         let mask = SlicedSelection::all(501).with_range(100..501);
//
//         // All mask should always return true for non_empty_range
//         assert!(mask.non_empty_range(150..201));
//         assert!(mask.non_empty_range(100..501));
//         assert!(mask.non_empty_range(50..601)); // Even overlapping ranges
//
//         // But not for completely non-overlapping ranges
//         assert!(!mask.non_empty_range(501..503));
//         assert!(!mask.non_empty_range(600..701));
//     }
//
//     #[test]
//     fn test_non_empty_range_spans_partitions() {
//         let mut treemap = RoaringTreemap::new();
//
//         // Insert values in two different partitions
//         // Partition 0 (upper 32 bits = 0)
//         treemap.insert(0xFFFFFFFF); // Last value in partition 0
//
//         // Partition 1 (upper 32 bits = 1)
//         treemap.insert(0x100000000); // First value in partition 1
//         treemap.insert(0x100000001); // Second value in partition 1
//
//         // Partition 2 (upper 32 bits = 2)
//         treemap.insert(0x200000000); // First value in partition 2
//
//         let mask = SlicedSelection::all(0x400000001).with_treemap(Arc::new(treemap));
//         let mask = mask._slice(0..0x300000001);
//
//         // Test range that spans from partition 0 to partition 1
//         assert!(mask.non_empty_range(0xFFFFFFF0..0x100000011));
//
//         // Test range that spans from partition 1 to partition 2
//         assert!(mask.non_empty_range(0x100000000..0x200000001));
//
//         // Test range in middle of partition 1 with no values
//         assert!(!mask.non_empty_range(0x100000010..0x1FFFFFFFE));
//
//         // Test range that spans all three partitions
//         assert!(mask.non_empty_range(0..0x300000001));
//
//         // Test empty range between values in different partitions
//         assert!(!mask.non_empty_range(0x100000002..0x200000000));
//     }
//
//     #[test]
//     fn test_exclude_mask() {
//         let mut treemap = RoaringTreemap::new();
//         treemap.insert(100);
//         treemap.insert(500);
//
//         let mask = SlicedSelection::all(1001)
//             .with_range(0..1001)
//             .with_treemap(Arc::new(treemap))
//             .with_exclude();
//
//         assert!(mask.non_empty_range(0..100));
//         assert!(mask.non_empty_range(101..201));
//         assert!(mask.non_empty_range(101..499));
//         assert!(mask.non_empty_range(501..(u32::MAX as u64)));
//         assert!(!mask.non_empty_range(100..101));
//         assert!(!mask.non_empty_range(500..501));
//         assert!(!mask.non_empty_range(0..10000));
//     }
//
//     use vortex_buffer::Buffer;
//
//     use super::Selection;
//
//     #[test]
//     fn test_row_mask_all() {
//         let selection = Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 5, 7]));
//         let range = 1..8;
//         let mask = selection
//             .tree_row_mask(7, Some(range.clone()))
//             ._slice(range);
//
//         println!(
//             "{:?}",
//             mask.mask()
//                 .collect::<VortexResult<Mask>>()
//                 .unwrap()
//                 .to_vec()
//         );
//
//         assert_eq!(
//             mask.mask().collect::<VortexResult<Mask>>().unwrap(),
//             Mask::from_indices(7, vec![0, 2, 4, 6])
//         );
//     }
//
//     #[test]
//     fn test_row_mask_slice() {
//         let selection = Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 5, 7]));
//         let range = 3..6;
//         let mask = selection
//             .tree_row_mask(3, Some(range.clone()))
//             ._slice(range);
//
//         assert_eq!(
//             mask.mask().collect::<VortexResult<Mask>>().unwrap(),
//             Mask::from_indices(3, vec![0, 2])
//         );
//     }
//
//     #[test]
//     fn test_row_mask_exclusive() {
//         let selection = Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 5, 7]));
//         let range = 3..5;
//         let mask = selection
//             .tree_row_mask(2, Some(range.clone()))
//             ._slice(range);
//
//         assert_eq!(
//             mask.mask().collect::<VortexResult<Mask>>().unwrap(),
//             Mask::from_indices(2, vec![0])
//         );
//     }
//
//     #[test]
//     fn test_row_mask_all_false() {
//         let selection = Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 5, 7]));
//         let range = 8..10;
//         let mask = selection
//             .tree_row_mask(2, Some(range.clone()))
//             ._slice(range);
//
//         assert_eq!(
//             mask.mask().collect::<VortexResult<Mask>>().unwrap(),
//             Mask::AllFalse(2)
//         );
//     }
//
//     #[test]
//     fn test_row_mask_all_true() {
//         let selection = Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 4, 5, 6]));
//         let range = 3..7;
//         let mask = selection
//             .tree_row_mask(4, Some(range.clone()))
//             ._slice(range);
//
//         assert_eq!(
//             mask.mask().collect::<VortexResult<Mask>>().unwrap(),
//             Mask::AllTrue(4)
//         );
//     }
//
//     #[test]
//     fn test_row_mask_zero() {
//         let selection = Selection::IncludeByIndex(Buffer::from_iter(vec![0]));
//         let range = 0..5;
//         let mask = selection
//             .tree_row_mask(5, Some(range.clone()))
//             ._slice(range);
//
//         assert_eq!(
//             mask.mask().collect::<VortexResult<Mask>>().unwrap(),
//             Mask::from_indices(5, vec![0])
//         );
//     }
// }
