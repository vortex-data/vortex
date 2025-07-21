// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::{max, min};
use std::iter;
use std::ops::Range;
use std::sync::Arc;

use itertools::Itertools;
use roaring::RoaringTreemap;
use vortex_mask::Mask;

use crate::masks::BoxMaskIterator;

#[allow(dead_code)]
#[derive(Clone)]
pub struct SlicedTreemap {
    pub treemap: Arc<RoaringTreemap>,
    pub offset: u64,
    pub length: u64,
}

#[allow(dead_code)]
#[derive(Clone)]
enum ValueConstraint {
    TreeMap(SlicedTreemap),
    Range(Range<u64>),
}

#[derive(Clone)]
pub struct TreeRowMask {
    offset: u64,
    length: u64,
    value_constraints: ValueConstraint,
    exclude: bool,
}

impl TreeRowMask {
    pub fn all(length: u64) -> Self {
        Self {
            offset: 0,
            length,
            value_constraints: ValueConstraint::Range(u64::MIN..u64::MAX),
            exclude: false,
        }
    }

    pub fn with_range(mut self, range: Range<u64>) -> Self {
        self.value_constraints = ValueConstraint::Range(range);
        self
    }

    pub fn with_treemap(mut self, treemap: SlicedTreemap) -> Self {
        self.value_constraints = ValueConstraint::TreeMap(treemap);
        self
    }

    pub fn with_exclude(mut self) -> Self {
        self.exclude = true;
        self
    }

    pub fn slice(mut self, range: Range<u64>) -> Self {
        self.offset += range.start;
        self.length = min(range.end - range.start, self.length);
        self
    }

    pub fn mask(&self) -> BoxMaskIterator {
        match &self.value_constraints {
            ValueConstraint::Range(range) => {
                if self.exclude {
                    let start = range.start;
                    let end = range.end;
                    let idx = (0..self.length)
                        .filter(|&i| i + self.offset >= start && i + self.offset < end)
                        .map(|i| i as usize)
                        .collect_vec();
                    Box::new(iter::once(Ok(Mask::from_excluded_indices(
                        self.length as usize,
                        idx,
                    ))))
                } else {
                    let slice_end = self.offset + self.length;
                    let r1 = self.offset..min(range.start, slice_end);
                    let r2 = max(range.start, self.offset)..min(range.end, slice_end);
                    let r3 = range.end..max(range.end, slice_end);
                    let mut masks = Vec::with_capacity(3);
                    if r1.start < r1.end {
                        masks.push(Ok(Mask::AllFalse((r1.end - r1.start) as usize)))
                    };
                    if r2.start < r2.end {
                        masks.push(Ok(Mask::AllTrue((r2.end - r2.start) as usize)))
                    }
                    if r3.start < r3.end {
                        masks.push(Ok(Mask::AllFalse((r3.end - r3.start) as usize)))
                    }

                    Box::new(masks.into_iter())
                }
            }
            ValueConstraint::TreeMap(SlicedTreemap {
                treemap,
                offset: tree_offset,
                length: tree_length,
            }) => {
                let range = self.offset..self.offset + self.length;

                let r1 = range.start..min(*tree_offset, range.end);
                let r2 =
                    max(*tree_offset, range.start)..min(*tree_offset + *tree_length, range.end);
                let r3 = *tree_offset + *tree_length..max(*tree_offset + *tree_length, range.end);

                let mut masks = vec![];
                if r1.start < r1.end {
                    masks.push(Ok(Mask::AllFalse((r1.end - r1.start) as usize)));
                }

                if r2.start < r2.end {
                    let start = r2.start;
                    let end = r2.end;

                    let start_partition = (start >> 32) as u32;
                    let end_partition = (end >> 32) as u32;
                    let iter = treemap
                        .bitmaps()
                        .skip_while(|(key, _)| *key < start_partition)
                        .take_while(|(key, _)| *key <= end_partition)
                        .flat_map(|(partition_key, bitmap)| {
                            let (low_start, low_end) =
                                clamp_range_to_partition(partition_key, start, end);

                            // Check all not values match the range
                            bitmap.range(low_start..low_end).map(move |i| {
                                ((((partition_key as u64) << 32) + (i as u64)) - self.offset)
                                    as usize
                            })
                        });

                    if self.exclude {
                        masks.push(Ok(Mask::from_excluded_indices(
                            (r2.end - r2.start) as usize,
                            iter,
                        )))
                    } else {
                        let iter = iter.fold(Vec::new(), |mut groups: Vec<Vec<_>>, value| {
                            match groups.last_mut() {
                                Some(last_group)
                                    if last_group.iter().all(|&x| (value - x) < 2048) =>
                                {
                                    last_group.push(value);
                                }
                                _ => {
                                    groups.push(vec![value]);
                                }
                            }
                            groups
                        });

                        let mut last_end = 0;

                        for group in iter {
                            let group_start = *group.first().unwrap();
                            let group_end = *group.last().unwrap();

                            // Add gap before this group if there is one
                            if last_end < group_start {
                                masks.push(Ok(Mask::AllFalse(group_start - last_end)));
                            }

                            // Add the group itself
                            let len = group_end - group_start + 1;
                            masks.push(Ok(Mask::from_indices(
                                len,
                                group.iter().map(|i| *i - group_start).collect_vec(),
                            )));

                            last_end = group_end + 1;
                        }

                        if last_end < r2.end as usize {
                            masks.push(Ok(Mask::AllFalse((end - start) as usize - last_end)));
                        }
                    }
                }

                if r3.start < r3.end {
                    masks.push(Ok(Mask::AllFalse((r3.end - r3.start) as usize)));
                }
                Box::new(masks.into_iter())
            }
        }
    }

    pub fn non_empty_range(&self, range: Range<u64>) -> bool {
        let abs_start = range.start + self.offset;
        let abs_end = min(self.offset + self.length, range.end + self.offset);
        if abs_start >= abs_end {
            return false;
        }
        match &self.value_constraints {
            ValueConstraint::Range(c_range) => {
                max(abs_start, c_range.start) < min(abs_end, c_range.end)
            }
            ValueConstraint::TreeMap(SlicedTreemap {
                treemap,
                offset: tree_offset,
                length: tree_length,
            }) => {
                let start = max(abs_start, *tree_offset);
                let end = min(abs_end, tree_offset + tree_length);
                non_empty_treemap_range(treemap, start, end)
            }
        }
    }
}

/// Uses a roaring treemap to store a set of row indices.
/// This can be refined and a row range query can be performed.
// #[derive(Debug, Clone)]
// pub struct TreeRowMask {
//     /// The covering row range, this must be checked along with the treemap
//     range: Range<u64>,
//     /// The offset into the range
//     offset: u64,
//     /// The length of the value offseted range
//     length: u64,
//     /// Empty if the treemap is all full
//     treemap: Option<Arc<RoaringTreemap>>,
//     /// True if the treemap is used for exclude, not include
//     treemap_exclude: bool,
// }
//
// impl TreeRowMask {
//     /// Create a full mask for all rows in the range
//     pub fn all(length: u64) -> Self {
//         Self {
//             range: u64::MIN..u64::MAX,
//             offset: 0,
//             length,
//             treemap: None,
//             treemap_exclude: false,
//         }
//     }
//
//     pub fn with_treemap(mut self, treemap: RoaringTreemap) -> Self {
//         self.treemap = Some(Arc::new(treemap));
//         self
//     }
//
//     pub fn with_exclude(mut self) -> Self {
//         self.treemap_exclude = true;
//         self
//     }
//
//     pub fn with_range(mut self, range: Range<u64>) -> Self {
//         self.range = range;
//         self
//     }
//
//     pub fn new(range: Range<u64>, treemap: RoaringTreemap) -> Self {
//         let length = range.end - range.start;
//         Self {
//             range,
//             offset: 0,
//             length,
//             treemap: Some(Arc::new(treemap)),
//             treemap_exclude: false,
//         }
//     }
//
//     pub fn exclude(range: Range<u64>, treemap: RoaringTreemap) -> Self {
//         let length = range.end - range.start;
//         Self {
//             range,
//             offset: 0,
//             length,
//             treemap: Some(Arc::new(treemap)),
//             treemap_exclude: true,
//         }
//     }
//
//     /// Get the range of this mask
//     pub fn range(&self) -> Range<u64> {
//         assert!(self.range.end >= self.range.start + self.offset + self.length);
//         self.range.start + self.offset..self.range.start + self.offset + self.length
//     }
//
//     pub fn non_empty_range(&self, range: Range<u64>) -> bool {
//         let mask_start = self.offset;
//         let mask_end = self.offset + self.length;
//
//         let global_start = range.start + self.offset;
//         let global_end = range.end + self.offset;
//
//         if global_start >= self.range.end || global_end <= self.range.start {
//             return false;
//         }
//
//         // Check if query range overlaps with mask's effective range
//         if range.start >= self.length {
//             return false;
//         }
//
//         if let Some(treemap) = &self.treemap {
//             // Clamp the query range to the mask's range
//             let clamped_start = max(global_start, mask_start);
//             let clamped_end = min(global_end, mask_end);
//
//             if self.treemap_exclude {
//                 all_empty_treemap_range(treemap, clamped_start, clamped_end)
//             } else {
//                 non_empty_treemap_range(treemap, clamped_start, clamped_end)
//             }
//         } else if self.treemap_exclude {
//             // all exclude mask
//             false
//         } else {
//             // all include mask
//             true
//         }
//     }
//
//     pub fn slice(mut self, range: Range<u64>) -> Self {
//         let offset = range.start;
//         let length = range.end - range.start;
//
//         self.offset += offset;
//         self.length = min(length, self.length);
//         self
//     }
//
//     pub fn mask(&self) -> Mask {
//         if let Some(treemap) = &self.treemap {
//             let start = self.offset;
//             let end = self.offset + self.length;
//
//             let start_partition = (start >> 32) as u32;
//             let end_partition = (end >> 32) as u32;
//             let iter = treemap
//                 .bitmaps()
//                 .skip_while(|(key, _)| *key < start_partition)
//                 .take_while(|(key, _)| *key <= end_partition)
//                 .flat_map(|(partition_key, bitmap)| {
//                     let (low_start, low_end) = clamp_range_to_partition(partition_key, start, end);
//
//                     // Check all not values match the range
//                     bitmap.range(low_start..low_end).map(move |i| {
//                         ((((partition_key as u64) << 32) + (i as u64)) - self.offset) as usize
//                     })
//                 })
//                 .collect_vec();
//
//             if self.treemap_exclude {
//                 Mask::from_excluded_indices(self.length as usize, iter)
//             } else {
//                 Mask::from_indices(self.length as usize, iter)
//             }
//         } else if self.treemap_exclude {
//             Mask::AllFalse(self.length as usize)
//         } else {
//             let start = self.offset;
//             let end = self.offset + self.length;
//             let start = self.range.start.max(start);
//             let end = self.range.end.min(end);
//
//             if start < end {
//                 Mask::from_indices(
//                     self.length as usize,
//                     (0..self.length)
//                         .filter(|&i| i >= start && i < end)
//                         .map(|i| i as usize)
//                         .collect_vec(),
//                 )
//             } else {
//                 Mask::AllFalse(self.length as usize)
//             }
//         }
//     }
// }
//
fn clamp_range_to_partition(partition_key: u32, start: u64, end: u64) -> (u32, u32) {
    let start_partition = (start >> 32) as u32;
    let end_partition = (end >> 32) as u32;

    if partition_key == start_partition && partition_key == end_partition {
        // Same partition - use exact range
        ((start & 0xFFFFFFFF) as u32, (end & 0xFFFFFFFF) as u32)
    } else if partition_key == start_partition {
        // First partition - from start to end of partition
        ((start & 0xFFFFFFFF) as u32, u32::MAX)
    } else if partition_key == end_partition {
        // Last partition - from start of partition to end
        (0, (end & 0xFFFFFFFF) as u32)
    } else {
        // Middle partition - entire range
        (0, u32::MAX)
    }
}

#[allow(dead_code)]
fn all_empty_treemap_range(treemap: &RoaringTreemap, start: u64, end: u64) -> bool {
    let start_partition = (start >> 32) as u32;
    let end_partition = (end >> 32) as u32;

    treemap
        .bitmaps()
        .skip_while(|(key, _)| *key < start_partition)
        .take_while(|(key, _)| *key <= end_partition)
        .all(|(partition_key, bitmap)| {
            let (low_start, low_end) = clamp_range_to_partition(partition_key, start, end);

            // Check all not values match the range
            bitmap.range(low_start..low_end).next().is_none()
        })
}

#[allow(dead_code)]
fn non_empty_treemap_range(treemap: &RoaringTreemap, start: u64, end: u64) -> bool {
    // Find the upper 32 bits, the keys of the treemap.
    let start_partition = (start >> 32) as u32;
    let end_partition = (end >> 32) as u32;

    treemap
        .bitmaps()
        .skip_while(|(key, _)| *key < start_partition)
        .take_while(|(key, _)| *key <= end_partition)
        .any(|(partition_key, bitmap)| {
            let (low_start, low_end) = clamp_range_to_partition(partition_key, start, end);
            // Check if this bitmap contains any values in the range
            bitmap.range(low_start..low_end).next().is_some()
        })
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;

    #[test]
    fn test_contains_range() {
        let mask = TreeRowMask::all(101).with_range(2..101);
        assert!(mask.non_empty_range(0..101));
        assert!(!mask.non_empty_range(101..103));
        assert!(!mask.non_empty_range(0..2));
    }

    #[test]
    fn test_range_masks() {
        let mask = TreeRowMask::all(41).with_range(10..20);
        assert_eq!(
            mask.mask().collect::<VortexResult<Mask>>().unwrap(),
            Mask::from_indices(41, (10..20).collect_vec())
        );

        let mask = TreeRowMask::all(10).with_range(0..9);
        assert_eq!(
            mask.mask().collect::<VortexResult<Mask>>().unwrap(),
            Mask::from_indices(10, (0..9).collect_vec())
        );

        let mask = TreeRowMask::all(10).with_range(2..10);
        assert_eq!(
            mask.mask().collect::<VortexResult<Mask>>().unwrap(),
            Mask::from_indices(10, (2..10).collect_vec())
        );
    }

    #[test]
    fn test_subset_basic() {
        let mask = TreeRowMask::all(201).with_range(100..201);
        let subset = mask.clone().slice(10..51);
        assert_eq!(
            subset.mask().collect::<VortexResult<Mask>>().unwrap(),
            Mask::AllFalse(41)
        ); // 100 + 10
        let subset = mask.slice(100..151);
        assert_eq!(
            subset.mask().collect::<VortexResult<Mask>>().unwrap(),
            Mask::AllTrue(51)
        ); // 100 + 10
    }

    #[test]
    fn test_subset_chain() {
        let mask = TreeRowMask::all(1001);
        let subset1 = mask.slice(100..501); // Range becomes [100, 501)
        let subset2 = subset1.slice(50..151); // Range becomes [150, 251)

        assert_eq!(
            subset2.mask().collect::<VortexResult<Mask>>().unwrap(),
            Mask::AllTrue(101)
        );
    }

    #[test]
    fn test_range_and() {
        let mask = TreeRowMask::all(30).with_range(5..15);

        assert!(mask.non_empty_range(0..10));
        assert!(mask.non_empty_range(10..20));
        assert!(!mask.non_empty_range(20..30));

        assert_eq!(
            Mask::from_indices(10, (5..10).collect_vec()),
            mask.clone()
                .slice(0..10)
                .mask()
                .collect::<VortexResult<Mask>>()
                .unwrap(),
        );

        assert_eq!(
            Mask::from_indices(10, (0..5).collect_vec()),
            mask.clone()
                .slice(10..20)
                .mask()
                .collect::<VortexResult<Mask>>()
                .unwrap(),
        );

        assert_eq!(
            Mask::from_indices(10, vec![]),
            mask.slice(20..30)
                .mask()
                .collect::<VortexResult<Mask>>()
                .unwrap(),
        );
    }

    #[test]
    fn test_subset_with_treemap() {
        let mut treemap = RoaringTreemap::new();
        // Add some values in the range
        treemap.insert(105);
        treemap.insert(120);
        treemap.insert(180);

        let mask = TreeRowMask::all(1001).with_treemap(SlicedTreemap {
            treemap: Arc::new(treemap),
            offset: 0,
            length: 180,
        });

        let subset = mask.slice(100..201);
        let subset = subset.slice(10..21);

        assert_eq!(
            subset.mask().collect::<VortexResult<Mask>>().unwrap(),
            Mask::from_indices(11, vec![10])
        );
    }

    #[test]
    fn test_subset_refines_range() {
        let mut treemap = RoaringTreemap::new();
        // Original range covers 0-1000
        for i in (0..1000).step_by(10) {
            treemap.insert(i);
        }

        let original_mask = TreeRowMask::all(2001).with_treemap(SlicedTreemap {
            treemap: Arc::new(treemap),
            offset: 0,
            length: 1001,
        });
        let subset_mask = original_mask.slice(0..1001);

        // Create subset that covers positions 100-200 in the original range
        let subset_mask = subset_mask.slice(100..201);

        // Test that non_empty_range works correctly on the subset
        assert!(subset_mask.non_empty_range(10..21)); // Should find value at 110
        assert!(subset_mask.non_empty_range(50..61)); // Should find value at 150
        assert!(!subset_mask.non_empty_range(150..161)); // Outside subset range
    }

    #[test]
    fn test_subset_with_large_numbers() {
        let mut treemap = RoaringTreemap::new();
        // Use large numbers that span multiple partitions
        let base = 0x123456780000000u64;
        treemap.insert(base + 1000);
        treemap.insert(base + 2000);
        treemap.insert(base + 3000);

        let mask = TreeRowMask::all(base + 5000).with_treemap(SlicedTreemap {
            treemap: Arc::new(treemap),
            offset: 0,
            length: base + 4000,
        });
        let slice = mask.slice(base..base + 20001);
        let slice = slice.slice(500..10501);
        let slice = slice.slice(0..2001);

        assert_eq!(
            slice.mask().collect::<VortexResult<Mask>>().unwrap(),
            Mask::from_indices(2001, vec![500, 1500])
        );

        // match: base + 500 + 1500..base + 500 + 1600 == base + 1900..base + 2100
        assert!(slice.non_empty_range(1500..2001));
        // Should not find values outside the subset range
        assert!(!slice.non_empty_range(1900..2000));
        assert!(!slice.non_empty_range(2001..2500));
    }
    //
    // #[test]
    // fn test_non_empty_range_edge_cases() {
    //     let mut treemap = RoaringTreemap::new();
    //     treemap.insert(50);
    //     treemap.insert(150);
    //
    //     let mask = TreeRowMask::new(0..1001, treemap);
    //     let mask = mask.slice(0..201);
    //
    //     // Test exact boundaries
    //     assert!(mask.non_empty_range(50..51)); // Exact match
    //     assert!(mask.non_empty_range(49..52)); // Contains 50
    //     assert!(!mask.non_empty_range(51..150)); // Gap between values
    //
    //     // Test range completely outside mask bounds
    //     assert!(!mask.non_empty_range(300..401));
    //     assert!(!mask.non_empty_range(0..1)); // Before mask start (but mask starts at 0)
    // }
    //
    // #[test]
    // fn test_all_mask_behavior() {
    //     let mask = TreeRowMask::all(501).with_range(100..501);
    //
    //     // All mask should always return true for non_empty_range
    //     assert!(mask.non_empty_range(150..201));
    //     assert!(mask.non_empty_range(100..501));
    //     assert!(mask.non_empty_range(50..601)); // Even overlapping ranges
    //
    //     // But not for completely non-overlapping ranges
    //     assert!(!mask.non_empty_range(501..503));
    //     assert!(!mask.non_empty_range(600..701));
    // }
    //
    // #[test]
    // fn test_subset_boundary_conditions() {
    //     let mut treemap = RoaringTreemap::new();
    //     for i in 0..100 {
    //         treemap.insert(i);
    //     }
    //
    //     let mask = TreeRowMask::new(0..1001, treemap);
    //     let mask = mask.slice(0..101);
    //
    //     // Subset that starts at the beginning
    //     let subset1 = mask.clone().slice(0..51);
    //     assert_eq!(subset1.range().start, 0);
    //     assert_eq!(subset1.range().end, 51);
    //
    //     // Subset that ends at the end
    //     let subset2 = mask.clone().slice(50..101);
    //     assert_eq!(subset2.range().start, 50);
    //     assert_eq!(subset2.range().end, 101);
    //
    //     // Single point subset
    //     let subset3 = mask.slice(25..26);
    //     assert_eq!(subset3.range().start, 25);
    //     assert_eq!(subset3.range().end, 26);
    // }
    //
    // #[test]
    // fn test_non_empty_range_spans_partitions() {
    //     let mut treemap = RoaringTreemap::new();
    //
    //     // Insert values in two different partitions
    //     // Partition 0 (upper 32 bits = 0)
    //     treemap.insert(0xFFFFFFFF); // Last value in partition 0
    //
    //     // Partition 1 (upper 32 bits = 1)
    //     treemap.insert(0x100000000); // First value in partition 1
    //     treemap.insert(0x100000001); // Second value in partition 1
    //
    //     // Partition 2 (upper 32 bits = 2)
    //     treemap.insert(0x200000000); // First value in partition 2
    //
    //     let mask = TreeRowMask::new(0..0x400000001, treemap);
    //     let mask = mask.slice(0..0x300000001);
    //
    //     // Test range that spans from partition 0 to partition 1
    //     assert!(mask.non_empty_range(0xFFFFFFF0..0x100000011));
    //
    //     // Test range that spans from partition 1 to partition 2
    //     assert!(mask.non_empty_range(0x100000000..0x200000001));
    //
    //     // Test range in middle of partition 1 with no values
    //     assert!(!mask.non_empty_range(0x100000010..0x1FFFFFFFE));
    //
    //     // Test range that spans all three partitions
    //     assert!(mask.non_empty_range(0..0x300000001));
    //
    //     // Test empty range between values in different partitions
    //     assert!(!mask.non_empty_range(0x100000002..0x200000000));
    // }
    //
    // #[test]
    // fn test_exclude_mask() {
    //     let mut treemap = RoaringTreemap::new();
    //     treemap.insert(100);
    //     treemap.insert(500);
    //
    //     let mask = TreeRowMask::exclude(0..1001, treemap);
    //
    //     assert!(mask.non_empty_range(0..100));
    //     assert!(mask.non_empty_range(101..201));
    //     assert!(mask.non_empty_range(101..499));
    //     assert!(mask.non_empty_range(501..(u32::MAX as u64)));
    //     assert!(!mask.non_empty_range(100..101));
    //     assert!(!mask.non_empty_range(500..501));
    //     assert!(!mask.non_empty_range(0..10000));
    // }
}
