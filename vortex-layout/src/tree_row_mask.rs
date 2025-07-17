// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::{max, min};
use std::ops::Range;
use std::sync::Arc;

use roaring::RoaringTreemap;

/// Uses a roaring treemap to store a set of row indices.
/// This can be refined and a row range query can be performed.
#[derive(Debug, Clone)]
pub struct TreeRowMask {
    /// The covering row range, this must be checked along with the treemap
    range: Range<u64>,
    /// The offset into the range
    offset: u64,
    /// The length of the value offseted range
    length: u64,
    /// Empty if the treemap is all full
    treemap: Option<Arc<RoaringTreemap>>,
    /// True if the treemap is used for exclude, not include
    treemap_exclude: bool,
}

impl TreeRowMask {
    /// Create a full mask for all rows in the range
    pub fn all(range: Range<u64>) -> Self {
        let length = range.end.saturating_sub(range.start);
        Self {
            range,
            offset: 0,
            length,
            treemap: None,
            treemap_exclude: false,
        }
    }

    pub fn new(range: Range<u64>, treemap: RoaringTreemap) -> Self {
        let length = range.end.saturating_sub(range.start);
        Self {
            range,
            offset: 0,
            length,
            treemap: Some(Arc::new(treemap)),
            treemap_exclude: false,
        }
    }

    pub fn exclude(range: Range<u64>, treemap: RoaringTreemap) -> Self {
        let length = range.end.saturating_sub(range.start);
        Self {
            range,
            offset: 0,
            length,
            treemap: Some(Arc::new(treemap)),
            treemap_exclude: true,
        }
    }

    /// Get the range of this mask
    pub fn range(&self) -> Range<u64> {
        assert!(self.range.end >= self.range.start + self.offset + self.length);
        self.range.start + self.offset..self.range.start + self.offset + self.length
    }

    pub fn non_empty_range(&self, range: Range<u64>) -> bool {
        let mask_start = self.range.start + self.offset;
        let mask_end = self.range.start + self.offset + self.length;

        // Check if query range overlaps with mask's effective range
        if range.start >= mask_end || range.end <= mask_start {
            return false;
        }

        if let Some(treemap) = &self.treemap {
            // Clamp the query range to the mask's range
            let clamped_start = max(range.start, mask_start);
            let clamped_end = min(range.end, mask_end);

            if self.treemap_exclude {
                all_empty_treemap_range(treemap, clamped_start, clamped_end)
            } else {
                non_empty_treemap_range(treemap, clamped_start, clamped_end)
            }
        } else if self.treemap_exclude {
            // all exclude mask
            false
        } else {
            // all include mask
            true
        }
    }

    pub fn slice(mut self, range: Range<u64>) -> Self {
        let offset = range.start;
        let length = range.end - range.start;

        self.offset += offset;
        self.length = min(length, self.length);
        self
    }
}

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
    use roaring::RoaringTreemap;

    use super::*;

    #[test]
    fn test_contains_range() {
        let mask = TreeRowMask::all(2..101);
        assert!(mask.non_empty_range(0..101));
        assert!(!mask.non_empty_range(101..103));
        assert!(!mask.non_empty_range(0..2));
    }

    #[test]
    fn test_subset_basic() {
        let mask = TreeRowMask::all(100..201);
        let subset = mask.slice(10..51);

        assert_eq!(subset.range().start, 110); // 100 + 10
        assert_eq!(subset.range().end, 151); // 100 + 51
    }

    #[test]
    fn test_subset_with_treemap() {
        let mut treemap = RoaringTreemap::new();
        // Add some values in the range
        treemap.insert(105);
        treemap.insert(120);
        treemap.insert(180);

        let mask = TreeRowMask::new(0..1001, treemap);
        let subset = mask.slice(100..201);
        let subset = subset.slice(10..51);

        assert_eq!(subset.range().start, 110);
        assert_eq!(subset.range().end, 151);

        // Should still have the same treemap reference
        assert!(subset.treemap.is_some());
    }

    #[test]
    fn test_subset_refines_range() {
        let mut treemap = RoaringTreemap::new();
        // Original range covers 0-1000
        for i in (0..1000).step_by(10) {
            treemap.insert(i);
        }

        let original_mask = TreeRowMask::new(0..2001, treemap);
        let subset_mask = original_mask.slice(0..1001);

        // Create subset that covers positions 100-200 in the original range
        let subset_mask = subset_mask.slice(100..201);

        assert_eq!(subset_mask.range().start, 100);
        assert_eq!(subset_mask.range().end, 201);

        // Test that non_empty_range works correctly on the subset
        assert!(subset_mask.non_empty_range(110..121)); // Should find value at 110
        assert!(subset_mask.non_empty_range(150..161)); // Should find value at 150
        assert!(!subset_mask.non_empty_range(250..261)); // Outside subset range
    }

    #[test]
    fn test_subset_chain() {
        let mask = TreeRowMask::all(0..1001);
        let subset1 = mask.slice(100..501); // Range becomes [100, 501)
        let subset2 = subset1.slice(50..151); // Range becomes [150, 251)

        assert_eq!(subset2.range().start, 150);
        assert_eq!(subset2.range().end, 251);
    }

    #[test]
    fn test_subset_with_large_numbers() {
        let mut treemap = RoaringTreemap::new();
        // Use large numbers that span multiple partitions
        let base = 0x123456780000000u64;
        treemap.insert(base + 1000);
        treemap.insert(base + 2000);
        treemap.insert(base + 3000);

        let mask = TreeRowMask::new(base..base + 20001, treemap);
        let subset = mask.slice(500..10501);
        let subset = subset.slice(0..2001);

        assert_eq!(subset.range().start, base + 500);
        assert_eq!(subset.range().end, base + 2501);

        // match: base + 500 + 1400..base + 500 + 1600 == base + 1900..base + 2100
        assert!(subset.non_empty_range(base + 1900..base + 2001));
        // Should not find values outside the subset range
        assert!(!subset.non_empty_range(base + 1900..base + 2000));
        assert!(!subset.non_empty_range(base + 2001..base + 2500));
    }

    #[test]
    fn test_non_empty_range_edge_cases() {
        let mut treemap = RoaringTreemap::new();
        treemap.insert(50);
        treemap.insert(150);

        let mask = TreeRowMask::new(0..1001, treemap);
        let mask = mask.slice(0..201);

        // Test exact boundaries
        assert!(mask.non_empty_range(50..51)); // Exact match
        assert!(mask.non_empty_range(49..52)); // Contains 50
        assert!(!mask.non_empty_range(51..150)); // Gap between values

        // Test range completely outside mask bounds
        assert!(!mask.non_empty_range(300..401));
        assert!(!mask.non_empty_range(0..1)); // Before mask start (but mask starts at 0)
    }

    #[test]
    fn test_all_mask_behavior() {
        let mask = TreeRowMask::all(100..501);

        // All mask should always return true for non_empty_range
        assert!(mask.non_empty_range(150..201));
        assert!(mask.non_empty_range(100..501));
        assert!(mask.non_empty_range(50..601)); // Even overlapping ranges

        // But not for completely non-overlapping ranges
        assert!(!mask.non_empty_range(501..503));
        assert!(!mask.non_empty_range(600..701));
    }

    #[test]
    fn test_subset_boundary_conditions() {
        let mut treemap = RoaringTreemap::new();
        for i in 0..100 {
            treemap.insert(i);
        }

        let mask = TreeRowMask::new(0..1001, treemap);
        let mask = mask.slice(0..101);

        // Subset that starts at the beginning
        let subset1 = mask.clone().slice(0..51);
        assert_eq!(subset1.range().start, 0);
        assert_eq!(subset1.range().end, 51);

        // Subset that ends at the end
        let subset2 = mask.clone().slice(50..101);
        assert_eq!(subset2.range().start, 50);
        assert_eq!(subset2.range().end, 101);

        // Single point subset
        let subset3 = mask.slice(25..26);
        assert_eq!(subset3.range().start, 25);
        assert_eq!(subset3.range().end, 26);
    }

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

        let mask = TreeRowMask::new(0..0x400000001, treemap);
        let mask = mask.slice(0..0x300000001);

        // Test range that spans from partition 0 to partition 1
        assert!(mask.non_empty_range(0xFFFFFFF0..0x100000011));

        // Test range that spans from partition 1 to partition 2
        assert!(mask.non_empty_range(0x100000000..0x200000001));

        // Test range in middle of partition 1 with no values
        assert!(!mask.non_empty_range(0x100000010..0x1FFFFFFFE));

        // Test range that spans all three partitions
        assert!(mask.non_empty_range(0..0x300000001));

        // Test empty range between values in different partitions
        assert!(!mask.non_empty_range(0x100000002..0x200000000));
    }

    #[test]
    fn test_exclude_mask() {
        let mut treemap = RoaringTreemap::new();
        treemap.insert(100);
        treemap.insert(500);

        let mask = TreeRowMask::exclude(0..1001, treemap);

        assert!(mask.non_empty_range(0..100));
        assert!(mask.non_empty_range(101..201));
        assert!(mask.non_empty_range(101..499));
        assert!(mask.non_empty_range(501..(u32::MAX as u64)));
        assert!(!mask.non_empty_range(100..101));
        assert!(!mask.non_empty_range(500..501));
        assert!(!mask.non_empty_range(0..10000));
    }
}
