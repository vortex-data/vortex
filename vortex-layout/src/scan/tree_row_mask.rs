// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::RangeInclusive;
use std::sync::Arc;

use roaring::RoaringTreemap;

/// Uses a roaring treemap to store a set of row indices.
/// This can be refined and a row range query can be performed.
#[derive(Debug, Clone)]
pub struct TreeRowMask {
    /// The covering row range, this must be checked along with the treemap
    range: RangeInclusive<u64>,
    /// The offset into the range
    offset: u64,
    /// The length of the value offseted range
    length: u64,
    /// Empty if the treemap is all full
    treemap: Option<Arc<RoaringTreemap>>,
}

impl TreeRowMask {
    /// Create a full mask for all rows in the range
    pub fn all(range: RangeInclusive<u64>) -> Self {
        let length = range.end().saturating_sub(*range.start()).saturating_add(1);
        Self {
            range,
            offset: 0,
            length,
            treemap: None,
        }
    }

    pub fn new(range: RangeInclusive<u64>, treemap: RoaringTreemap) -> Self {
        let length = (range.end() - *range.start()).saturating_add(1);
        Self {
            range,
            offset: 0,
            length,
            treemap: Some(Arc::new(treemap)),
        }
    }

    /// Get the range of this mask
    pub fn range(&self) -> RangeInclusive<u64> {
        assert!(*self.range.end() >= self.range.start() + self.offset + self.length - 1);
        self.range.start() + self.offset..=self.range.start() + self.offset + self.length - 1
    }

    pub fn non_empty_range(&self, range: RangeInclusive<u64>) -> bool {
        let start = self.range.start() + self.offset + range.start();
        let end = self.range.start() + self.offset + range.end();
        let mask_end = self.range.start() + self.offset + self.length - 1;

        // Check if query range overlaps with mask's effective range
        if start > mask_end || end < self.range.start() + self.offset {
            return false;
        }

        // Clamp to mask boundaries
        let clamped_start = start.max(self.range.start() + self.offset);
        let clamped_end = end.min(mask_end);
        if let Some(treemap) = &self.treemap {
            // Find the upper 32 bits, the keys of the treemap.
            let start_partition = (clamped_start >> 32) as u32;
            let end_partition = (clamped_end >> 32) as u32;

            treemap
                .bitmaps()
                .skip_while(|(key, _)| *key < start_partition)
                .take_while(|(key, _)| *key <= end_partition)
                .any(|(partition_key, bitmap)| {
                    let (low_start, low_end) =
                        if partition_key == start_partition && partition_key == end_partition {
                            // Same partition - use exact range
                            (
                                (clamped_start & 0xFFFFFFFF) as u32,
                                (clamped_end & 0xFFFFFFFF) as u32,
                            )
                        } else if partition_key == start_partition {
                            // First partition - from start to end of partition
                            ((clamped_start & 0xFFFFFFFF) as u32, u32::MAX)
                        } else if partition_key == end_partition {
                            // Last partition - from start of partition to end
                            (0, (clamped_end & 0xFFFFFFFF) as u32)
                        } else {
                            // Middle partition - entire range
                            (0, u32::MAX)
                        };

                    // Check if this bitmap contains any values in the range
                    bitmap.range(low_start..=low_end).next().is_some()
                })
        } else {
            // no treemap means a full range
            true
        }
    }

    pub fn subset(mut self, range: RangeInclusive<u64>) -> Self {
        let offset = *range.start();
        let length = *range.end() - *range.start() + 1;

        assert!(
            offset + length <= self.length,
            "Subset range exceeds current mask length"
        );

        self.offset = self.offset + offset;
        self.length = length;
        self
    }
}

#[cfg(test)]
mod tests {
    use roaring::RoaringTreemap;

    use super::*;

    #[test]
    fn test_contains_range() {
        let mask = TreeRowMask::all(0..=100);
        assert!(mask.non_empty_range(0..=100));
        assert!(!mask.non_empty_range(101..=102));
    }

    #[test]
    fn test_subset_basic() {
        let mask = TreeRowMask::all(100..=200);
        let subset = mask.subset(10..=50);

        assert_eq!(*subset.range().start(), 110); // 100 + 10
        assert_eq!(*subset.range().end(), 150); // 100 + 50
    }

    #[test]
    fn test_subset_with_treemap() {
        let mut treemap = RoaringTreemap::new();
        // Add some values in the range
        treemap.insert(105);
        treemap.insert(120);
        treemap.insert(180);

        let mask = TreeRowMask::new(0..=1000, treemap);
        let subset = mask.subset(100..=200);
        let subset = subset.subset(10..=50);

        assert_eq!(*subset.range().start(), 110);
        assert_eq!(*subset.range().end(), 150);

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

        let original_mask = TreeRowMask::new(0..=2000, treemap);
        let subset_mask = original_mask.subset(0..=1000);

        // Create subset that covers positions 100-200 in the original range
        let subset_mask = subset_mask.subset(100..=200);

        assert_eq!(*subset_mask.range().start(), 100);
        assert_eq!(*subset_mask.range().end(), 200);

        // Test that non_empty_range works correctly on the subset
        assert!(subset_mask.non_empty_range(10..=20)); // Should find value at 110
        assert!(subset_mask.non_empty_range(50..=60)); // Should find value at 150
        assert!(!subset_mask.non_empty_range(250..=260)); // Outside subset range
    }

    #[test]
    fn test_subset_chain() {
        let mask = TreeRowMask::all(0..=1000);
        let subset1 = mask.subset(100..=500); // Range becomes [100, 500]
        let subset2 = subset1.subset(50..=150); // Range becomes [150, 250]

        assert_eq!(*subset2.range().start(), 150);
        assert_eq!(*subset2.range().end(), 250);
    }

    #[test]
    fn test_subset_with_large_numbers() {
        let mut treemap = RoaringTreemap::new();
        // Use large numbers that span multiple partitions
        let base = 0x123456780000000u64;
        treemap.insert(base + 1000);
        treemap.insert(base + 2000);
        treemap.insert(base + 3000);

        let mask = TreeRowMask::new(base..=base + 20000, treemap);
        let subset = mask.subset(500..=10500);
        let subset = subset.subset(0..=2000);

        assert_eq!(*subset.range().start(), base + 500);
        assert_eq!(*subset.range().end(), base + 2500);

        // match: base + 500 + 1400..base + 500 + 1600 == base + 1900..base + 2100
        assert!(subset.non_empty_range(1400..=1500));
        // Should not find values outside the subset range
        assert!(!subset.non_empty_range(1400..=1499));
        assert!(!subset.non_empty_range(1501..=2499));
    }

    #[test]
    fn test_non_empty_range_edge_cases() {
        let mut treemap = RoaringTreemap::new();
        treemap.insert(50);
        treemap.insert(150);

        let mask = TreeRowMask::new(0..=1000, treemap);
        let mask = mask.subset(0..=200);

        // Test exact boundaries
        assert!(mask.non_empty_range(50..=50)); // Exact match
        assert!(mask.non_empty_range(49..=51)); // Contains 50
        assert!(!mask.non_empty_range(51..=149)); // Gap between values

        // Test range completely outside mask bounds
        assert!(!mask.non_empty_range(300..=400));
        assert!(!mask.non_empty_range(0..=0)); // Before mask start (but mask starts at 0)
    }

    #[test]
    fn test_all_mask_behavior() {
        let mask = TreeRowMask::all(100..=500);

        // All mask should always return true for non_empty_range
        assert!(mask.non_empty_range(150..=200));
        assert!(mask.non_empty_range(100..=500));
        assert!(mask.non_empty_range(50..=600)); // Even overlapping ranges

        // But not for completely non-overlapping ranges
        assert!(!mask.non_empty_range(501..=502));
        assert!(!mask.non_empty_range(600..=700));
    }

    #[test]
    fn test_subset_boundary_conditions() {
        let mut treemap = RoaringTreemap::new();
        for i in 0..100 {
            treemap.insert(i);
        }

        let mask = TreeRowMask::new(0..=1000, treemap);
        let mask = mask.subset(0..=100);

        // Subset that starts at the beginning
        let subset1 = mask.clone().subset(0..=50);
        assert_eq!(*subset1.range().start(), 0);
        assert_eq!(*subset1.range().end(), 50);

        // Subset that ends at the end
        let subset2 = mask.clone().subset(50..=100);
        assert_eq!(*subset2.range().start(), 50);
        assert_eq!(*subset2.range().end(), 100);

        // Single point subset
        let subset3 = mask.subset(25..=25);
        assert_eq!(*subset3.range().start(), 25);
        assert_eq!(*subset3.range().end(), 25);
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

        let mask = TreeRowMask::new(0..=0x400000000, treemap);
        let mask = mask.subset(0..=0x300000000);

        // Test range that spans from partition 0 to partition 1
        assert!(mask.non_empty_range(0xFFFFFFF0..=0x100000010));

        // Test range that spans from partition 1 to partition 2
        assert!(mask.non_empty_range(0x100000000..=0x200000000));

        // Test range in middle of partition 1 with no values
        assert!(!mask.non_empty_range(0x100000010..=0x1FFFFFFFE));

        // Test range that spans all three partitions
        assert!(mask.non_empty_range(0..=0x300000000));

        // Test empty range between values in different partitions
        assert!(!mask.non_empty_range(0x100000002..=0x1FFFFFFFF));
    }
}
