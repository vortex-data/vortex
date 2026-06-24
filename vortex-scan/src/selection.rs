// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Defines a selection mask over a scan.

use std::ops::Not;
use std::ops::Range;

use vortex_buffer::Buffer;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::row_mask::RowMask;

/// A buffer whose values are known to be strictly sorted in ascending order.
#[derive(Clone, Debug)]
pub struct StrictSortedBuffer<T> {
    buffer: Buffer<T>,
}

impl<T> StrictSortedBuffer<T> {
    /// Return the sorted values.
    pub fn as_slice(&self) -> &[T] {
        self.buffer.as_slice()
    }

    /// Return the sorted buffer.
    pub fn into_inner(self) -> Buffer<T> {
        self.buffer
    }

    /// Return true if the buffer contains no values.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Return the number of values in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl<T: Ord> StrictSortedBuffer<T> {
    /// Create a new buffer, failing if the values are not strictly increasing.
    pub fn try_new(buffer: Buffer<T>) -> VortexResult<Self> {
        validate_strictly_sorted(buffer.as_slice())?;
        Ok(Self { buffer })
    }
}

impl<T> Default for StrictSortedBuffer<T> {
    fn default() -> Self {
        Self {
            buffer: Buffer::default(),
        }
    }
}

impl<T: Ord> TryFrom<Buffer<T>> for StrictSortedBuffer<T> {
    type Error = VortexError;

    fn try_from(value: Buffer<T>) -> Result<Self, Self::Error> {
        Self::try_new(value)
    }
}

impl<T> From<StrictSortedBuffer<T>> for Buffer<T> {
    fn from(value: StrictSortedBuffer<T>) -> Self {
        value.into_inner()
    }
}

impl<T> std::ops::Deref for StrictSortedBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> AsRef<[T]> for StrictSortedBuffer<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

/// A selection identifies a set of rows to include in the scan (in addition to applying any
/// filter predicates).
#[derive(Default, Clone, Debug)]
pub enum Selection {
    /// No selection, all rows are included.
    #[default]
    All,
    /// A selection of sorted, unique rows to include by index.
    IncludeByIndex(StrictSortedBuffer<u64>),
    /// A selection of sorted, unique rows to exclude by index.
    ExcludeByIndex(StrictSortedBuffer<u64>),
    /// A selection of rows to include using a [`roaring::RoaringTreemap`].
    IncludeRoaring(roaring::RoaringTreemap),
    /// A selection of rows to exclude using a [`roaring::RoaringTreemap`].
    ExcludeRoaring(roaring::RoaringTreemap),
}

impl Selection {
    /// Return the row count for this selection.
    pub fn row_count(&self, total_rows: u64) -> u64 {
        match self {
            Selection::All => total_rows,
            Selection::IncludeByIndex(include) => include.len() as u64,
            Selection::ExcludeByIndex(exclude) => total_rows.saturating_sub(exclude.len() as u64),
            Selection::IncludeRoaring(roaring) => roaring.len(),
            Selection::ExcludeRoaring(roaring) => total_rows.saturating_sub(roaring.len()),
        }
    }

    /// Extract the [`RowMask`] for the given range from this selection.
    pub fn row_mask(&self, range: &Range<u64>) -> RowMask {
        if range.start >= range.end {
            return RowMask::new(0, Mask::AllFalse(0));
        }

        // Saturating subtraction to prevent underflow, though range should be valid
        let range_diff = range.end.saturating_sub(range.start);
        let range_len = usize::try_from(range_diff).unwrap_or_else(|_| {
            // If the range is too large for usize, cap it at usize::MAX
            // This is a defensive measure; in practice, ranges should be reasonable
            tracing::warn!(
                "Range length {} exceeds usize::MAX, capping at usize::MAX",
                range_diff
            );
            usize::MAX
        });

        match self {
            Selection::All => RowMask::new(range.start, Mask::new_true(range_len)),
            Selection::IncludeByIndex(include) => {
                let indices = include.as_slice();
                let mask = indices_range(range, indices)
                    .map(|idx_range| {
                        Mask::from_indices(
                            range_len,
                            indices[idx_range]
                                .iter()
                                .map(|idx| {
                                    idx.checked_sub(range.start).unwrap_or_else(|| {
                                        vortex_panic!(
                                            "index underflow, range: {:?}, idx: {:?}",
                                            range,
                                            idx
                                        )
                                    })
                                })
                                .filter_map(|idx| {
                                    // Only include indices that fit in usize
                                    usize::try_from(idx).ok()
                                }),
                        )
                    })
                    .unwrap_or_else(|| Mask::new_false(range_len));

                RowMask::new(range.start, mask)
            }
            Selection::ExcludeByIndex(exclude) => {
                let indices = exclude.as_slice();
                let mask = indices_range(range, indices)
                    .map(|idx_range| {
                        Mask::from_indices(
                            range_len,
                            indices[idx_range]
                                .iter()
                                .map(|idx| {
                                    idx.checked_sub(range.start).unwrap_or_else(|| {
                                        vortex_panic!(
                                            "index underflow, range: {:?}, idx: {:?}",
                                            range,
                                            idx
                                        )
                                    })
                                })
                                .filter_map(|idx| usize::try_from(idx).ok()),
                        )
                    })
                    .unwrap_or_else(|| Mask::new_false(range_len));
                RowMask::new(range.start, mask.not())
            }
            Selection::IncludeRoaring(roaring) => {
                use std::ops::BitAnd;

                // First we perform a cheap is_disjoint check
                let mut range_treemap = roaring::RoaringTreemap::new();
                range_treemap.insert_range(range.clone());

                if roaring.is_disjoint(&range_treemap) {
                    return RowMask::new(range.start, Mask::new_false(range_len));
                }

                // Otherwise, intersect with the selected range and shift to relativize.
                let roaring = roaring.bitand(range_treemap);
                let mask = Mask::from_indices(
                    range_len,
                    roaring
                        .iter()
                        .map(|idx| {
                            idx.checked_sub(range.start).unwrap_or_else(|| {
                                vortex_panic!("index underflow, range: {:?}, idx: {:?}", range, idx)
                            })
                        })
                        .filter_map(|idx| {
                            // Only include indices that fit in usize
                            usize::try_from(idx).ok()
                        }),
                );

                RowMask::new(range.start, mask)
            }
            Selection::ExcludeRoaring(roaring) => {
                use std::ops::BitAnd;

                let mut range_treemap = roaring::RoaringTreemap::new();
                range_treemap.insert_range(range.clone());

                // If all indices in range are excluded, return all false mask
                if roaring.intersection_len(&range_treemap) == range_len as u64 {
                    return RowMask::new(range.start, Mask::new_false(range_len));
                }

                // Otherwise, intersect with the selected range and shift to relativize.
                let roaring = roaring.bitand(range_treemap);
                let mask = Mask::from_excluded_indices(
                    range_len,
                    roaring
                        .iter()
                        .map(|idx| {
                            idx.checked_sub(range.start).unwrap_or_else(|| {
                                vortex_panic!("index underflow, range: {:?}, idx: {:?}", range, idx)
                            })
                        })
                        .filter_map(|idx| usize::try_from(idx).ok()),
                );

                RowMask::new(range.start, mask)
            }
        }
    }
}

fn validate_strictly_sorted<T: Ord>(values: &[T]) -> VortexResult<()> {
    for (idx, window) in values.windows(2).enumerate() {
        if window[0] >= window[1] {
            vortex_bail!(
                "buffer values must be strictly increasing at positions {} and {}",
                idx,
                idx + 1
            );
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use vortex_buffer::Buffer;

    use super::Selection;
    use super::StrictSortedBuffer;

    fn strict_sorted(indices: impl IntoIterator<Item = u64>) -> StrictSortedBuffer<u64> {
        StrictSortedBuffer::try_new(Buffer::from_iter(indices))
            .expect("test indices should be strictly increasing")
    }

    fn include(indices: impl IntoIterator<Item = u64>) -> Selection {
        Selection::IncludeByIndex(strict_sorted(indices))
    }

    fn exclude(indices: impl IntoIterator<Item = u64>) -> Selection {
        Selection::ExcludeByIndex(strict_sorted(indices))
    }

    #[test]
    fn strict_sorted_buffer_rejects_unsorted_values() {
        let err = StrictSortedBuffer::try_new(Buffer::from_iter([3, 1])).unwrap_err();
        assert!(err.to_string().contains("strictly increasing"));
    }

    #[test]
    fn strict_sorted_buffer_rejects_duplicate_values() {
        let err = StrictSortedBuffer::try_new(Buffer::from_iter([1, 1])).unwrap_err();
        assert!(err.to_string().contains("strictly increasing"));
    }

    #[test]
    fn test_row_mask_all() {
        let selection = include([1, 3, 5, 7]);
        let range = 1..8;
        let row_mask = selection.row_mask(&range);

        assert_eq!(row_mask.mask().values().unwrap().indices(), &[0, 2, 4, 6]);
    }

    #[test]
    fn test_row_mask_slice() {
        let selection = include([1, 3, 5, 7]);
        let range = 3..6;
        let row_mask = selection.row_mask(&range);

        assert_eq!(row_mask.mask().values().unwrap().indices(), &[0, 2]);
    }

    #[test]
    fn test_row_mask_exclusive() {
        let selection = include([1, 3, 5, 7]);
        let range = 3..5;
        let row_mask = selection.row_mask(&range);

        assert_eq!(row_mask.mask().values().unwrap().indices(), &[0]);
    }

    #[test]
    fn test_row_mask_all_false() {
        let selection = include([1, 3, 5, 7]);
        let range = 8..10;
        let row_mask = selection.row_mask(&range);

        assert!(row_mask.mask().all_false());
    }

    #[test]
    fn test_row_mask_all_true() {
        let selection = include([1, 3, 4, 5, 6]);
        let range = 3..7;
        let row_mask = selection.row_mask(&range);

        assert!(row_mask.mask().all_true());
    }

    #[test]
    fn test_row_mask_zero() {
        let selection = include([0]);
        let range = 0..5;
        let row_mask = selection.row_mask(&range);

        assert_eq!(row_mask.mask().values().unwrap().indices(), &[0]);
    }

    mod roaring_tests {
        use roaring::RoaringTreemap;

        use super::*;

        #[test]
        fn test_roaring_include_basic() {
            let mut roaring = RoaringTreemap::new();
            roaring.insert(1);
            roaring.insert(3);
            roaring.insert(5);
            roaring.insert(7);

            let selection = Selection::IncludeRoaring(roaring);
            let range = 1..8;
            let row_mask = selection.row_mask(&range);

            assert_eq!(row_mask.mask().values().unwrap().indices(), &[0, 2, 4, 6]);
        }

        #[test]
        fn test_roaring_include_slice() {
            let mut roaring = RoaringTreemap::new();
            roaring.insert(1);
            roaring.insert(3);
            roaring.insert(5);
            roaring.insert(7);

            let selection = Selection::IncludeRoaring(roaring);
            let range = 3..6;
            let row_mask = selection.row_mask(&range);

            assert_eq!(row_mask.mask().values().unwrap().indices(), &[0, 2]);
        }

        #[test]
        fn test_roaring_include_disjoint() {
            let mut roaring = RoaringTreemap::new();
            roaring.insert(1);
            roaring.insert(3);
            roaring.insert(5);
            roaring.insert(7);

            let selection = Selection::IncludeRoaring(roaring);
            let range = 8..10;
            let row_mask = selection.row_mask(&range);

            assert!(row_mask.mask().all_false());
        }

        #[test]
        fn test_roaring_include_large_range() {
            let mut roaring = RoaringTreemap::new();
            // Insert a large number of indices
            for i in (0..1000000).step_by(2) {
                roaring.insert(i);
            }

            let selection = Selection::IncludeRoaring(roaring);
            let range = 1000..2000;
            let row_mask = selection.row_mask(&range);

            // Should have 500 selected indices (every even number)
            assert_eq!(row_mask.mask().true_count(), 500);
        }

        #[test]
        fn test_roaring_exclude_basic() {
            let mut roaring = RoaringTreemap::new();
            roaring.insert(1);
            roaring.insert(3);
            roaring.insert(5);

            let selection = Selection::ExcludeRoaring(roaring);
            let range = 0..7;
            let row_mask = selection.row_mask(&range);

            // Should exclude indices 1, 3, 5, so we get 0, 2, 4, 6
            assert_eq!(row_mask.mask().values().unwrap().indices(), &[0, 2, 4, 6]);
        }

        #[test]
        fn test_roaring_exclude_all() {
            let mut roaring = RoaringTreemap::new();
            // Exclude all indices in range
            for i in 10..20 {
                roaring.insert(i);
            }

            let selection = Selection::ExcludeRoaring(roaring);
            let range = 10..20;
            let row_mask = selection.row_mask(&range);

            assert!(row_mask.mask().all_false());
        }

        #[test]
        fn test_roaring_exclude_none() {
            let mut roaring = RoaringTreemap::new();
            roaring.insert(100);
            roaring.insert(101);

            let selection = Selection::ExcludeRoaring(roaring);
            let range = 0..10;
            let row_mask = selection.row_mask(&range);

            // Nothing to exclude in this range
            assert!(row_mask.mask().all_true());
        }

        #[test]
        fn test_roaring_exclude_partial() {
            let mut roaring = RoaringTreemap::new();
            roaring.insert(5);
            roaring.insert(6);
            roaring.insert(7);
            roaring.insert(15); // Outside range

            let selection = Selection::ExcludeRoaring(roaring);
            let range = 5..10;
            let row_mask = selection.row_mask(&range);

            // Should exclude 5, 6, 7 (mapped to 0, 1, 2), keep 8, 9 (mapped to 3, 4)
            assert_eq!(row_mask.mask().values().unwrap().indices(), &[3, 4]);
        }

        #[test]
        fn test_roaring_include_empty() {
            let roaring = RoaringTreemap::new();
            let selection = Selection::IncludeRoaring(roaring);
            let range = 0..100;
            let row_mask = selection.row_mask(&range);

            assert!(row_mask.mask().all_false());
        }

        #[test]
        fn test_roaring_exclude_empty() {
            let roaring = RoaringTreemap::new();
            let selection = Selection::ExcludeRoaring(roaring);
            let range = 0..100;
            let row_mask = selection.row_mask(&range);

            assert!(row_mask.mask().all_true());
        }

        #[test]
        fn test_roaring_include_boundary() {
            let mut roaring = RoaringTreemap::new();
            roaring.insert(0);
            roaring.insert(99);

            let selection = Selection::IncludeRoaring(roaring);
            let range = 0..100;
            let row_mask = selection.row_mask(&range);

            assert_eq!(row_mask.mask().values().unwrap().indices(), &[0, 99]);
        }

        #[test]
        fn test_roaring_include_range_insertion() {
            let mut roaring = RoaringTreemap::new();
            // Use insert_range for efficiency
            roaring.insert_range(10..20);
            roaring.insert_range(30..40);

            let selection = Selection::IncludeRoaring(roaring);
            let range = 15..35;
            let row_mask = selection.row_mask(&range);

            // Should include 15-19 (mapped to 0-4) and 30-34 (mapped to 15-19)
            let expected: Vec<usize> = (0..5).chain(15..20).collect();
            assert_eq!(row_mask.mask().values().unwrap().indices(), &expected);
        }

        #[test]
        fn test_roaring_overflow_protection() {
            let mut roaring = RoaringTreemap::new();
            // Insert very large indices
            roaring.insert(u64::MAX - 1);
            roaring.insert(u64::MAX);

            let selection = Selection::IncludeRoaring(roaring);
            let range = u64::MAX - 10..u64::MAX;
            let row_mask = selection.row_mask(&range);

            // Should handle overflow gracefully
            assert_eq!(row_mask.mask().true_count(), 1); // Only u64::MAX - 1 is in range
        }

        #[test]
        fn test_roaring_exclude_overflow_protection() {
            let mut roaring = RoaringTreemap::new();
            roaring.insert(u64::MAX - 1);

            let selection = Selection::ExcludeRoaring(roaring);
            let range = u64::MAX - 10..u64::MAX;
            let row_mask = selection.row_mask(&range);

            // Should handle overflow gracefully, excluding index u64::MAX - 1
            assert_eq!(row_mask.mask().true_count(), 9); // All except one
        }

        #[test]
        fn test_roaring_include_vs_buffer_equivalence() {
            // Test that RoaringTreemap and Buffer produce same results
            let indices = vec![1, 3, 5, 7, 9];

            let buffer_selection = include(indices.clone());

            let mut roaring = RoaringTreemap::new();
            for idx in &indices {
                roaring.insert(*idx);
            }
            let roaring_selection = Selection::IncludeRoaring(roaring);

            let range = 0..12;
            let buffer_mask = buffer_selection.row_mask(&range);
            let roaring_mask = roaring_selection.row_mask(&range);

            assert_eq!(
                buffer_mask.mask().values().unwrap().indices(),
                roaring_mask.mask().values().unwrap().indices()
            );
        }

        #[test]
        fn test_roaring_exclude_vs_buffer_equivalence() {
            // Test that ExcludeRoaring and ExcludeByIndex produce same results
            let indices = vec![2, 4, 6, 8];

            let buffer_selection = exclude(indices.clone());

            let mut roaring = RoaringTreemap::new();
            for idx in &indices {
                roaring.insert(*idx);
            }
            let roaring_selection = Selection::ExcludeRoaring(roaring);

            let range = 0..10;
            let buffer_mask = buffer_selection.row_mask(&range);
            let roaring_mask = roaring_selection.row_mask(&range);

            assert_eq!(
                buffer_mask.mask().values().unwrap().indices(),
                roaring_mask.mask().values().unwrap().indices()
            );
        }
    }
}
