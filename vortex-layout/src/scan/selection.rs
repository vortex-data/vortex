use std::ops::{Not, Range};

use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_mask::Mask;

use crate::scan::row_mask::RowMask;

/// A selection identifies a set of rows to include in the scan (in addition to applying any
/// filter predicates).
#[derive(Default)]
pub enum Selection {
    /// No selection, all rows are included.
    #[default]
    All,
    /// A selection of rows to include by index.
    IncludeByIndex(Buffer<u64>),
    /// A selection of rows to exclude by index.
    ExcludeByIndex(Buffer<u64>),
    /// A selection of rows to include using a [`roaring::RoaringTreemap`].
    #[cfg(feature = "roaring")]
    IncludeRoaring(roaring::RoaringTreemap),
    /// A selection of rows to exclude using a [`roaring::RoaringTreemap`].
    #[cfg(feature = "roaring")]
    ExcludeRoaring(roaring::RoaringTreemap),
}

impl Selection {
    /// Extract the [`RowMask`] for the given range from this selection.
    pub(crate) fn row_mask(&self, range: &Range<u64>) -> RowMask {
        let range_len = usize::try_from(range.end - range.start)
            .vortex_expect("Range length does not fit into a usize");

        match self {
            Selection::All => RowMask::new(range.start, Mask::new_true(range_len)),
            Selection::IncludeByIndex(include) => {
                let mask = indices_range(range, include)
                    .map(|idx_range| {
                        Mask::from_indices(
                            range_len,
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
                    .unwrap_or_else(|| Mask::new_false(range_len));

                RowMask::new(range.start, mask)
            }
            Selection::ExcludeByIndex(exclude) => {
                let mask = Selection::IncludeByIndex(exclude.clone())
                    .row_mask(range)
                    .mask()
                    .clone();
                RowMask::new(range.start, mask.not())
            }
            #[cfg(feature = "roaring")]
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
                        .map(|idx| idx - range.start)
                        .map(|idx| {
                            usize::try_from(idx).vortex_expect("Index does not fit into a usize")
                        })
                        .collect(),
                );

                RowMask::new(range.start, mask)
            }
            #[cfg(feature = "roaring")]
            Selection::ExcludeRoaring(roaring) => {
                use std::ops::BitAnd;

                let mut range_treemap = roaring::RoaringTreemap::new();
                range_treemap.insert_range(range.clone());

                // If there are no deletions in the intersection, then we have an all true mask.
                if roaring.intersection_len(&range_treemap) == range_len as u64 {
                    return RowMask::new(range.start, Mask::new_true(range_len));
                }

                // Otherwise, intersect with the selected range and shift to relativize.
                let roaring = roaring.bitand(range_treemap);
                let mask = Mask::from_excluded_indices(
                    range_len,
                    roaring.iter().map(|idx| idx - range.start).map(|idx| {
                        usize::try_from(idx).vortex_expect("Index does not fit into a usize")
                    }),
                );

                RowMask::new(range.start, mask)
            }
        }
    }
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

    #[test]
    fn test_row_mask_all() {
        let selection = super::Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 5, 7]));
        let range = 1..8;
        let row_mask = selection.row_mask(&range);

        assert_eq!(row_mask.mask().values().unwrap().indices(), &[0, 2, 4, 6]);
    }

    #[test]
    fn test_row_mask_slice() {
        let selection = super::Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 5, 7]));
        let range = 3..6;
        let row_mask = selection.row_mask(&range);

        assert_eq!(row_mask.mask().values().unwrap().indices(), &[0, 2]);
    }

    #[test]
    fn test_row_mask_exclusive() {
        let selection = super::Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 5, 7]));
        let range = 3..5;
        let row_mask = selection.row_mask(&range);

        assert_eq!(row_mask.mask().values().unwrap().indices(), &[0]);
    }

    #[test]
    fn test_row_mask_all_false() {
        let selection = super::Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 5, 7]));
        let range = 8..10;
        let row_mask = selection.row_mask(&range);

        assert!(row_mask.mask().all_false());
    }

    #[test]
    fn test_row_mask_all_true() {
        let selection = super::Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 4, 5, 6]));
        let range = 3..7;
        let row_mask = selection.row_mask(&range);

        assert!(row_mask.mask().all_true());
    }

    #[test]
    fn test_row_mask_zero() {
        let selection = super::Selection::IncludeByIndex(Buffer::from_iter(vec![0]));
        let range = 0..5;
        let row_mask = selection.row_mask(&range);

        assert_eq!(row_mask.mask().values().unwrap().indices(), &[0]);
    }
}
