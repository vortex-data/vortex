// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use roaring::RoaringTreemap;
use vortex_buffer::Buffer;
use vortex_layout::tree_row_mask::TreeRowMask;

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
    IncludeRoaring(RoaringTreemap),
    /// A selection of rows to exclude using a [`roaring::RoaringTreemap`].
    ExcludeRoaring(RoaringTreemap),
}

impl Selection {
    pub fn tree_row_mask(&self, range: &Range<u64>) -> TreeRowMask {
        if range.start == range.end {
            return TreeRowMask::all(range.start..range.start);
        }
        match &self {
            Selection::All => TreeRowMask::all(range.clone()),
            Selection::IncludeByIndex(indices) => {
                let mut treemap = RoaringTreemap::new();
                for idx in indices.iter() {
                    treemap.insert(*idx);
                }
                TreeRowMask::new(range.clone(), treemap)
            }
            Selection::ExcludeByIndex(indices) => {
                let mut treemap = RoaringTreemap::new();
                for idx in indices.iter() {
                    treemap.insert(*idx);
                }
                TreeRowMask::exclude(range.clone(), treemap)
            }
            #[cfg(feature = "roaring")]
            Selection::IncludeRoaring(mask) => TreeRowMask::new(range.clone(), mask.clone()),
            #[cfg(feature = "roaring")]
            Selection::ExcludeRoaring(mask) => TreeRowMask::exclude(range.clone(), mask.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::Buffer;

    #[test]
    fn test_row_mask_all() {
        let selection = super::Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 5, 7]));
        let range = 1..8;
        let mask = selection.tree_row_mask(&range);

        // assert_eq!(mask.values().unwrap().indices(), &[0, 2, 4, 6]);
        range
            .map(|x| (x, [0, 2, 4, 6].contains(&x)))
            .for_each(|(x, t)| {
                if t {
                    assert!(mask.non_empty_range(x..x + 1), "x={x}");
                } else {
                    assert!(!mask.non_empty_range(x..x + 1), "x={x}");
                }
            })
    }

    // #[test]
    // fn test_row_mask_slice() {
    //     let selection = super::Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 5, 7]));
    //     let range = 3..6;
    //     let mask = selection.row_mask(&range);
    //
    //     assert_eq!(mask.values().unwrap().indices(), &[0, 2]);
    // }
    //
    // #[test]
    // fn test_row_mask_exclusive() {
    //     let selection = super::Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 5, 7]));
    //     let range = 3..5;
    //     let mask = selection.row_mask(&range);
    //
    //     assert_eq!(mask.values().unwrap().indices(), &[0]);
    // }
    //
    // #[test]
    // fn test_row_mask_all_false() {
    //     let selection = super::Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 5, 7]));
    //     let range = 8..10;
    //     let mask = selection.row_mask(&range);
    //
    //     assert!(mask.all_false());
    // }
    //
    // #[test]
    // fn test_row_mask_all_true() {
    //     let selection = super::Selection::IncludeByIndex(Buffer::from_iter(vec![1, 3, 4, 5, 6]));
    //     let range = 3..7;
    //     let mask = selection.row_mask(&range);
    //
    //     assert!(mask.all_true());
    // }
    //
    // #[test]
    // fn test_row_mask_zero() {
    //     let selection = super::Selection::IncludeByIndex(Buffer::from_iter(vec![0]));
    //     let range = 0..5;
    //     let mask = selection.row_mask(&range);
    //
    //     assert_eq!(mask.values().unwrap().indices(), &[0]);
    // }
}
