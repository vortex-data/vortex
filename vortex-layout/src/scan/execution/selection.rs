// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Resolves known selections before allocating a chunk's planner or its masks.

use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::selection::Selection;

use super::SelectedRows;

/// The selected rows of a chunk, bounded to their span, with one mask per chunk.
///
/// `kept` is the mask zone pruning left over the scan's rows, with the row its first bit
/// stands for. A chunk none of whose rows were selected and kept has no rows.
pub(super) fn select_chunk(
    scope: WorkScope,
    selection: &Selection,
    kept: Option<(&Mask, u64)>,
) -> Option<SelectedRows> {
    let selected = select_rows(scope, selection)?;
    let Some((kept, first_row)) = kept else {
        return Some(selected);
    };
    let offset = usize::try_from(selected.scope.rows.start - first_row)
        .vortex_expect("a work chunk fits usize");
    let kept = kept.slice(offset..offset + selected.mask.len());
    SelectedRows::narrowed(selected.scope, &selected.mask & &kept)
}

impl SelectedRows {
    /// The rows of `scope` that `mask`, over all of them, selects, bounded to their span, or
    /// `None` when it selects none.
    pub(super) fn narrowed(mut scope: WorkScope, mask: Mask) -> Option<Self> {
        let first = mask.first()?;
        let end = mask.last()? + 1;
        scope.rows = scope.rows.start + first as u64..scope.rows.start + end as u64;
        Some(Self {
            scope,
            mask: mask.slice(first..end),
        })
    }
}

fn select_rows(mut scope: WorkScope, selection: &Selection) -> Option<SelectedRows> {
    let count = match selection {
        Selection::IncludeByIndex(indices) => {
            let first = indices.partition_point(|&row| row < scope.rows.start);
            let end = indices.partition_point(|&row| row < scope.rows.end);
            if first == end {
                return None;
            }
            scope.rows = indices[first]..indices[end - 1] + 1;
            (end - first) as u64
        }
        _ => selection.row_count_in_range(&scope.rows),
    };
    if count == 0 {
        return None;
    }
    // Contiguous selections need neither a bitmap nor a filtering operator.
    let mask = if count == scope.rows.end - scope.rows.start {
        Mask::new_true(usize::try_from(count).vortex_expect("a work chunk fits usize"))
    } else {
        selection.row_mask(&scope.rows).mask().clone()
    };
    SelectedRows::narrowed(scope, mask)
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;

    use super::*;

    #[rstest]
    #[case::point(vec![7], 7..8, true)]
    #[case::contiguous(vec![7, 8, 9], 7..10, true)]
    #[case::sparse(vec![7, 9], 7..10, false)]
    #[case::clipped(vec![0, 7, 9, 100], 7..10, false)]
    fn indexed_selection_is_bounded_before_mask_allocation(
        #[case] indices: Vec<u64>,
        #[case] expected: Range<u64>,
        #[case] all_true: bool,
    ) -> VortexResult<()> {
        let selection =
            Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::from(indices))?);
        let selected = select_rows(
            WorkScope {
                file_ordinal: 3,
                rows: 5..20,
            },
            &selection,
        )
        .expect("selected rows");
        assert_eq!(selected.scope.file_ordinal, 3);
        assert_eq!(selected.scope.rows, expected);
        assert_eq!(selected.mask.all_true(), all_true);
        Ok(())
    }

    #[test]
    fn excluded_rows_can_prove_the_whole_chunk_empty() -> VortexResult<()> {
        let selection =
            Selection::ExcludeByIndex(StrictSortedBuffer::try_new(Buffer::from(vec![5, 6, 7]))?);
        assert!(
            select_rows(
                WorkScope {
                    file_ordinal: 0,
                    rows: 5..8
                },
                &selection
            )
            .is_none()
        );
        Ok(())
    }
}
