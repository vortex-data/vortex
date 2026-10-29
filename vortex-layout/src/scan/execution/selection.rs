// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Resolves known selections before allocating a chunk's planner or its masks.

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::selection::Selection;

use super::PreparedScan;
use super::SelectedRows;
use crate::plan::physical::pipeline::SegmentLease;

/// Plans only the selected span of a chunk, with one planner and one mask per chunk.
///
/// `kept` is the mask zone pruning left over the scan's rows, with the row its first bit
/// stands for. A chunk none of whose rows were kept is not planned.
pub(super) fn plan_selection(
    scan: &Arc<PreparedScan>,
    scope: WorkScope,
    selection: &Selection,
    kept: Option<(&Mask, u64)>,
    lease: Arc<SegmentLease>,
) -> Option<(WorkScope, Box<dyn Planner>)> {
    let mut selected = select_rows(scope, selection)?;
    if let Some((kept, first_row)) = kept {
        let offset = usize::try_from(selected.scope.rows.start - first_row)
            .vortex_expect("a work chunk fits usize");
        let kept = kept.slice(offset..offset + selected.mask.len());
        let mask = &selected.mask & &kept;
        let first = mask.first()?;
        let end = mask.last()? + 1;
        selected.scope.rows =
            selected.scope.rows.start + first as u64..selected.scope.rows.start + end as u64;
        selected.mask = mask.slice(first..end);
    }
    selected.lease = Some(lease);
    let scope = selected.scope.clone();
    Some((scope, scan.split_planner(selected)))
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
    let first = mask.first()?;
    let end = mask.last()? + 1;
    scope.rows = scope.rows.start + first as u64..scope.rows.start + end as u64;
    Some(SelectedRows {
        scope,
        mask: mask.slice(first..end),
        lease: None,
    })
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
