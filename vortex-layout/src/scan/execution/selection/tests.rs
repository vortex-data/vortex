// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::selection::Selection;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;

use super::SelectionChunks;
use super::select_rows;

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
    let selection = Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::from(indices))?);
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

#[rstest]
#[case::whole(0..150_000, vec![0..65_536, 65_536..131_072, 131_072..150_000])]
#[case::unaligned(10..140_000, vec![10..65_536, 65_536..131_072, 131_072..140_000])]
#[case::boundary(65_536..131_072, vec![65_536..131_072])]
#[case::empty(5..5, vec![])]
#[case::near_max(u64::MAX - 3..u64::MAX, vec![u64::MAX - 3..u64::MAX])]
fn fixed_row_chunks(#[case] rows: Range<u64>, #[case] expected: Vec<Range<u64>>) {
    let actual: Vec<_> = SelectionChunks {
        rows,
        chunk_rows: 65_536,
        selection: Selection::All,
    }
    .collect();
    assert_eq!(actual, expected);
}

#[test]
fn first_chunk_does_not_enumerate_the_scan() {
    let mut chunks = SelectionChunks {
        rows: 0..u64::MAX,
        chunk_rows: 8192,
        selection: Selection::All,
    };
    assert_eq!(chunks.next(), Some(0..8192));
    assert_eq!(chunks.next(), Some(8192..16_384));
}

#[rstest]
fn sparse_indices_jump_over_empty_chunks(#[values(false, true)] roaring: bool) -> VortexResult<()> {
    let indices = [7, 8, u64::MAX - 2, u64::MAX - 1];
    let selection = if roaring {
        Selection::IncludeRoaring(indices.into_iter().collect())
    } else {
        Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::from_iter(indices))?)
    };
    let actual: Vec<_> = SelectionChunks {
        rows: 0..u64::MAX,
        chunk_rows: 8192,
        selection,
    }
    .collect();
    assert_eq!(actual, [0..8192, (u64::MAX / 8192 * 8192)..u64::MAX]);
    Ok(())
}

#[rstest]
#[case::dense((0..200).collect(), 0..200)]
#[case::clustered(vec![0, 1, 2, 3, 7, 8, 15, 16, 17, 190, 199], 0..200)]
#[case::clipped(vec![0, 1, 4, 7, 8, 16, 17, 190, 199], 3..193)]
#[case::outside(vec![0, 1, 2, 199], 3..193)]
#[case::empty(vec![], 0..200)]
fn chunk_ownership_and_selection(
    #[case] indices: Vec<u64>,
    #[case] rows: Range<u64>,
    #[values(false, true)] roaring: bool,
    #[values(false, true)] exclude: bool,
) -> VortexResult<()> {
    let selection = match (roaring, exclude) {
        (false, false) => {
            Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::from(indices.clone()))?)
        }
        (false, true) => {
            Selection::ExcludeByIndex(StrictSortedBuffer::try_new(Buffer::from(indices.clone()))?)
        }
        (true, false) => Selection::IncludeRoaring(indices.iter().copied().collect()),
        (true, true) => Selection::ExcludeRoaring(indices.iter().copied().collect()),
    };
    let expected: Vec<_> = rows
        .clone()
        .filter(|row| indices.contains(row) != exclude)
        .collect();
    let mut actual = Vec::new();
    let mut owners = Vec::new();
    let mut chunks = SelectionChunks {
        rows,
        chunk_rows: 8,
        selection,
    };
    while let Some(chunk) = chunks.next() {
        owners.push(chunk.start / 8);
        if let Some(selected) = select_rows(
            WorkScope {
                file_ordinal: 0,
                rows: chunk,
            },
            &chunks.selection,
        ) {
            actual.extend(selected.scope.rows.clone().filter(|row| {
                selected
                    .mask
                    .value(usize::try_from(row - selected.scope.rows.start).unwrap())
            }));
        }
    }
    assert!(owners.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn smaller_chunks_keep_file_alignment() {
    let actual: Vec<_> = SelectionChunks {
        rows: 7500..33_000,
        chunk_rows: 16_384,
        selection: Selection::All,
    }
    .collect();
    assert_eq!(actual, [7500..16_384, 16_384..32_768, 32_768..33_000]);
}
