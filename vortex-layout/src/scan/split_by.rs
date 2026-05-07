// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::iter::once;
use std::ops::Range;

use vortex_array::dtype::FieldMask;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::LayoutReader;

/// Defines how the Vortex file is split into batches for reading.
///
/// Note that each split must fit into the platform's maximum usize.
#[derive(Copy, Clone, Debug)]
pub enum SplitBy {
    /// Splits any time there is a chunk boundary in the file.
    Layout {
        /// Coalesce adjacent layout splits until each split contains at least this many rows,
        /// where possible.
        min_rows: Option<usize>,
        /// Add extra row-count split points so each split contains at most this many rows.
        max_rows: Option<usize>,
    },
    /// Splits every n rows.
    RowCount(usize),
    // UncompressedSize(u64),
}

impl Default for SplitBy {
    fn default() -> Self {
        Self::layout()
    }
}

impl SplitBy {
    /// Splits any time there is a chunk boundary in the file.
    pub const fn layout() -> Self {
        Self::Layout {
            min_rows: None,
            max_rows: None,
        }
    }

    /// Splits on layout boundaries, coalescing or subdividing by row count when requested.
    pub const fn layout_with_row_limits(min_rows: Option<usize>, max_rows: Option<usize>) -> Self {
        Self::Layout { min_rows, max_rows }
    }

    /// Compute the splits for the given layout.
    // TODO(ngates): remove this once layout readers are stream based.
    pub fn splits(
        &self,
        layout_reader: &dyn LayoutReader,
        row_range: &Range<u64>,
        field_mask: &[FieldMask],
    ) -> VortexResult<BTreeSet<u64>> {
        Ok(match *self {
            SplitBy::Layout { min_rows, max_rows } => {
                validate_row_limit("min_rows", min_rows)?;
                validate_row_limit("max_rows", max_rows)?;
                if let (Some(min_rows), Some(max_rows)) = (min_rows, max_rows)
                    && min_rows > max_rows
                {
                    vortex_bail!(
                        "layout split min_rows ({min_rows}) cannot be greater than max_rows ({max_rows})"
                    );
                }

                let mut row_splits = BTreeSet::<u64>::new();
                row_splits.insert(row_range.start);
                row_splits.insert(row_range.end);

                // Register all splits in the row range for all layouts that are needed
                // to read the field mask.
                layout_reader.register_splits(field_mask, row_range, &mut row_splits)?;
                apply_layout_row_limits(row_splits, min_rows, max_rows)
            }
            SplitBy::RowCount(n) => {
                validate_row_limit("row count", Some(n))?;
                row_range
                    .clone()
                    .step_by(n)
                    .chain(once(row_range.end))
                    .collect()
            }
        })
    }
}

fn validate_row_limit(name: &str, rows: Option<usize>) -> VortexResult<()> {
    if rows == Some(0) {
        vortex_bail!("{name} split size must be greater than zero");
    }
    Ok(())
}

fn apply_layout_row_limits(
    mut row_splits: BTreeSet<u64>,
    min_rows: Option<usize>,
    max_rows: Option<usize>,
) -> BTreeSet<u64> {
    if let Some(min_rows) = min_rows {
        row_splits = coalesce_min_row_limit(&row_splits, min_rows as u64);
    }
    if let Some(max_rows) = max_rows {
        row_splits = split_max_row_limit(&row_splits, max_rows as u64);
    }
    row_splits
}

fn coalesce_min_row_limit(row_splits: &BTreeSet<u64>, min_rows: u64) -> BTreeSet<u64> {
    let boundaries = row_splits.iter().copied().collect::<Vec<_>>();
    if boundaries.len() <= 2 {
        return row_splits.clone();
    }

    let start = boundaries[0];
    let end = boundaries[boundaries.len() - 1];
    let mut coalesced = Vec::with_capacity(boundaries.len());
    coalesced.push(start);

    let mut chunk_start = start;
    for boundary in boundaries.into_iter().skip(1) {
        if boundary - chunk_start >= min_rows {
            coalesced.push(boundary);
            chunk_start = boundary;
        }
    }

    if coalesced[coalesced.len() - 1] != end {
        coalesced.push(end);
    }

    let last_idx = coalesced.len() - 1;
    if last_idx >= 2 && end - coalesced[last_idx - 1] < min_rows {
        coalesced.remove(last_idx - 1);
    }

    coalesced.into_iter().collect()
}

fn split_max_row_limit(row_splits: &BTreeSet<u64>, max_rows: u64) -> BTreeSet<u64> {
    let boundaries = row_splits.iter().copied().collect::<Vec<_>>();
    if boundaries.len() <= 1 {
        return row_splits.clone();
    }

    let mut split = BTreeSet::new();
    split.insert(boundaries[0]);

    for window in boundaries.windows(2) {
        let start = window[0];
        let end = window[1];
        let row_count = end - start;
        if row_count > max_rows {
            let parts = row_count.div_ceil(max_rows);
            let base_rows = row_count / parts;
            let remainder = row_count % parts;
            let mut offset = 0;
            for part_idx in 0..parts - 1 {
                offset += base_rows + u64::from(part_idx < remainder);
                split.insert(start + offset);
            }
        }
        split.insert(end);
    }

    split
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::dtype::FieldPath;
    use vortex_buffer::buffer;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;

    use super::*;
    use crate::LayoutReaderRef;
    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::scan::test::SCAN_SESSION;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;

    fn reader() -> LayoutReaderRef {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let layout = block_on(|handle| async {
            let session = SCAN_SESSION.clone().with_handle(handle);
            FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    Arc::<TestSegments>::clone(&segments),
                    buffer![1_i32; 10]
                        .into_array()
                        .to_array_stream()
                        .sequenced(ptr),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        layout
            .new_reader("".into(), segments, &SCAN_SESSION)
            .unwrap()
    }

    #[test]
    fn test_layout_splits_flat() {
        let reader = reader();

        let splits = SplitBy::layout()
            .splits(
                reader.as_ref(),
                &(0..10),
                &[FieldMask::Exact(FieldPath::root())],
            )
            .unwrap();
        assert_eq!(splits, [0, 10].into_iter().collect());
    }

    #[test]
    fn test_row_count_splits() {
        let reader = reader();

        let splits = SplitBy::RowCount(3)
            .splits(
                reader.as_ref(),
                &(0..10),
                &[FieldMask::Exact(FieldPath::root())],
            )
            .unwrap();
        assert_eq!(splits, [0, 3, 6, 9, 10].into_iter().collect());
    }

    #[test]
    fn test_layout_splits_with_max_rows() {
        let reader = reader();

        let splits = SplitBy::Layout {
            min_rows: None,
            max_rows: Some(3),
        }
        .splits(
            reader.as_ref(),
            &(0..10),
            &[FieldMask::Exact(FieldPath::root())],
        )
        .unwrap();
        assert_eq!(splits, [0, 3, 6, 8, 10].into_iter().collect());
    }

    #[test]
    fn test_layout_splits_with_min_rows() {
        let splits =
            apply_layout_row_limits([0, 10, 20, 30, 100].into_iter().collect(), Some(25), None);
        assert_eq!(splits, [0, 30, 100].into_iter().collect());
    }

    #[test]
    fn test_layout_splits_with_min_rows_merges_final_tail() {
        let splits = apply_layout_row_limits([0, 30, 40].into_iter().collect(), Some(25), None);
        assert_eq!(splits, [0, 40].into_iter().collect());
    }

    #[test]
    fn test_layout_splits_with_min_and_max_rows() {
        let splits = apply_layout_row_limits(
            [0, 10, 20, 30, 100].into_iter().collect(),
            Some(25),
            Some(40),
        );
        assert_eq!(splits, [0, 30, 65, 100].into_iter().collect());
    }

    #[test]
    fn test_invalid_split_row_limits() {
        let reader = reader();

        assert!(
            SplitBy::layout_with_row_limits(Some(0), None)
                .splits(
                    reader.as_ref(),
                    &(0..10),
                    &[FieldMask::Exact(FieldPath::root())],
                )
                .is_err()
        );
        assert!(
            SplitBy::layout_with_row_limits(Some(10), Some(5))
                .splits(
                    reader.as_ref(),
                    &(0..10),
                    &[FieldMask::Exact(FieldPath::root())],
                )
                .is_err()
        );
        assert!(
            SplitBy::RowCount(0)
                .splits(
                    reader.as_ref(),
                    &(0..10),
                    &[FieldMask::Exact(FieldPath::root())],
                )
                .is_err()
        );
    }
}
