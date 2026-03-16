// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;

use vortex_array::dtype::FieldMask;
use vortex_error::VortexResult;
use vortex_layout::LayoutReader;
use vortex_layout::SplitPointIter;

/// Defines how the Vortex file is split into batches for reading.
///
/// Note that each split must fit into the platform's maximum usize.
#[derive(Default, Copy, Clone, Debug)]
pub enum SplitBy {
    #[default]
    /// Splits any time there is a chunk boundary in the file.
    Layout,
    /// Splits every n rows.
    RowCount(usize),
    // UncompressedSize(u64),
}

impl SplitBy {
    pub fn split_points(
        &self,
        layout_reader: &dyn LayoutReader,
        row_range: Range<u64>,
        field_mask: Vec<FieldMask>,
    ) -> VortexResult<SplitPointIter> {
        if row_range.is_empty() {
            return Ok(Box::new(std::iter::empty()));
        }

        Ok(match *self {
            SplitBy::Layout => layout_reader.split_points(field_mask, row_range)?,
            SplitBy::RowCount(n) => {
                let mut points: Vec<u64> = row_range.clone().step_by(n).skip(1).collect();
                if points.last().copied() != Some(row_range.end) {
                    points.push(row_range.end);
                }
                Box::new(points.into_iter())
            }
        })
    }

    /// Compute the splits for the given layout.
    pub fn splits(
        &self,
        layout_reader: &dyn LayoutReader,
        row_range: &Range<u64>,
        field_mask: &[FieldMask],
    ) -> VortexResult<BTreeSet<u64>> {
        let mut row_splits = BTreeSet::<u64>::new();
        row_splits.insert(row_range.start);
        row_splits.extend(self.split_points(
            layout_reader,
            row_range.clone(),
            field_mask.to_vec(),
        )?);
        Ok(row_splits)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::dtype::FieldPath;
    use vortex_buffer::buffer;
    use vortex_io::runtime::single::block_on;
    use vortex_layout::LayoutReaderRef;
    use vortex_layout::LayoutStrategy;
    use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
    use vortex_layout::segments::TestSegments;
    use vortex_layout::sequence::SequenceId;
    use vortex_layout::sequence::SequentialArrayStreamExt;

    use super::*;
    use crate::test::SCAN_SESSION;

    fn reader() -> LayoutReaderRef {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let layout = block_on(|handle| async {
            FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    segments.clone(),
                    buffer![1_i32; 10]
                        .into_array()
                        .to_array_stream()
                        .sequenced(ptr),
                    eof,
                    handle,
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

        let splits = SplitBy::Layout
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
}
