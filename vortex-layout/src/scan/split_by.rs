// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::iter::once;
use std::ops::Range;

use vortex_array::dtype::FieldMask;
use vortex_error::VortexResult;

use crate::LayoutReader;

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
    /// Compute the splits for the given layout.
    // TODO(ngates): remove this once layout readers are stream based.
    pub fn splits(
        &self,
        layout_reader: &dyn LayoutReader,
        row_range: &Range<u64>,
        field_mask: &[FieldMask],
    ) -> VortexResult<BTreeSet<u64>> {
        Ok(match *self {
            SplitBy::Layout => {
                let mut row_splits = BTreeSet::<u64>::new();
                row_splits.insert(row_range.start);

                // Register all splits in the row range for all layouts that are needed
                // to read the field mask.
                layout_reader.register_splits(field_mask, row_range, &mut row_splits)?;
                row_splits
            }
            SplitBy::RowCount(n) => row_range
                .clone()
                .step_by(n)
                .chain(once(row_range.end))
                .collect(),
        })
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
            FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    Arc::<TestSegments>::clone(&segments),
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
