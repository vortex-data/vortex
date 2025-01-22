use std::collections::BTreeSet;
use std::ops::Range;

use itertools::Itertools;
use vortex_dtype::FieldMask;
use vortex_error::VortexResult;
use vortex_layout::LayoutData;

/// Defines how the Vortex file is split into batches for reading.
///
/// Note that each split must fit into the platform's maximum usize.
#[derive(Copy, Clone)]
pub enum SplitBy {
    /// Splits any time there is a chunk boundary in the file.
    Layout,
    /// Splits every n rows.
    RowCount(usize),
    // UncompressedSize(u64),
}

impl SplitBy {
    /// Compute the splits for the given layout.
    pub(crate) fn splits(
        &self,
        layout: &LayoutData,
        field_mask: &[FieldMask],
    ) -> VortexResult<Vec<Range<u64>>> {
        Ok(match *self {
            SplitBy::Layout => {
                let mut row_splits = BTreeSet::<u64>::new();
                // Make sure we always have the first and last row.
                row_splits.insert(0);
                row_splits.insert(layout.row_count());
                // Register the splits for all the layouts.
                layout.register_splits(field_mask, 0, &mut row_splits)?;

                row_splits
                    .into_iter()
                    .tuple_windows()
                    .map(|(start, end)| start..end)
                    .collect()
            }
            SplitBy::RowCount(n) => {
                let row_count = layout.row_count();
                let mut splits =
                    Vec::with_capacity(usize::try_from((row_count + n as u64) / n as u64)?);
                for start in (0..row_count).step_by(n) {
                    let end = (start + n as u64).min(row_count);
                    splits.push(start..end);
                }
                splits
            }
        })
    }
}

#[cfg(test)]
mod test {
    use vortex_array::IntoArrayData;
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, FieldPath};
    use vortex_layout::layouts::flat::writer::FlatLayoutWriter;
    use vortex_layout::strategies::LayoutWriterExt;

    use super::*;
    use crate::segments::writer::BufferedSegmentWriter;

    #[test]
    fn test_layout_splits_flat() {
        let mut segments = BufferedSegmentWriter::default();
        let layout = FlatLayoutWriter::new(DType::Bool(NonNullable), Default::default())
            .push_one(&mut segments, buffer![1; 10].into_array())
            .unwrap();
        let splits = SplitBy::Layout
            .splits(&layout, &[FieldMask::Exact(FieldPath::root())])
            .unwrap();
        assert_eq!(splits, vec![0..10]);
    }

    #[test]
    fn test_row_count_splits() {
        let mut segments = BufferedSegmentWriter::default();
        let layout = FlatLayoutWriter::new(DType::Bool(NonNullable), Default::default())
            .push_one(&mut segments, buffer![1; 10].into_array())
            .unwrap();
        let splits = SplitBy::RowCount(3)
            .splits(&layout, &[FieldMask::Exact(FieldPath::root())])
            .unwrap();
        assert_eq!(splits, vec![0..3, 3..6, 6..9, 9..10]);
    }
}
