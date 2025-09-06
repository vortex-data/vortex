// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;

use itertools::Itertools;
use vortex_array::stats::StatBound;
use vortex_dtype::FieldMask;
use vortex_error::{vortex_err, VortexResult};

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
        field_mask: &[FieldMask],
    ) -> VortexResult<Vec<Range<u64>>> {
        Ok(match *self {
            SplitBy::Layout => {
                let mut row_splits = BTreeSet::<u64>::new();
                row_splits.insert(0);

                // Register the splits for all the layouts.
                layout_reader.register_splits(field_mask, 0, &mut row_splits)?;

                row_splits
                    .into_iter()
                    .tuple_windows()
                    .map(|(start, end)| start..end)
                    .collect()
            }
            SplitBy::RowCount(n) => {
                let row_count = *layout_reader.row_count().to_exact().ok_or_else(|| {
                    vortex_err!("Cannot split layout by row count, row count is not exact")
                })?;
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
    use std::sync::Arc;

    use vortex_array::{ArrayContext, IntoArray};
    use vortex_buffer::buffer;
    use vortex_dtype::FieldPath;
    use vortex_io::runtime::single::SingleThreadRuntime;
    use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
    use vortex_layout::segments::{SegmentSource, TestSegments};
    use vortex_layout::sequence::{SequenceId, SequentialArrayStreamExt};
    use vortex_layout::LayoutStrategy;

    use super::*;

    #[test]
    fn test_layout_splits_flat() {
        let ctx = ArrayContext::empty();
        let segments = TestSegments::default();
        let (ptr, eof) = SequenceId::root().split();
        let layout = SingleThreadRuntime::block_on(|handle| async {
            FlatLayoutStrategy::default()
                .write_stream(
                    &ctx,
                    &segments,
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

        let segments: Arc<dyn SegmentSource> = Arc::new(segments);
        let reader = layout.new_reader("".into(), segments).unwrap();

        let splits = SplitBy::Layout
            .splits(reader.as_ref(), &[FieldMask::Exact(FieldPath::root())])
            .unwrap();
        assert_eq!(splits, vec![0..10]);
    }

    #[test]
    fn test_row_count_splits() {
        let ctx = ArrayContext::empty();
        let segments = TestSegments::default();
        let (ptr, eof) = SequenceId::root().split();
        let layout = SingleThreadRuntime::block_on(|handle| async {
            FlatLayoutStrategy::default()
                .write_stream(
                    &ctx,
                    &segments,
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

        let segments: Arc<dyn SegmentSource> = Arc::new(segments);
        let reader = layout.new_reader("".into(), segments).unwrap();

        let splits = SplitBy::RowCount(3)
            .splits(reader.as_ref(), &[FieldMask::Exact(FieldPath::root())])
            .unwrap();
        assert_eq!(splits, vec![0..3, 3..6, 6..9, 9..10]);
    }
}
