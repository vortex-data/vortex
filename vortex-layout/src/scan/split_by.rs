// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::once;
use std::ops::Range;

use vortex_array::dtype::FieldMask;
use vortex_error::VortexResult;

use crate::LayoutReader;
use crate::RowSplits;
use crate::SplitRange;

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
    ) -> VortexResult<Vec<u64>> {
        Ok(match *self {
            SplitBy::Layout => {
                // We usually have under 100 splits so reserving upfront saves
                // us some allocations
                let mut row_splits = RowSplits::new_capacity(128);
                row_splits.push(row_range.start);
                layout_reader.register_splits(
                    field_mask,
                    &SplitRange::root(row_range.clone())?,
                    &mut row_splits,
                )?;
                row_splits.into_sorted_deduped()
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
    use std::any::Any;
    use std::sync::Arc;

    use futures::future::BoxFuture;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldPath;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::BoundExpr;
    use vortex_buffer::buffer;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_mask::Mask;

    use super::*;
    use crate::LayoutReaderRef;
    use crate::LayoutStrategy;
    use crate::RowSplits;
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
            .new_reader("".into(), segments, &SCAN_SESSION, &Default::default())
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
        assert_eq!(splits, vec![0u64, 10]);
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
        assert_eq!(splits, vec![0u64, 3, 6, 9, 10]);
    }

    #[test]
    fn test_layout_splits_dedup() {
        struct DupReader {
            name: Arc<str>,
            dtype: DType,
        }

        impl LayoutReader for DupReader {
            fn name(&self) -> &Arc<str> {
                &self.name
            }

            fn as_any(&self) -> &dyn Any {
                self
            }

            fn dtype(&self) -> &DType {
                &self.dtype
            }

            fn row_count(&self) -> u64 {
                10
            }

            fn register_splits(
                &self,
                _field_mask: &[FieldMask],
                split_range: &SplitRange,
                splits: &mut RowSplits,
            ) -> VortexResult<()> {
                splits.push(split_range.row_offset() + 5);
                splits.push(split_range.row_offset() + 5);
                splits.push(split_range.root_row_range().end);
                Ok(())
            }

            fn pruning_evaluation(
                &self,
                _: &Range<u64>,
                _: &BoundExpr,
                _: Mask,
            ) -> VortexResult<MaskFuture> {
                unimplemented!()
            }

            fn filter_evaluation(
                &self,
                _: &Range<u64>,
                _: &BoundExpr,
                _: MaskFuture,
            ) -> VortexResult<MaskFuture> {
                unimplemented!()
            }

            fn projection_evaluation(
                &self,
                _: &Range<u64>,
                _: &BoundExpr,
                _: MaskFuture,
            ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
                unimplemented!()
            }
        }

        let reader = DupReader {
            name: Arc::from("dup"),
            dtype: DType::Primitive(PType::U8, Nullability::NonNullable),
        };
        let splits = SplitBy::Layout
            .splits(&reader, &(0..10), &[FieldMask::All])
            .unwrap();
        assert_eq!(splits, vec![0u64, 5, 10]);
    }
}
