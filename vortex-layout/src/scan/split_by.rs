// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use futures::executor::block_on;
    use futures::{TryStreamExt, stream};
    use itertools::Itertools;
    use vortex_array::{ArrayContext, IntoArray};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, FieldMask, FieldPath, PType};
    use vortex_mask::Mask;

    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::scan::tree_row_mask::TreeRowMask;
    use crate::segments::{SegmentSource, SequenceWriter, TestSegments};
    use crate::sequence::SequenceId;
    use crate::{LayoutStrategy, SequentialStreamAdapter, SequentialStreamExt as _};

    #[tokio::test]
    async fn test_layout_splits_flat() {
        let segments = TestSegments::default();
        let layout = block_on(
            FlatLayoutStrategy::default().write_stream(
                &ArrayContext::empty(),
                SequenceWriter::new(Box::new(segments.clone())),
                SequentialStreamAdapter::new(
                    DType::Primitive(PType::I32, NonNullable),
                    stream::once(async {
                        Ok((
                            SequenceId::root().downgrade(),
                            buffer![1_i32; 10].into_array(),
                        ))
                    }),
                )
                .sendable(),
            ),
        )
        .unwrap();

        let segments: Arc<dyn SegmentSource> = Arc::new(segments);
        let reader = layout.new_reader("".into(), segments).unwrap();
        let splits = reader
            .row_masks(
                &TreeRowMask::all(0..=10),
                &[FieldMask::Exact(FieldPath::root())],
            )
            .try_collect::<Vec<Mask>>()
            .await
            .unwrap();

        assert_eq!(splits, vec![Mask::from_indices(10, (0..10).collect_vec())]);
    }
}
