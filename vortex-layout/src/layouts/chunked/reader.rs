// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::stream::FuturesOrdered;
use futures::{FutureExt, TryStreamExt};
use vortex_array::arrays::ChunkedArray;
use vortex_array::stats::Precision;
use vortex_array::{ArrayRef, MaskFuture};
use vortex_dtype::{DType, FieldMask};
use vortex_error::{VortexExpect, VortexResult, vortex_panic};
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::layouts::chunked::ChunkedLayout;
use crate::reader::LayoutReader;
use crate::segments::SegmentSource;
use crate::{LayoutReaderRef, LazyReaderChildren};

/// A [`LayoutReader`] for chunked layouts.
pub struct ChunkedReader {
    layout: ChunkedLayout,
    name: Arc<str>,
    lazy_children: LazyReaderChildren,
    /// Row offset for each chunk
    chunk_offsets: Vec<u64>,
}

impl ChunkedReader {
    pub fn new(
        layout: ChunkedLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
    ) -> Self {
        let nchildren = layout.nchildren();

        let mut chunk_offsets = vec![0; nchildren + 1];
        for i in 1..nchildren {
            chunk_offsets[i] = chunk_offsets[i - 1] + layout.children.child_row_count(i - 1);
        }
        chunk_offsets[nchildren] = layout.row_count();

        let lazy_children = LazyReaderChildren::new(layout.children.clone(), segment_source);

        Self {
            layout,
            name,
            lazy_children,
            chunk_offsets,
        }
    }

    /// Return the [`LayoutReader`] for the given chunk.
    fn chunk_reader(&self, idx: usize) -> VortexResult<&LayoutReaderRef> {
        self.lazy_children.get(
            idx,
            self.layout.dtype(),
            &format!("{}.[{}]", self.name, idx).into(),
        )
    }

    fn chunk_offset(&self, idx: usize) -> u64 {
        self.chunk_offsets.get(idx).copied().unwrap_or_else(|| {
            vortex_panic!(
                "Internal error: Chunk offset {idx} out of bounds (num_children: {}, num_offsets: {}). \
                This indicates a bug in ChunkedReader initialization or chunk_range calculation.",
                self.layout.nchildren(),
                self.chunk_offsets.len()
            )
        })
    }

    fn chunk_range(&self, row_range: &Range<u64>) -> Range<usize> {
        let start_chunk = self
            .chunk_offsets
            .binary_search(&row_range.start)
            .unwrap_or_else(|x| x.saturating_sub(1));
        let end_chunk = self
            .chunk_offsets
            .binary_search(&row_range.end)
            .unwrap_or_else(|x| x);
        start_chunk..end_chunk
    }

    fn ranges<'a>(
        &'a self,
        row_range: &'a Range<u64>,
    ) -> impl Iterator<Item = (usize, Range<u64>, Range<usize>)> + 'a {
        self.chunk_range(row_range).map(move |chunk_idx| {
            // Figure out the chunk row range relative to the mask's row range.
            let chunk_row_range = self.chunk_offset(chunk_idx)..self.chunk_offset(chunk_idx + 1);

            // Find the intersection of the mask and the chunk row ranges.
            let intersecting_row_range =
                row_range.start.max(chunk_row_range.start)..row_range.end.min(chunk_row_range.end);
            let intersecting_len = usize::try_from(
                intersecting_row_range
                    .end
                    .checked_sub(intersecting_row_range.start)
                    .vortex_expect("Invalid row range"),
            )
            .vortex_expect("Row range length exceeds usize::MAX");

            // Figure out the offset into the mask.
            let mask_relative_start = usize::try_from(
                intersecting_row_range
                    .start
                    .checked_sub(row_range.start)
                    .vortex_expect("Invalid row range"),
            )
            .vortex_expect("Mask offset exceeds usize::MAX");
            let mask_relative_end = mask_relative_start
                .checked_add(intersecting_len)
                .vortex_expect("Mask range calculation overflow");
            let mask_range = mask_relative_start..mask_relative_end;

            // Figure out the row range within the chunk.
            let chunk_relative_start = intersecting_row_range
                .start
                .checked_sub(chunk_row_range.start)
                .vortex_expect("Chunk range calculation underflow");
            let chunk_relative_end = chunk_relative_start
                .checked_add(intersecting_len as u64)
                .vortex_expect("Chunk range calculation overflow");
            let chunk_range = chunk_relative_start..chunk_relative_end;

            (chunk_idx, chunk_range, mask_range)
        })
    }
}

impl LayoutReader for ChunkedReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::Exact(self.layout.row_count())
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        let mut offset = row_offset;
        for i in 0..self.layout.nchildren() {
            let child = self.chunk_reader(i)?;
            child.register_splits(field_mask, offset, splits)?;
            offset += self.layout.child(i)?.row_count();
            splits.insert(offset);
        }
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        let mut chunk_evals = vec![];

        for (chunk_idx, chunk_range, mask_range) in self.ranges(row_range) {
            let chunk_reader = self.chunk_reader(chunk_idx)?;
            let chunk_eval =
                chunk_reader.pruning_evaluation(&chunk_range, expr, mask.slice(mask_range))?;

            chunk_evals.push(chunk_eval);
        }

        let name = self.name.clone();
        Ok(MaskFuture::new(mask.len(), async move {
            log::debug!(
                "Chunked pruning evaluation {} (mask = {})",
                name,
                mask.density()
            );

            // Split the mask over each chunk.
            let masks: Vec<_> = FuturesOrdered::from_iter(chunk_evals).try_collect().await?;

            // If there is only one mask, we can return it directly.
            if masks.len() == 1 {
                return Ok(masks.into_iter().next().vortex_expect("one mask"));
            }

            // Combine the masks.
            Ok(Mask::from_iter(masks))
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let mut chunk_evals = vec![];

        for (chunk_idx, chunk_range, mask_range) in self.ranges(row_range) {
            let chunk_reader = self.chunk_reader(chunk_idx)?;
            let chunk_eval =
                chunk_reader.filter_evaluation(&chunk_range, expr, mask.slice(mask_range))?;
            chunk_evals.push(chunk_eval);
        }

        let name = self.name.clone();
        Ok(MaskFuture::new(mask.len(), async move {
            log::debug!("Chunked mask evaluation {}", name);

            // Split the mask over each chunk.
            let masks: Vec<_> = FuturesOrdered::from_iter(chunk_evals).try_collect().await?;

            // If there is only one mask, we can return it directly.
            if masks.len() == 1 {
                return Ok(masks.into_iter().next().vortex_expect("one mask"));
            }

            // Combine the masks.
            Ok(Mask::from_iter(masks))
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let dtype = expr.return_dtype(self.dtype())?;
        let mut chunk_evals = vec![];

        for (chunk_idx, chunk_range, mask_range) in self.ranges(row_range) {
            let chunk_reader = self.chunk_reader(chunk_idx)?;
            let chunk_eval =
                chunk_reader.projection_evaluation(&chunk_range, expr, mask.slice(mask_range))?;
            chunk_evals.push(chunk_eval);
        }

        Ok(async move {
            // Split the mask over each chunk.
            let chunks: Vec<_> = FuturesOrdered::from_iter(chunk_evals).try_collect().await?;

            // If there is only one chunk, we can return it directly.
            if chunks.len() == 1 {
                return Ok(chunks.into_iter().next().vortex_expect("one chunk"));
            }

            // Combine the arrays.
            Ok(ChunkedArray::try_new(chunks, dtype)?.to_array())
        }
        .boxed())
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use futures::stream;
    use rstest::{fixture, rstest};
    use vortex_array::{ArrayContext, IntoArray, MaskFuture, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, PType};
    use vortex_expr::root;
    use vortex_io::runtime::single::block_on;

    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::{SegmentSource, TestSegments};
    use crate::sequence::{SequenceId, SequentialStreamAdapter, SequentialStreamExt as _};
    use crate::{LayoutRef, LayoutStrategy};

    #[fixture]
    /// Create a chunked layout with three chunks of primitive arrays.
    fn chunked_layout() -> (Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let strategy = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default());
        let (mut sequence_id, eof) = SequenceId::root().split();
        let layout = block_on(|handle| {
            strategy.write_stream(
                ctx,
                segments.clone(),
                SequentialStreamAdapter::new(
                    DType::Primitive(PType::I32, NonNullable),
                    stream::iter([
                        Ok((sequence_id.advance(), buffer![1, 2, 3].into_array())),
                        Ok((sequence_id.advance(), buffer![4, 5, 6].into_array())),
                        Ok((sequence_id.advance(), buffer![7, 8, 9].into_array())),
                    ]),
                )
                .sendable(),
                eof,
                handle,
            )
        })
        .unwrap();

        (segments, layout)
    }

    #[rstest]
    fn test_chunked_evaluator(
        #[from(chunked_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        block_on(|_h| async {
            let result = layout
                .new_reader("".into(), segments)
                .unwrap()
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &root(),
                    MaskFuture::new_true(usize::try_from(layout.row_count()).unwrap()),
                )
                .unwrap()
                .await
                .unwrap()
                .to_primitive();

            assert_eq!(result.len(), 9);
            assert_eq!(result.as_slice::<i32>(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        })
    }
}
