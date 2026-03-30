// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::future;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::stream::FuturesOrdered;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::Expression;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::LayoutReaderRef;
use crate::LazyReaderChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::reader::LayoutReader;
use crate::segments::SegmentSource;

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
        session: &VortexSession,
    ) -> Self {
        let nchildren = layout.nchildren();

        let mut chunk_offsets = vec![0; nchildren + 1];
        for i in 1..nchildren {
            chunk_offsets[i] = chunk_offsets[i - 1] + layout.children.child_row_count(i - 1);
        }
        chunk_offsets[nchildren] = layout.row_count();

        let dtypes = vec![layout.dtype.clone(); nchildren];
        let names = (0..nchildren)
            .map(|idx| Arc::from(format!("{name}.[{idx}]")))
            .collect();
        let lazy_children = LazyReaderChildren::new(
            layout.children.clone(),
            dtypes,
            names,
            segment_source,
            session.clone(),
        );

        Self {
            layout,
            name,
            lazy_children,
            chunk_offsets,
        }
    }

    /// Return the [`LayoutReader`] for the given chunk.
    fn chunk_reader(&self, idx: usize) -> VortexResult<&LayoutReaderRef> {
        self.lazy_children.get(idx)
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

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        if row_range.is_empty() {
            return Ok(());
        }

        for (index, (&start, &end)) in self
            .chunk_offsets
            .iter()
            .tuple_windows::<(_, _)>()
            .enumerate()
        {
            if end < row_range.start {
                continue;
            }

            if start >= row_range.end {
                break;
            }

            // Child overlaps in whole or in part with split
            let child = self.chunk_reader(index)?;
            let child_range =
                std::cmp::max(row_range.start, start)..std::cmp::min(row_range.end, end);

            // Register any splits from the child
            child.register_splits(field_mask, &child_range, splits)?;

            // Register the split indicating the end of this chunk
            splits.insert(child_range.end);
        }

        Ok(())
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        if row_range.is_empty() {
            return Ok(MaskFuture::ready(mask));
        }

        let mut chunk_evals = vec![];

        for (chunk_idx, chunk_range, mask_range) in self.ranges(row_range) {
            let chunk_reader = self.chunk_reader(chunk_idx)?;
            let chunk_eval = chunk_reader
                .pruning_evaluation(&chunk_range, expr, mask.slice(mask_range))
                .map_err(|err| {
                    err.with_context(format!(
                        "While evaluating pruning filter on chunk {chunk_idx}"
                    ))
                })?;

            chunk_evals.push(chunk_eval);
        }

        let name = self.name.clone();
        Ok(MaskFuture::new(mask.len(), async move {
            tracing::debug!(
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
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        if row_range.is_empty() {
            return Ok(mask);
        }

        let mut chunk_evals = vec![];

        for (chunk_idx, chunk_range, mask_range) in self.ranges(row_range) {
            let chunk_reader = self.chunk_reader(chunk_idx)?;
            let chunk_eval = chunk_reader
                .filter_evaluation(&chunk_range, expr, mask.slice(mask_range))
                .map_err(|err| {
                    err.with_context(format!("While evaluating filter on chunk {chunk_idx}"))
                })?;
            chunk_evals.push(chunk_eval);
        }

        let name = self.name.clone();
        Ok(MaskFuture::new(mask.len(), async move {
            tracing::debug!("Chunked mask evaluation {}", name);

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
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let dtype = expr.return_dtype(self.dtype())?;
        if row_range.is_empty() {
            return Ok(future::ready(Ok(Canonical::empty(&dtype).into_array())).boxed());
        }

        let mut chunk_evals = vec![];

        for (chunk_idx, chunk_range, mask_range) in self.ranges(row_range) {
            let chunk_reader = self.chunk_reader(chunk_idx)?;
            let chunk_eval = chunk_reader
                .projection_evaluation(&chunk_range, expr, mask.slice(mask_range))
                .map_err(|err| {
                    err.with_context(format!("While evaluating projection on chunk {chunk_idx}"))
                })?;
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
            Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
        }
        .boxed())
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use futures::stream;
    use rstest::fixture;
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::expr::root;
    use vortex_buffer::buffer;
    use vortex_io::runtime::single::block_on;

    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialStreamAdapter;
    use crate::sequence::SequentialStreamExt as _;
    use crate::test::SESSION;

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
                .new_reader("".into(), segments, &SESSION)
                .unwrap()
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &root(),
                    MaskFuture::new_true(usize::try_from(layout.row_count()).unwrap()),
                )
                .unwrap()
                .await
                .unwrap();

            let expected = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
            assert_arrays_eq!(result, expected);
        })
    }
}
