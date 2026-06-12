// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use futures::FutureExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::stream::FuturesOrdered;
use tracing::trace;
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
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LazyReaderChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::reader::LayoutReader;
use crate::reader::RowSplits;
use crate::reader::SplitRange;
use crate::segments::SegmentSource;

/// A [`LayoutReader`] for chunked layouts.
pub struct ChunkedReader {
    layout: ChunkedLayout,
    name: Arc<str>,
    lazy_children: LazyReaderChildren,
}

static UNKNOWN: LazyLock<Arc<str>> = LazyLock::new(|| Arc::from("chunked-child"));

impl ChunkedReader {
    pub fn new(
        layout: ChunkedLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: LayoutReaderContext,
    ) -> Self {
        let nchildren = layout.nchildren();
        let dtypes = vec![layout.dtype.clone(); nchildren];

        // format!() has non-marginal overhead for short queries like random
        // access benchmarks
        let names = if cfg!(debug_assertions) {
            (0..nchildren)
                .map(|idx| Arc::from(format!("{name}.[{idx}]")))
                .collect()
        } else {
            vec![Arc::clone(&*UNKNOWN); nchildren]
        };

        let lazy_children = LazyReaderChildren::new(
            Arc::clone(&layout.children),
            dtypes,
            names,
            segment_source,
            session.clone(),
            ctx,
        );

        Self {
            layout,
            name,
            lazy_children,
        }
    }

    /// Return the [`LayoutReader`] for the given chunk.
    fn chunk_reader(&self, idx: usize) -> VortexResult<&LayoutReaderRef> {
        self.lazy_children.get(idx)
    }

    fn chunk_offset(&self, idx: usize) -> u64 {
        if idx >= self.layout.chunk_offsets.len() {
            vortex_panic!(
                "Internal error: Chunk offset {idx} out of bounds (num_children: {}, num_offsets: {}). \
                This indicates a bug in ChunkedReader initialization or chunk_range calculation.",
                self.layout.nchildren(),
                self.layout.chunk_offsets.len()
            )
        }
        self.layout.chunk_offsets[idx]
    }

    fn chunk_range(&self, row_range: &Range<u64>) -> Range<usize> {
        let start_chunk = self
            .layout
            .chunk_offsets
            .binary_search(&row_range.start)
            .unwrap_or_else(|x| x.saturating_sub(1));
        let end_chunk = self
            .layout
            .chunk_offsets
            .binary_search(&row_range.end)
            .unwrap_or_else(|x| x);
        start_chunk..end_chunk
    }

    fn ranges<'a>(
        &'a self,
        row_range: &'a Range<u64>,
    ) -> impl Iterator<Item = (usize, u64, Range<u64>, Range<usize>)> + 'a {
        self.chunk_range(row_range).map(move |chunk_idx| {
            // Figure out the chunk row range relative to the mask's row range.
            let chunk_row_range = self.chunk_offset(chunk_idx)..self.chunk_offset(chunk_idx + 1);
            let chunk_start = chunk_row_range.start;

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

            (chunk_idx, chunk_start, chunk_range, mask_range)
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
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        split_range.check_bounds(self.layout.row_count())?;

        if split_range.is_empty() {
            return Ok(());
        }

        let iter = self.ranges(split_range.row_range());
        splits.reserve(iter.size_hint().0);

        for (chunk_idx, chunk_start, child_range, _) in iter {
            let child = self.chunk_reader(chunk_idx)?;
            let child_row_offset = split_range
                .row_offset()
                .checked_add(chunk_start)
                .vortex_expect("Chunked layout split offset overflow");
            let child_split_range = SplitRange::try_new(child_row_offset, child_range)?;

            child.register_splits(field_mask, &child_split_range, splits)?;

            // Register the split indicating the end of this chunk
            splits.push(
                split_range
                    .row_offset()
                    .checked_add(chunk_start + child_split_range.row_range().end)
                    .vortex_expect("Chunked layout split offset overflow"),
            );
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

        for (chunk_idx, _, chunk_range, mask_range) in self.ranges(row_range) {
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

        let name = Arc::clone(&self.name);
        Ok(MaskFuture::new(mask.len(), async move {
            trace!(
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

        for (chunk_idx, _, chunk_range, mask_range) in self.ranges(row_range) {
            let chunk_reader = self.chunk_reader(chunk_idx)?;
            let chunk_eval = chunk_reader
                .filter_evaluation(&chunk_range, expr, mask.slice(mask_range))
                .map_err(|err| {
                    err.with_context(format!("While evaluating filter on chunk {chunk_idx}"))
                })?;
            chunk_evals.push(chunk_eval);
        }

        let name = Arc::clone(&self.name);
        Ok(MaskFuture::new(mask.len(), async move {
            trace!("Chunked mask evaluation {}", name);

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
        if row_range.is_empty() {
            return Ok(future::ready(Ok(
                Canonical::empty(&expr.return_dtype(self.dtype())?).into_array()
            ))
            .boxed());
        }

        let mut chunk_evals = vec![];

        for (chunk_idx, _, chunk_range, mask_range) in self.ranges(row_range) {
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

            vortex_ensure!(!chunks.is_empty(), "Empty chunks were checked earlier");

            // If there is only one chunk, we can return it directly.
            if chunks.len() == 1 {
                return Ok(chunks.into_iter().next().vortex_expect("one chunk"));
            }

            let return_dtype = chunks[0].dtype().clone();
            // Combine the arrays.
            Ok(ChunkedArray::try_new(chunks, return_dtype)?.into_array())
        }
        .boxed())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
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
    use vortex_array::dtype::FieldMask;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::expr::root;
    use vortex_buffer::buffer;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_session::registry::ReadContext;

    use crate::IntoLayout;
    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::OwnedLayoutChildren;
    use crate::layouts::chunked::ChunkedLayout;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::FlatLayout;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::scan::split_by::SplitBy;
    use crate::segments::SegmentId;
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
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = SESSION.clone().with_handle(handle);
            strategy
                .write_stream(
                    ctx,
                    segments2,
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
                    &session,
                )
                .await
        })
        .unwrap();

        (segments, layout)
    }

    fn nested_chunked_layout() -> LayoutRef {
        let dtype = DType::Primitive(PType::U8, NonNullable);
        let ctx = ReadContext::new([]);
        let flat = |segment_id| {
            FlatLayout::new(5, dtype.clone(), SegmentId::from(segment_id), ctx.clone())
                .into_layout()
        };
        let nested = |first_segment_id| {
            ChunkedLayout::new(
                10,
                dtype.clone(),
                OwnedLayoutChildren::layout_children(vec![
                    flat(first_segment_id),
                    flat(first_segment_id + 1),
                ]),
            )
            .into_layout()
        };

        ChunkedLayout::new(
            30,
            dtype.clone(),
            OwnedLayoutChildren::layout_children(vec![nested(0), nested(2), nested(4)]),
        )
        .into_layout()
    }

    #[rstest]
    #[case(0..30, [0, 5, 10, 15, 20, 25, 30])]
    #[case(7..23, [7, 10, 15, 20, 23])]
    fn test_nested_chunked_layout_splits(
        #[case] row_range: std::ops::Range<u64>,
        #[case] expected: impl IntoIterator<Item = u64>,
    ) {
        let layout = nested_chunked_layout();
        let reader = layout
            .new_reader(
                "".into(),
                Arc::new(TestSegments::default()),
                &SESSION,
                &Default::default(),
            )
            .unwrap();

        let splits = SplitBy::Layout
            .splits(reader.as_ref(), &row_range, &[FieldMask::All])
            .unwrap();

        assert_eq!(splits, expected.into_iter().collect::<Vec<_>>());
    }

    #[rstest]
    fn test_chunked_evaluator(
        #[from(chunked_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        block_on(|_h| async {
            let result = layout
                .new_reader("".into(), segments, &SESSION, &Default::default())
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
