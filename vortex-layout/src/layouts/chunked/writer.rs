// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use vortex_array::dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::paged::writer::write_page;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt as _;

#[derive(Clone)]
pub struct ChunkedLayoutStrategy {
    /// The layout strategy for each chunk.
    pub chunk_strategy: Arc<dyn LayoutStrategy>,
    /// If set, chunks are grouped into pages of at most this many, each serialized into its own
    /// segment rather than inline in this layout's flatbuffer. See [`crate::layouts::paged`].
    pub page_size: Option<NonZeroUsize>,
}

impl ChunkedLayoutStrategy {
    pub fn new<S: LayoutStrategy>(chunk_strategy: S) -> Self {
        Self {
            chunk_strategy: Arc::new(chunk_strategy),
            page_size: None,
        }
    }

    /// Group chunks into pages of at most `page_size` chunks each.
    ///
    /// A layout is a single recursive flatbuffer verified against a table limit, so a file with
    /// enough chunks becomes one its own reader rejects. Paging bounds the tables in any one
    /// flatbuffer to roughly `page_size`. A page size of zero leaves the chunks inline.
    pub fn with_page_size(mut self, page_size: usize) -> Self {
        self.page_size = NonZeroUsize::new(page_size);
        self
    }
}

/// Replace groups of `page_size` consecutive chunks with pages holding them in segments.
async fn paginate(
    children: Vec<LayoutRef>,
    page_size: usize,
    dtype: &DType,
    ctx: &LayoutWriterContext,
    segment_sink: &SegmentSinkRef,
    eof: &mut SequencePointer,
) -> VortexResult<Vec<LayoutRef>> {
    let mut pages = Vec::with_capacity(children.len().div_ceil(page_size));
    for group in children.chunks(page_size) {
        // The chunk boundaries within the page, which travel with it so scans can still be
        // planned at chunk granularity without reading it.
        let mut row_count = 0u64;
        let row_offsets = group
            .iter()
            .map(|layout| {
                row_count = row_count
                    .checked_add(layout.row_count())
                    .ok_or_else(|| vortex_err!("Paged chunk row counts overflow"))?;
                Ok(row_count)
            })
            .collect::<VortexResult<Vec<u64>>>()?;

        let page = ChunkedLayout::new(
            row_count,
            dtype.clone(),
            OwnedLayoutChildren::layout_children(group.to_vec()),
        )
        .into_layout();

        pages.push(
            write_page(
                &page,
                &row_offsets,
                ReadContext::new(ctx.array_ctx().to_ids()),
                segment_sink,
                eof.advance(),
            )
            .await?,
        );
    }
    Ok(pages)
}

#[async_trait]
impl LayoutStrategy for ChunkedLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let dtype2 = dtype.clone();
        let chunk_strategy = Arc::clone(&self.chunk_strategy);
        let handle = session.handle();

        // The eofs used for the chunks should appear _before_ the pages that reference them.
        let mut chunks_eof = eof.split_off();
        let page_ctx = ctx.clone();
        let page_sink = Arc::clone(&segment_sink);

        // We spawn each child to allow parallelism when processing chunks.
        let stream = stream! {
            let mut stream = stream;
            while let Some(chunk) = stream.next().await {
                let chunk_eof = chunks_eof.split_off();

                let chunk_strategy = Arc::clone(&chunk_strategy);
                let ctx = ctx.clone();
                let segment_sink = Arc::clone(&segment_sink);
                let dtype = dtype2.clone();
                let session = session.clone();

                yield handle.spawn_nested(move |handle| async move {
                    let session = session.with_handle(handle);
                    chunk_strategy
                        .write_stream(
                            ctx,
                            segment_sink,
                            SequentialStreamAdapter::new(
                                dtype,
                                stream::iter([chunk]),
                            )
                            .sendable(),
                            chunk_eof,
                            &session,
                        )
                        .await
                })
            }
        };

        // Poll all of our children concurrently to accumulate their layouts.
        let mut child_layouts: Vec<LayoutRef> = stream.buffered(usize::MAX).try_collect().await?;

        if child_layouts.len() == 1 {
            return Ok(child_layouts.pop().vortex_expect("must have one child"));
        }

        let row_count = child_layouts.iter().map(|layout| layout.row_count()).sum();

        if let Some(page_size) = self.page_size {
            child_layouts = paginate(
                child_layouts,
                page_size.get(),
                &dtype,
                &page_ctx,
                &page_sink,
                &mut eof,
            )
            .await?;
        }

        Ok(ChunkedLayout::new(
            row_count,
            dtype,
            OwnedLayoutChildren::layout_children(child_layouts),
        )
        .into_layout())
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use futures::stream;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::expr::root;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;

    use crate::LayoutStrategy;
    use crate::LayoutWriterContext;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialStreamAdapter;
    use crate::sequence::SequentialStreamExt as _;
    use crate::test::new_session;

    /// Six chunks of three rows written with a page size of two: the root should hold three
    /// paged children rather than six inline chunks, and still read back as the same 18 rows.
    #[test]
    fn paged_chunks_round_trip() {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let mut exec = session.create_execution_ctx();
            let segments = Arc::new(TestSegments::default());
            let (mut sequence_id, eof) = SequenceId::root().split();

            let chunks = (0..6i32)
                .map(|chunk| {
                    let base = chunk * 3;
                    let chunk: PrimitiveArray = (base..base + 3).collect();
                    Ok((sequence_id.advance(), chunk.into_array()))
                })
                .collect::<Vec<_>>();

            let layout = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default())
                .with_page_size(2)
                .write_stream(
                    LayoutWriterContext::new(ArrayContext::empty()),
                    Arc::<TestSegments>::clone(&segments),
                    SequentialStreamAdapter::new(
                        DType::Primitive(PType::I32, NonNullable),
                        stream::iter(chunks),
                    )
                    .sendable(),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            assert_eq!(layout.encoding_id().as_ref(), "vortex.chunked");
            assert_eq!(layout.row_count(), 18);
            assert_eq!(layout.nchildren(), 3);

            let page = layout.slot(0).unwrap().unwrap();
            assert_eq!(page.encoding_id().as_ref(), "vortex.paged");
            assert_eq!(page.row_count(), 6);
            assert_eq!(
                page.nchildren(),
                0,
                "a page's subtree lives in its segment, not inline"
            );

            let reader = layout
                .new_reader("".into(), segments, &session, &Default::default())
                .unwrap();
            let expr = root().bind(reader.dtype()).unwrap();
            let result = reader
                .projection_evaluation(&(0..18), &expr, MaskFuture::new_true(18))
                .unwrap()
                .await
                .unwrap();

            let expected: PrimitiveArray = (0i32..18).collect();
            assert_arrays_eq!(result, expected.into_array(), &mut exec);
        })
    }
}
