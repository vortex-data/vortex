// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::future::try_join_all;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::list::ListLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStream;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Strategy for writing list-typed arrays as an Apache Arrow-style [`ListLayout`].
///
/// For each input chunk, the strategy:
///  1. Canonicalizes the chunk into a [`ListViewArray`].
///  2. Calls [`list_from_list_view`] to rebuild it into zero-copy-to-list form
///     (sorted, gapless, non-overlapping offsets) and produce a [`ListArray`].
///  3. Writes the `elements`, `n+1` `offsets`, and (when nullable) `validity` columns into
///     separately configurable downstream strategies, producing a single [`ListLayout`] for
///     that chunk.
///
/// When the input stream contains multiple chunks, each chunk becomes one `ListLayout` and the
/// strategy wraps them in a [`ChunkedLayout`].
///
/// [`ListArray`]: vortex_array::arrays::ListArray
pub struct ListLayoutStrategy {
    elements: Arc<dyn LayoutStrategy>,
    offsets: Arc<dyn LayoutStrategy>,
    validity: Arc<dyn LayoutStrategy>,
}

impl ListLayoutStrategy {
    pub fn new(
        elements: Arc<dyn LayoutStrategy>,
        offsets: Arc<dyn LayoutStrategy>,
        validity: Arc<dyn LayoutStrategy>,
    ) -> Self {
        Self {
            elements,
            offsets,
            validity,
        }
    }
}

#[async_trait]
impl LayoutStrategy for ListLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        if !dtype.is_list() {
            return Err(vortex_err!(
                "ListLayoutStrategy requires a List dtype, got {dtype}"
            ));
        }
        let is_nullable = dtype.is_nullable();

        let mut stream = stream;
        let mut chunk_layouts: Vec<LayoutRef> = Vec::new();

        while let Some(item) = stream.next().await {
            let (sequence_id, array) = item?;
            let chunk_eof = eof.split_off();

            let chunk_layout = self
                .write_chunk(
                    ctx.clone(),
                    Arc::clone(&segment_sink),
                    sequence_id,
                    chunk_eof,
                    session,
                    &dtype,
                    is_nullable,
                    array,
                )
                .await?;
            chunk_layouts.push(chunk_layout);
        }

        if chunk_layouts.is_empty() {
            // Empty stream: emit an empty ListLayout. We write empty children of the same dtype
            // (with u32 offsets) so the children's encoding ID is still derivable.
            let chunk_layout = self
                .write_empty_chunk(ctx, segment_sink, eof, session, dtype.clone(), is_nullable)
                .await?;
            return Ok(chunk_layout);
        }

        if chunk_layouts.len() == 1 {
            return Ok(chunk_layouts.pop().expect("len == 1"));
        }

        let row_count = chunk_layouts.iter().map(|l| l.row_count()).sum();
        Ok(ChunkedLayout::new(
            row_count,
            dtype,
            OwnedLayoutChildren::layout_children(chunk_layouts),
        )
        .into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.elements.buffered_bytes()
            + self.offsets.buffered_bytes()
            + self.validity.buffered_bytes()
    }
}

impl ListLayoutStrategy {
    #[allow(clippy::too_many_arguments)]
    async fn write_chunk(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        sequence_id: SequenceId,
        mut chunk_eof: SequencePointer,
        session: &VortexSession,
        dtype: &DType,
        is_nullable: bool,
        array: ArrayRef,
    ) -> VortexResult<LayoutRef> {
        let mut exec_ctx = session.create_execution_ctx();

        // Canonicalize then rebuild into ZCTL form.
        let listview = array.execute::<ListViewArray>(&mut exec_ctx)?;
        let list = list_from_list_view(listview)?;
        let parts = list.into_data_parts();
        let row_count = parts.offsets.len().saturating_sub(1);

        let elements_dtype = parts.elements.dtype().clone();
        let offsets_dtype = parts.offsets.dtype().clone();

        // Build single-chunk streams for each child. Each child sees one array and EOF.
        let mut sequence_pointer = sequence_id.descend();
        let elements_seq = sequence_pointer.advance();
        let offsets_seq = sequence_pointer.advance();
        let validity_seq = if is_nullable {
            Some(sequence_pointer.advance())
        } else {
            None
        };
        drop(sequence_pointer);

        let elements_eof = chunk_eof.split_off();
        let offsets_eof = chunk_eof.split_off();
        let validity_eof = if is_nullable {
            Some(chunk_eof.split_off())
        } else {
            None
        };
        drop(chunk_eof);

        let elements_stream = SequentialStreamAdapter::new(
            elements_dtype,
            futures::stream::once(async move { Ok((elements_seq, parts.elements)) }).boxed(),
        )
        .sendable();
        let offsets_stream = SequentialStreamAdapter::new(
            offsets_dtype,
            futures::stream::once(async move { Ok((offsets_seq, parts.offsets)) }).boxed(),
        )
        .sendable();

        let validity_array = if is_nullable {
            Some(
                parts
                    .validity
                    .execute_mask(row_count, &mut exec_ctx)?
                    .into_array(),
            )
        } else {
            None
        };
        let validity_stream = validity_array.map(|arr| {
            let seq = validity_seq.expect("validity sequence id");
            SequentialStreamAdapter::new(
                DType::Bool(Nullability::NonNullable),
                futures::stream::once(async move { Ok((seq, arr)) }).boxed(),
            )
            .sendable()
        });

        let handle = session.handle();

        let elements_strategy = Arc::clone(&self.elements);
        let elements_ctx = ctx.clone();
        let elements_sink = Arc::clone(&segment_sink);
        let elements_session = session.clone();
        let elements_task = handle.spawn_nested(move |h| async move {
            let session = elements_session.with_handle(h);
            elements_strategy
                .write_stream(
                    elements_ctx,
                    elements_sink,
                    elements_stream,
                    elements_eof,
                    &session,
                )
                .await
        });

        let offsets_strategy = Arc::clone(&self.offsets);
        let offsets_ctx = ctx.clone();
        let offsets_sink = Arc::clone(&segment_sink);
        let offsets_session = session.clone();
        let offsets_task = handle.spawn_nested(move |h| async move {
            let session = offsets_session.with_handle(h);
            offsets_strategy
                .write_stream(
                    offsets_ctx,
                    offsets_sink,
                    offsets_stream,
                    offsets_eof,
                    &session,
                )
                .await
        });

        let mut tasks = vec![elements_task, offsets_task];
        if let (Some(validity_stream), Some(validity_eof)) = (validity_stream, validity_eof) {
            let validity_strategy = Arc::clone(&self.validity);
            let validity_ctx = ctx;
            let validity_sink = segment_sink;
            let validity_session = session.clone();
            tasks.push(handle.spawn_nested(move |h| async move {
                let session = validity_session.with_handle(h);
                validity_strategy
                    .write_stream(
                        validity_ctx,
                        validity_sink,
                        validity_stream,
                        validity_eof,
                        &session,
                    )
                    .await
            }));
        }

        let mut child_layouts = try_join_all(tasks).await?;
        let elements_layout = child_layouts.remove(0);
        let offsets_layout = child_layouts.remove(0);
        let validity_layout = if is_nullable {
            Some(child_layouts.remove(0))
        } else {
            None
        };

        Ok(ListLayout::try_new(
            dtype.clone(),
            elements_layout,
            offsets_layout,
            validity_layout,
        )?
        .into_layout())
    }

    async fn write_empty_chunk(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        mut eof: SequencePointer,
        session: &VortexSession,
        dtype: DType,
        is_nullable: bool,
    ) -> VortexResult<LayoutRef> {
        let elements_dtype = dtype
            .as_list_element_opt()
            .ok_or_else(|| vortex_err!("ListLayoutStrategy requires a List dtype, got {dtype}"))?
            .as_ref()
            .clone();

        let elements_eof = eof.split_off();
        let offsets_eof = eof.split_off();
        let validity_eof = if is_nullable { Some(eof.split_off()) } else { None };

        let elements_layout = self
            .elements
            .write_stream(
                ctx.clone(),
                Arc::clone(&segment_sink),
                empty_stream(elements_dtype),
                elements_eof,
                session,
            )
            .await?;
        // Offsets buffer of length 1 (a single 0 boundary) representing 0 lists.
        // We can't easily synthesize a single-element primitive array here without coupling to
        // a builder, so we just emit an empty offsets buffer and rely on `row_count()` returning
        // 0 via `saturating_sub`. The reader treats `offsets.row_count() == 0` as 0 rows.
        let offsets_layout = self
            .offsets
            .write_stream(
                ctx.clone(),
                Arc::clone(&segment_sink),
                empty_stream(DType::Primitive(PType::U32, Nullability::NonNullable)),
                offsets_eof,
                session,
            )
            .await?;
        let validity_layout = if let Some(validity_eof) = validity_eof {
            Some(
                self.validity
                    .write_stream(
                        ctx,
                        segment_sink,
                        empty_stream(DType::Bool(Nullability::NonNullable)),
                        validity_eof,
                        session,
                    )
                    .await?,
            )
        } else {
            None
        };

        Ok(ListLayout::try_new(dtype, elements_layout, offsets_layout, validity_layout)?
            .into_layout())
    }
}

fn empty_stream(dtype: DType) -> SendableSequentialStream {
    SequentialStreamAdapter::new(dtype, futures::stream::empty().boxed()).sendable()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::root;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::list::writer::ListLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;

    async fn round_trip(list: ArrayRef) {
        let segments = Arc::new(TestSegments::default());
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        let writer = ListLayoutStrategy::new(
            Arc::clone(&flat),
            Arc::clone(&flat),
            Arc::clone(&flat),
        );

        let (ptr, eof) = SequenceId::root().split();
        let stream = list.clone().to_array_stream().sequenced(ptr);

        let layout = writer
            .write_stream(
                ArrayContext::empty(),
                Arc::<TestSegments>::clone(&segments),
                stream,
                eof,
                &SESSION,
            )
            .await
            .unwrap();

        let reader = layout
            .new_reader(Arc::from("test"), segments, &SESSION)
            .unwrap();

        let row_count = usize::try_from(layout.row_count()).unwrap();
        let result = reader
            .projection_evaluation(
                &(0..layout.row_count()),
                &root(),
                MaskFuture::new_true(row_count),
            )
            .unwrap()
            .await
            .unwrap();

        assert_arrays_eq!(result, list);
    }

    #[tokio::test]
    async fn round_trip_non_nullable() {
        let elements = buffer![1i32, 2, 3, 4, 5].into_array();
        let offsets = buffer![0u32, 2, 5, 5].into_array(); // 3 lists: [1,2], [3,4,5], []
        let list = ListArray::try_new(elements, offsets, Validity::NonNullable)
            .unwrap()
            .into_array();
        round_trip(list).await;
    }

    #[tokio::test]
    async fn round_trip_nullable() {
        let elements = buffer![10i32, 20, 30, 40, 50].into_array();
        let offsets = buffer![0u32, 2, 3, 5].into_array(); // 3 lists
        let validity = Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
        let list = ListArray::try_new(elements, offsets, validity)
            .unwrap()
            .into_array();
        round_trip(list).await;
    }
}
