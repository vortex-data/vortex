// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::list::ListDataParts;
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

/// Strategy for writing list-typed arrays.
///
/// For each input chunk, the strategy:
///  1. Canonicalizes the chunk into a [`ListViewArray`].
///  2. Calls [`list_from_list_view`] to rebuild it into zero-copy-to-list form
///     (sorted, gapless, non-overlapping offsets) and produce a [`ListArray`].
///  3. Writes the `elements`, `n+1` `offsets`, and (when nullable) `validity` columns into
///     separately configurable downstream strategies, producing a single [`ListLayout`] for
///     that chunk.
///
/// # Multi-chunk handling
///
/// Each input chunk is a self-contained list array — its own elements buffer, its own
/// `n_i + 1` offsets starting from 0, its own validity. When the stream contains multiple
/// chunks, the strategy produces one `ListLayout` per chunk and wraps them in a
/// [`ChunkedLayout`], rather than merging them into a single `ListLayout` by rebasing offsets
/// across chunks.
///
/// This mirrors how `ChunkedReader` reads back the column: it returns a `ChunkedArray` of
/// per-chunk `ListArray`s rather than concatenating into one big `ListArray`. The chunk
/// boundary is preserved end-to-end, consistent with how every other dtype's chunked column
/// is handled in Vortex.
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
        let mut chunk_layouts = Vec::<LayoutRef>::new();

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
        // Canonicalize to ListView, then rebuild into zero-copy-to-list form with `n+1`
        // monotonic offsets.
        let ListDataParts {
            elements,
            offsets,
            validity,
            ..
        } = list_from_list_view(array.execute::<ListViewArray>(&mut exec_ctx)?)?
            .into_data_parts();
        // `offsets` is the Arrow-canonical `n+1` entries, so the list count is one less.
        let validity_array = is_nullable
            .then(|| {
                validity
                    .execute_mask(offsets.len().saturating_sub(1), &mut exec_ctx)
                    .map(|m| m.into_array())
            })
            .transpose()?;

        // Closure to spawn one child writer with its own sequence id and EOF pointer, advanced
        // in invocation order: elements, offsets, validity (when nullable).
        let mut sp = sequence_id.descend();
        let handle = session.handle();
        let mut spawn = |strategy: &Arc<dyn LayoutStrategy>, array: ArrayRef| {
            spawn_layout_write(
                &handle,
                Arc::clone(strategy),
                ctx.clone(),
                Arc::clone(&segment_sink),
                session,
                single_chunk_stream(array.dtype().clone(), sp.advance(), array),
                chunk_eof.split_off(),
            )
        };

        let elements_task = spawn(&self.elements, elements);
        let offsets_task = spawn(&self.offsets, offsets);
        let validity_task = validity_array.map(|arr| spawn(&self.validity, arr));
        drop(spawn);

        let (elements_layout, offsets_layout, validity_layout) = futures::try_join!(
            elements_task,
            offsets_task,
            async {
                match validity_task {
                    Some(task) => task.await.map(Some),
                    None => Ok(None),
                }
            }
        )?;

        Ok(ListLayout::try_new(
            dtype.clone(),
            elements_layout,
            offsets_layout,
            validity_layout,
        )?
        .into_layout())
    }

    /// Empty-stream variant: produces a `ListLayout` whose children all encode 0 rows.
    /// `offsets.row_count() == 0` is treated as 0 lists by [`ListLayout::row_count`].
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

        let mut write_empty = |strategy: &Arc<dyn LayoutStrategy>, child_dtype: DType| {
            write_empty_child(
                Arc::clone(strategy),
                ctx.clone(),
                Arc::clone(&segment_sink),
                session,
                child_dtype,
                eof.split_off(),
            )
        };

        let elements_layout = write_empty(&self.elements, elements_dtype).await?;
        let offsets_layout = write_empty(
            &self.offsets,
            DType::Primitive(PType::U32, Nullability::NonNullable),
        )
        .await?;
        let validity_layout = if is_nullable {
            Some(
                write_empty(&self.validity, DType::Bool(Nullability::NonNullable)).await?,
            )
        } else {
            None
        };

        Ok(
            ListLayout::try_new(dtype, elements_layout, offsets_layout, validity_layout)?
                .into_layout(),
        )
    }
}

fn empty_stream(dtype: DType) -> SendableSequentialStream {
    SequentialStreamAdapter::new(dtype, futures::stream::empty().boxed()).sendable()
}

/// Drive a single child writer with an empty stream of the given `dtype`.
async fn write_empty_child(
    strategy: Arc<dyn LayoutStrategy>,
    ctx: ArrayContext,
    sink: SegmentSinkRef,
    session: &VortexSession,
    dtype: DType,
    eof: SequencePointer,
) -> VortexResult<LayoutRef> {
    strategy
        .write_stream(ctx, sink, empty_stream(dtype), eof, session)
        .await
}

/// Wrap a single array as a one-shot [`SendableSequentialStream`] for handoff to a child writer.
fn single_chunk_stream(
    dtype: DType,
    sequence_id: SequenceId,
    array: ArrayRef,
) -> SendableSequentialStream {
    SequentialStreamAdapter::new(
        dtype,
        futures::stream::once(async move { Ok((sequence_id, array)) }).boxed(),
    )
    .sendable()
}

/// Spawn a child layout writer task onto the session handle.
///
/// Captures the strategy, ctx, sink, and a cloned session so the spawned future is `'static`.
fn spawn_layout_write(
    handle: &vortex_io::runtime::Handle,
    strategy: Arc<dyn LayoutStrategy>,
    ctx: ArrayContext,
    sink: SegmentSinkRef,
    session: &VortexSession,
    stream: SendableSequentialStream,
    eof: SequencePointer,
) -> vortex_io::runtime::Task<VortexResult<LayoutRef>> {
    let session = session.clone();
    handle.spawn_nested(move |h| async move {
        let session = session.with_handle(h);
        strategy.write_stream(ctx, sink, stream, eof, &session).await
    })
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
        let writer =
            ListLayoutStrategy::new(Arc::clone(&flat), Arc::clone(&flat), Arc::clone(&flat));

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

    /// Writes a list, then reads back only a sub-range to exercise projection over a slice.
    async fn round_trip_subset(list: ArrayRef, row_range: std::ops::Range<u64>) {
        let segments = Arc::new(TestSegments::default());
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        let writer =
            ListLayoutStrategy::new(Arc::clone(&flat), Arc::clone(&flat), Arc::clone(&flat));

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

        let mask_len = usize::try_from(row_range.end - row_range.start).unwrap();
        let result = reader
            .projection_evaluation(&row_range, &root(), MaskFuture::new_true(mask_len))
            .unwrap()
            .await
            .unwrap();

        let expected = list
            .slice(
                usize::try_from(row_range.start).unwrap()
                    ..usize::try_from(row_range.end).unwrap(),
            )
            .unwrap();
        assert_arrays_eq!(result, expected);
    }

    #[tokio::test]
    async fn round_trip_subset_non_nullable() {
        // 5 lists: [1,2], [3], [], [4,5,6], [7]
        let elements = buffer![1i32, 2, 3, 4, 5, 6, 7].into_array();
        let offsets = buffer![0u32, 2, 3, 3, 6, 7].into_array();
        let list = ListArray::try_new(elements, offsets, Validity::NonNullable)
            .unwrap()
            .into_array();
        // Read the middle three lists: [3], [], [4,5,6]
        round_trip_subset(list, 1..4).await;
    }

    #[tokio::test]
    async fn round_trip_subset_nullable() {
        // 4 lists with validity [true, false, true, true]:
        // [10,20], null, [30], [40,50,60]
        let elements = buffer![10i32, 20, 30, 40, 50, 60].into_array();
        let offsets = buffer![0u32, 2, 2, 3, 6].into_array();
        let validity =
            Validity::Array(BoolArray::from_iter([true, false, true, true]).into_array());
        let list = ListArray::try_new(elements, offsets, validity)
            .unwrap()
            .into_array();
        // Read lists 1..3: null, [30]
        round_trip_subset(list, 1..3).await;
    }
}
