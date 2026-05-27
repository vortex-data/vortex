// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::list::ListDataParts;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
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
/// Single-chunk only. The strategy:
///  1. Canonicalizes the input chunk into a [`ListViewArray`].
///  2. Calls [`list_from_list_view`] to rebuild it into zero-copy-to-list form
///     (sorted, gapless, non-overlapping offsets) and produce a [`ListArray`].
///  3. Writes the `elements`, `offsets`, and (when nullable) `validity` columns into
///     separately configurable downstream strategies, producing a single [`ListLayout`].
///
/// # Chunking
///
/// `ListLayoutStrategy` bails on empty or multi-chunk input, matching the convention used by
/// [`FlatLayoutStrategy`](crate::layouts::flat::writer::FlatLayoutStrategy).
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
        mut stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        if !dtype.is_list() {
            vortex_bail!("ListLayoutStrategy requires a List dtype, got {dtype}");
        }

        // Writer wants exactly one chunk
        let Some(chunk) = stream.next().await else {
            vortex_bail!("ListLayoutStrategy needs a single chunk");
        };
        let (sequence_id, array) = chunk?;

        // Canonicalize to ListView, then rebuild into zctl
        let mut exec_ctx = session.create_execution_ctx();
        let ListDataParts {
            elements,
            offsets,
            validity,
            ..
        } = list_from_list_view(array.execute::<ListViewArray>(&mut exec_ctx)?)?.into_data_parts();

        // There is one extra element in `offsets`
        let row_count = offsets.len().saturating_sub(1);
        let validity_array = if dtype.is_nullable() {
            Some(
                validity
                    .execute_mask(row_count, &mut exec_ctx)?
                    .into_array(),
            )
        } else {
            None
        };

        // Spawn each child write onto the runtime so they run concurrently
        let handle = session.handle();
        let (elements_task, offsets_task, validity_task) = {
            let mut sp = sequence_id.descend();
            let mut spawn_layout_writer = |strategy: Arc<dyn LayoutStrategy>, array: ArrayRef| {
                let stream = single_chunk_stream(array.dtype().clone(), sp.advance(), array);
                let child_eof = eof.split_off();
                let ctx = ctx.clone();
                let segment_sink = segment_sink.clone();
                let session = session.clone();
                handle.spawn_nested(move |h| async move {
                    let session = session.with_handle(h);
                    strategy
                        .write_stream(ctx, segment_sink, stream, child_eof, &session)
                        .await
                })
            };
            (
                spawn_layout_writer(self.elements.clone(), elements),
                spawn_layout_writer(self.offsets.clone(), offsets),
                validity_array.map(|arr| spawn_layout_writer(self.validity.clone(), arr)),
            )
        };

        // Should not have more than one chunk
        if stream.next().await.is_some() {
            vortex_bail!("ListLayoutStrategy received more than a single chunk");
        }

        let (elements_layout, offsets_layout, validity_layout) =
            futures::try_join!(elements_task, offsets_task, async move {
                match validity_task {
                    Some(t) => t.await.map(Some),
                    None => Ok(None),
                }
            },)?;

        Ok(ListLayout::new(dtype, elements_layout, offsets_layout, validity_layout).into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.elements.buffered_bytes()
            + self.offsets.buffered_bytes()
            + self.validity.buffered_bytes()
    }
}

/// Wrap a single array as a one-shot [`SendableSequentialStream`] for handoff to a child writer.
fn single_chunk_stream(
    dtype: DType,
    sequence_id: SequenceId,
    array: ArrayRef,
) -> SendableSequentialStream {
    SequentialStreamAdapter::new(
        dtype,
        stream::once(async move { Ok((sequence_id, array)) }).boxed(),
    )
    .sendable()
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
                usize::try_from(row_range.start).unwrap()..usize::try_from(row_range.end).unwrap(),
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

    // -- tree shape visualization ---------------------------------------------------------
    //
    // These tests are mostly for development/inspection — they show what the resulting
    // layout tree looks like for various input shapes. Run with `--nocapture` to see the
    // pretty-printed trees:
    //
    //   cargo test -p vortex-layout layouts::list::writer::tests::tree -- --nocapture

    use vortex_array::ArrayContext as _ArrayContextAlias;

    /// Write `array` directly through `ListLayoutStrategy` (no ChunkedLayoutStrategy wrap)
    /// and return the resulting top-level layout.
    async fn write_through_list_strategy(array: ArrayRef) -> crate::LayoutRef {
        let segments = Arc::new(TestSegments::default());
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        let writer =
            ListLayoutStrategy::new(Arc::clone(&flat), Arc::clone(&flat), Arc::clone(&flat));
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        writer
            .write_stream(_ArrayContextAlias::empty(), segments, stream, eof, &SESSION)
            .await
            .unwrap()
    }

    /// Wrap `ListLayoutStrategy` in `ChunkedLayoutStrategy` and write `array`. For a chunked
    /// input, each chunk becomes one `ListLayout` under the outer `ChunkedLayout`.
    async fn write_through_chunked_list_strategy(array: ArrayRef) -> crate::LayoutRef {
        use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
        let segments = Arc::new(TestSegments::default());
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        let list_strategy =
            ListLayoutStrategy::new(Arc::clone(&flat), Arc::clone(&flat), Arc::clone(&flat));
        let writer = ChunkedLayoutStrategy::new(list_strategy);
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        writer
            .write_stream(_ArrayContextAlias::empty(), segments, stream, eof, &SESSION)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn tree_shape_single_chunk_non_nullable() {
        let elements = buffer![1i32, 2, 3, 4, 5].into_array();
        let offsets = buffer![0u32, 2, 5, 5].into_array();
        let list = ListArray::try_new(elements, offsets, Validity::NonNullable)
            .unwrap()
            .into_array();

        let layout = write_through_list_strategy(list).await;
        let tree = layout.display_tree().to_string();
        eprintln!("--- single-chunk non-nullable ---\n{tree}");
        // Top level is a single ListLayout with 2 children (elements, offsets).
        assert!(tree.starts_with("vortex.list"));
        assert!(tree.contains("elements"));
        assert!(tree.contains("offsets"));
        assert!(!tree.contains("validity"));
    }

    #[tokio::test]
    async fn tree_shape_single_chunk_nullable() {
        let elements = buffer![10i32, 20, 30, 40, 50].into_array();
        let offsets = buffer![0u32, 2, 3, 5].into_array();
        let validity = Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
        let list = ListArray::try_new(elements, offsets, validity)
            .unwrap()
            .into_array();

        let layout = write_through_list_strategy(list).await;
        let tree = layout.display_tree().to_string();
        eprintln!("--- single-chunk nullable ---\n{tree}");
        // Top level is a single ListLayout with 3 children (elements, offsets, validity).
        assert!(tree.starts_with("vortex.list"));
        assert!(tree.contains("elements"));
        assert!(tree.contains("offsets"));
        assert!(tree.contains("validity"));
    }

    #[tokio::test]
    async fn tree_shape_multi_chunk_via_chunked_strategy() {
        use std::sync::Arc as StdArc;
        use vortex_array::arrays::ChunkedArray;
        use vortex_array::dtype::DType;
        use vortex_array::dtype::Nullability;
        use vortex_array::dtype::PType;

        // Two list-array chunks fed through ChunkedArray -> ChunkedLayoutStrategy.
        let chunk0 = ListArray::try_new(
            buffer![1i32, 2, 3].into_array(),
            buffer![0u32, 2, 3].into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();
        let chunk1 = ListArray::try_new(
            buffer![4i32, 5, 6, 7].into_array(),
            buffer![0u32, 1, 4].into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();

        let list_dtype = DType::List(
            StdArc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::NonNullable,
        );
        let chunked = ChunkedArray::try_new(vec![chunk0, chunk1], list_dtype)
            .unwrap()
            .into_array();

        let layout = write_through_chunked_list_strategy(chunked).await;
        let tree = layout.display_tree().to_string();
        eprintln!("--- multi-chunk via ChunkedLayoutStrategy ---\n{tree}");
        // Top level is a ChunkedLayout containing two ListLayouts.
        assert!(tree.starts_with("vortex.chunked"));
        assert_eq!(tree.matches("vortex.list").count(), 2);
    }
}
