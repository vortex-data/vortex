// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::List;
use vortex_array::arrays::ListView;
use vortex_array::arrays::list::ListDataParts;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::dtype::DType;
use vortex_array::matcher::Matcher;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::list::ListLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStream;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Strategy for writing list-typed arrays, with a fallback for non-list dtypes.
///
/// Single-chunk only. For list-typed input the strategy:
///  1. Canonicalizes the input chunk into a [`ListViewArray`].
///  2. Calls [`list_from_list_view`] to rebuild it into zero-copy-to-list form
///     (sorted, gapless, non-overlapping offsets) and produce a [`ListArray`].
///  3. Writes the `elements`, `offsets`, and (when nullable) `validity` columns into
///     separately configurable downstream strategies, producing a single [`ListLayout`].
///
/// For input whose dtype is not [`DType::List`], the stream is forwarded unchanged to the
/// configured `fallback` strategy. This lets `ListLayoutStrategy` slot in as a leaf strategy in
/// a heterogeneous column writer where some columns are lists and others are not.
///
/// # Chunking
///
/// `ListLayoutStrategy` bails on empty or multi-chunk input, matching the convention used by
/// [`FlatLayoutStrategy`](crate::layouts::flat::writer::FlatLayoutStrategy).
///
/// [`ListArray`]: vortex_array::arrays::ListArray
#[derive(Clone)]
pub struct ListLayoutStrategy {
    elements: Arc<dyn LayoutStrategy>,
    offsets: Arc<dyn LayoutStrategy>,
    validity: Arc<dyn LayoutStrategy>,
    fallback: Arc<dyn LayoutStrategy>,
}

impl Default for ListLayoutStrategy {
    /// Routes every child (elements, offsets, validity) and the non-list fallback through
    /// [`FlatLayoutStrategy`]. Override individual children with the `with_*` builder methods.
    fn default() -> Self {
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        Self {
            elements: Arc::clone(&flat),
            offsets: Arc::clone(&flat),
            validity: Arc::clone(&flat),
            fallback: flat,
        }
    }
}

impl ListLayoutStrategy {
    /// Strategy for the `elements` child.
    pub fn with_elements(mut self, elements: Arc<dyn LayoutStrategy>) -> Self {
        self.elements = elements;
        self
    }

    /// Strategy for the `offsets` child.
    pub fn with_offsets(mut self, offsets: Arc<dyn LayoutStrategy>) -> Self {
        self.offsets = offsets;
        self
    }

    /// Strategy for the `validity` child (written only when the list is nullable).
    pub fn with_validity(mut self, validity: Arc<dyn LayoutStrategy>) -> Self {
        self.validity = validity;
        self
    }

    /// Strategy for non-list input, which is forwarded through this strategy unchanged.
    pub fn with_fallback(mut self, fallback: Arc<dyn LayoutStrategy>) -> Self {
        self.fallback = fallback;
        self
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
            // Non-list input: route to the configured fallback strategy unchanged.
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        // Writer wants exactly one chunk
        let Some(chunk) = stream.next().await else {
            vortex_bail!("ListLayoutStrategy needs a single chunk");
        };
        let (sequence_id, array) = chunk?;

        let mut exec_ctx = session.create_execution_ctx();
        let ListDataParts {
            elements,
            offsets,
            validity,
            ..
        } = canonicalize_to_list_parts(array, &mut exec_ctx)?;

        // There is one extra element in `offsets`
        let row_count = offsets.len().saturating_sub(1);
        let validity_array = dtype
            .is_nullable()
            .then(|| {
                validity
                    .execute_mask(row_count, &mut exec_ctx)
                    .map(|m| m.into_array())
            })
            .transpose()?;

        // Spawn each child write onto the runtime so they run concurrently
        let handle = session.handle();
        let (elements_task, offsets_task, validity_task) = {
            let mut sp = sequence_id.descend();
            let mut spawn_layout_writer = |strategy: Arc<dyn LayoutStrategy>, array: ArrayRef| {
                let stream = single_chunk_stream(array.dtype().clone(), sp.advance(), array);
                let child_eof = eof.split_off();
                let ctx = ctx.clone();
                let segment_sink = Arc::clone(&segment_sink);
                let session = session.clone();
                handle.spawn_nested(move |h| async move {
                    let session = session.with_handle(h);
                    strategy
                        .write_stream(ctx, segment_sink, stream, child_eof, &session)
                        .await
                })
            };
            (
                spawn_layout_writer(Arc::clone(&self.elements), elements),
                spawn_layout_writer(Arc::clone(&self.offsets), offsets),
                validity_array.map(|arr| spawn_layout_writer(Arc::clone(&self.validity), arr)),
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
        let list_bytes = self.elements.buffered_bytes()
            + self.offsets.buffered_bytes()
            + self.validity.buffered_bytes();
        list_bytes.max(self.fallback.buffered_bytes())
    }
}

/// Canonicalize a list-dtype array into [`ListDataParts`]. Short-circuits when the input is
/// already a `List` or `ListView` array — otherwise drives the execution loop until one of
/// those forms appears. `ListView` is rebuilt into zero-copy-to-list form via
/// [`list_from_list_view`] before its parts are extracted.
fn canonicalize_to_list_parts(
    array: ArrayRef,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ListDataParts> {
    let canonical = array.execute_until::<AnyList>(exec_ctx)?;
    if let Some(list) = canonical.as_opt::<List>() {
        Ok(list.into_owned().into_data_parts())
    } else if let Some(view) = canonical.as_opt::<ListView>() {
        Ok(list_from_list_view(view.into_owned(), exec_ctx)?.into_data_parts())
    } else {
        unreachable!("AnyList matcher guarantees List or ListView")
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

/// Matcher for `Array<List>` or `Array<ListView>`. Used to short-circuit the execution loop
/// when the input is already in (or directly produces) a list form, avoiding a redundant
/// `ListView` round-trip when the writer already has the parts it needs.
struct AnyList;

impl Matcher for AnyList {
    type Match<'a> = ();

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.as_opt::<List>().is_some() || array.as_opt::<ListView>().is_some()).then_some(())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use super::*;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::table::TableStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;

    fn flat_list_strategy() -> ListLayoutStrategy {
        ListLayoutStrategy::default()
    }

    async fn write<S: LayoutStrategy>(strategy: &S, array: ArrayRef) -> VortexResult<LayoutRef> {
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        strategy
            .write_stream(ArrayContext::empty(), segments, stream, eof, &SESSION)
            .await
    }

    fn i32_list_dtype(nullable: bool) -> DType {
        DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            if nullable {
                Nullability::Nullable
            } else {
                Nullability::NonNullable
            },
        )
    }

    fn create_basic_list(validity: Validity) -> ArrayRef {
        ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5].into_array(),
            buffer![0u32, 2, 5, 5].into_array(),
            validity,
        )
        .unwrap()
        .into_array()
    }

    #[tokio::test]
    async fn basic_non_nullable_input() -> VortexResult<()> {
        let list = create_basic_list(Validity::NonNullable);

        let layout = write(&flat_list_strategy(), list).await?;
        assert_eq!(layout.row_count(), 3);

        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.list, dtype: list(i32), children: 2
        ├── elements: vortex.flat, dtype: i32, segment: 0
        └── offsets: vortex.flat, dtype: u32, segment: 1
        ");
        Ok(())
    }

    #[tokio::test]
    async fn basic_nullable_input() -> VortexResult<()> {
        let list = create_basic_list(Validity::Array(
            BoolArray::from_iter([true, false, true]).into_array(),
        ));

        let layout = write(&flat_list_strategy(), list).await?;
        assert_eq!(layout.row_count(), 3);

        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.list, dtype: list(i32)?, children: 3
        ├── elements: vortex.flat, dtype: i32, segment: 0
        ├── offsets: vortex.flat, dtype: u32, segment: 1
        └── validity: vortex.flat, dtype: bool, segment: 2
        ");
        Ok(())
    }

    /// Non-list input dispatches to the fallback strategy unchanged.
    #[tokio::test]
    async fn non_list_input_routes_to_fallback() -> VortexResult<()> {
        let primitive = buffer![1i32, 2, 3].into_array();
        let layout = write(&flat_list_strategy(), primitive).await?;
        insta::assert_snapshot!(layout.display_tree(), @"vortex.flat, dtype: i32, segment: 0");
        Ok(())
    }

    #[tokio::test]
    async fn empty_stream_errors() {
        let segments = Arc::new(TestSegments::default());
        let (_, eof) = SequenceId::root().split();
        let empty = stream::empty::<VortexResult<(SequenceId, ArrayRef)>>().boxed();
        let stream = SequentialStreamAdapter::new(i32_list_dtype(false), empty).sendable();

        let res = flat_list_strategy()
            .write_stream(ArrayContext::empty(), segments, stream, eof, &SESSION)
            .await;
        assert!(res.is_err())
    }

    #[tokio::test]
    async fn chunked_list_input_without_chunked_strategy_fails() -> VortexResult<()> {
        let chunk0 = ListArray::try_new(
            buffer![1i32, 2].into_array(),
            buffer![0u32, 2].into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();
        let chunk1 = ListArray::try_new(
            buffer![3i32, 4, 5].into_array(),
            buffer![0u32, 3].into_array(),
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();
        let chunked =
            ChunkedArray::try_new(vec![chunk0, chunk1], i32_list_dtype(false))?.into_array();

        let res = write(&flat_list_strategy(), chunked).await;
        assert!(res.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn list_of_struct_tree() -> VortexResult<()> {
        let struct_array = StructArray::from_fields(
            [
                ("a", buffer![1i32, 2, 3, 4, 5].into_array()),
                ("b", buffer![10i32, 20, 30, 40, 50].into_array()),
            ]
            .as_slice(),
        )?
        .into_array();
        let list = ListArray::try_new(
            struct_array,
            buffer![0u32, 2, 5, 5].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        let table_strategy: Arc<dyn LayoutStrategy> =
            Arc::new(TableStrategy::new(Arc::clone(&flat), Arc::clone(&flat)));
        let writer = ListLayoutStrategy::default().with_elements(table_strategy);

        let layout = write(&writer, list).await?;
        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.list, dtype: list({a=i32, b=i32}), children: 2
        ├── elements: vortex.struct, dtype: {a=i32, b=i32}, children: 2
        │   ├── a: vortex.flat, dtype: i32, segment: 1
        │   └── b: vortex.flat, dtype: i32, segment: 2
        └── offsets: vortex.flat, dtype: u32, segment: 0
        ");
        Ok(())
    }

    #[tokio::test]
    async fn list_of_list_tree() -> VortexResult<()> {
        let inner_list = ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            buffer![0u32, 2, 5, 5, 6].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let list = ListArray::try_new(
            inner_list,
            buffer![0u32, 2, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let writer =
            ListLayoutStrategy::default().with_elements(Arc::new(ListLayoutStrategy::default()));
        let layout = write(&writer, list).await?;
        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.list, dtype: list(list(i32)), children: 2
        ├── elements: vortex.list, dtype: list(i32), children: 2
        │   ├── elements: vortex.flat, dtype: i32, segment: 1
        │   └── offsets: vortex.flat, dtype: u32, segment: 2
        └── offsets: vortex.flat, dtype: u32, segment: 0
        ");
        Ok(())
    }

    #[tokio::test]
    async fn list_of_list_of_list_tree() -> VortexResult<()> {
        let innermost = ListArray::try_new(
            buffer![1i32, 2, 3, 4].into_array(),
            buffer![0u32, 2, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let middle = ListArray::try_new(
            innermost,
            buffer![0u32, 2].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let outer =
            ListArray::try_new(middle, buffer![0u32, 1].into_array(), Validity::NonNullable)?
                .into_array();

        let writer = ListLayoutStrategy::default().with_elements(Arc::new(
            ListLayoutStrategy::default().with_elements(Arc::new(ListLayoutStrategy::default())),
        ));
        let layout = write(&writer, outer).await?;
        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.list, dtype: list(list(list(i32))), children: 2
        ├── elements: vortex.list, dtype: list(list(i32)), children: 2
        │   ├── elements: vortex.list, dtype: list(i32), children: 2
        │   │   ├── elements: vortex.flat, dtype: i32, segment: 2
        │   │   └── offsets: vortex.flat, dtype: u32, segment: 3
        │   └── offsets: vortex.flat, dtype: u32, segment: 1
        └── offsets: vortex.flat, dtype: u32, segment: 0
        ");
        Ok(())
    }

    #[tokio::test]
    async fn chunked_list_input_with_chunked_strategy_succeeds() -> VortexResult<()> {
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

        let chunked =
            ChunkedArray::try_new(vec![chunk0, chunk1], i32_list_dtype(false))?.into_array();

        let layout = write(&ChunkedLayoutStrategy::new(flat_list_strategy()), chunked).await?;

        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.chunked, dtype: list(i32), children: 2
        ├── [0]: vortex.list, dtype: list(i32), children: 2
        │   ├── elements: vortex.flat, dtype: i32, segment: 0
        │   └── offsets: vortex.flat, dtype: u32, segment: 1
        └── [1]: vortex.list, dtype: list(i32), children: 2
            ├── elements: vortex.flat, dtype: i32, segment: 2
            └── offsets: vortex.flat, dtype: u32, segment: 3
        ");
        Ok(())
    }
}
