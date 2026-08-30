// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::future::try_join_all;
use futures::stream::once;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::List;
use vortex_array::arrays::ListView;
use vortex_array::arrays::list::ListDataParts;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::matcher::Matcher;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::list::ListLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Strategy for writing list-typed arrays, with a fallback for non-list dtypes.
///
/// This is a *structural* writer that decomposes a list column into independent `elements`,
/// `offsets`, and (when nullable) `validity` sub-columns, each written through its own downstream
/// strategy, producing a single [`ListLayout`].
///
/// For list-typed input the strategy accepts exactly one chunk, canonicalizes its outer list
/// container (rebuilding a [`ListView`] via [`list_from_list_view`] when necessary), and writes
/// its encoded `elements`, `offsets`, and `validity` children concurrently. Chunking belongs
/// outside this strategy: callers with a multi-chunk stream should wrap it in a
/// [`ChunkedLayoutStrategy`]. Keeping the input chunk intact preserves the compressor's encoding
/// choices for both elements and offsets.
///
/// For non-list input, the stream is forwarded unchanged to the configured `fallback` strategy.
///
/// [`ChunkedLayoutStrategy`]: crate::layouts::chunked::writer::ChunkedLayoutStrategy
#[derive(Clone)]
pub struct ListLayoutStrategy {
    elements: Arc<dyn LayoutStrategy>,
    offsets: Arc<dyn LayoutStrategy>,
    validity: Arc<dyn LayoutStrategy>,
    fallback: Arc<dyn LayoutStrategy>,
}

impl Default for ListLayoutStrategy {
    /// Writes every child and the non-list fallback through [`FlatLayoutStrategy`].
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
    /// Strategy used for every list child. A later per-child override takes precedence.
    pub fn with_children_strategy(mut self, strategy: Arc<dyn LayoutStrategy>) -> Self {
        self.validity = Arc::clone(&strategy);
        self.offsets = Arc::clone(&strategy);
        self.elements = strategy;
        self
    }

    /// Strategy used for the elements child.
    pub fn with_elements_strategy(mut self, elements: Arc<dyn LayoutStrategy>) -> Self {
        self.elements = elements;
        self
    }

    /// Strategy used for the offsets child.
    pub fn with_offsets_strategy(mut self, offsets: Arc<dyn LayoutStrategy>) -> Self {
        self.offsets = offsets;
        self
    }

    /// Strategy used for the validity child.
    pub fn with_validity_strategy(mut self, validity: Arc<dyn LayoutStrategy>) -> Self {
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
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        if !dtype.is_list() {
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        let Some((sequence_id, array)) = stream.next().await.transpose()? else {
            vortex_bail!("ListLayoutStrategy needs exactly one chunk");
        };
        if stream.next().await.is_some() {
            vortex_bail!("ListLayoutStrategy received more than one chunk");
        }

        let mut exec_ctx = session.create_execution_ctx();
        let ListDataParts {
            elements,
            offsets,
            validity,
            ..
        } = canonicalize_to_list_parts(array, &mut exec_ctx)?;
        let row_count = offsets.len().saturating_sub(1);
        let mut sequence = sequence_id.descend();
        let mut child_specs = vec![
            (elements, Arc::clone(&self.elements), sequence.advance()),
            (offsets, Arc::clone(&self.offsets), sequence.advance()),
        ];
        if dtype.is_nullable() {
            child_specs.push((
                validity
                    .execute_mask(row_count, &mut exec_ctx)?
                    .into_array(),
                Arc::clone(&self.validity),
                sequence.advance(),
            ));
        }

        let handle = session.handle();
        let layout_futures: Vec<_> = child_specs
            .into_iter()
            .map(|(child, strategy, child_sequence)| {
                let child_dtype = child.dtype().clone();
                let child_stream = SequentialStreamAdapter::new(
                    child_dtype,
                    once(async move { Ok((child_sequence, child)) }),
                )
                .sendable();
                let child_eof = eof.split_off();
                let ctx = ctx.clone();
                let segment_sink = Arc::clone(&segment_sink);
                let session = session.clone();
                handle.spawn_nested(move |_| async move {
                    strategy
                        .write_stream(ctx, segment_sink, child_stream, child_eof, &session)
                        .await
                })
            })
            .collect();

        let mut layouts = try_join_all(layout_futures).await?.into_iter();
        let elements_layout = layouts.next().vortex_expect("elements layout present");
        let offsets_layout = layouts.next().vortex_expect("offsets layout present");
        let validity_layout = dtype
            .is_nullable()
            .then(|| layouts.next().vortex_expect("validity layout present"));

        Ok(ListLayout::new(dtype, elements_layout, offsets_layout, validity_layout).into_layout())
    }
}

/// Canonicalize a list-dtype array into [`ListDataParts`].
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

/// Matcher for `Array<List>` or `Array<ListView>`.
struct AnyList;

impl Matcher for AnyList {
    type Match<'a> = ();

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.as_opt::<List>().is_some() || array.as_opt::<ListView>().is_some()).then_some(())
    }
}

#[cfg(test)]
mod tests {
    use futures::stream;
    use vortex_array::ArrayContext;
    use vortex_array::MaskFuture;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::root;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_io::session::RuntimeSession;

    use super::*;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::session::LayoutSession;

    fn layout_test_session() -> VortexSession {
        vortex_array::array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_tokio()
    }

    fn flat_list_strategy() -> ListLayoutStrategy {
        ListLayoutStrategy::default()
    }

    async fn write<S: LayoutStrategy>(strategy: &S, array: ArrayRef) -> VortexResult<LayoutRef> {
        let session = layout_test_session();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        strategy
            .write_stream(
                ArrayContext::empty().into(),
                segments,
                stream,
                eof,
                &session,
            )
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

    #[tokio::test]
    async fn preserves_encoded_element_leaf() -> VortexResult<()> {
        let elements = DictArray::try_new(
            buffer![0u8, 1, 0, 2, 1].into_array(),
            VarBinArray::from(vec!["alpha", "beta", "gamma"]).into_array(),
        )?
        .into_array();
        let list = ListArray::try_new(
            elements,
            buffer![0u32, 2, 5].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let session = layout_test_session();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let layout = flat_list_strategy()
            .write_stream(
                ArrayContext::empty().into(),
                Arc::<TestSegments>::clone(&segments),
                list.to_array_stream().sequenced(ptr),
                eof,
                &session,
            )
            .await?;

        let elements_layout = layout
            .slot(0)?
            .vortex_expect("ListLayout elements child is present");
        let reader =
            elements_layout.new_reader("".into(), segments, &session, &Default::default())?;
        let expr = root().bind(reader.dtype())?;
        let read = reader
            .projection_evaluation(
                &(0..elements_layout.row_count()),
                &expr,
                MaskFuture::new_true(usize::try_from(elements_layout.row_count())?),
            )?
            .await?;

        assert!(read.is::<Dict>());
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
        let session = layout_test_session();

        let res = flat_list_strategy()
            .write_stream(
                ArrayContext::empty().into(),
                segments,
                stream,
                eof,
                &session,
            )
            .await;
        assert!(res.is_err())
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
