// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::future::try_join_all;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::List;
use vortex_array::arrays::ListView;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::list::ListDataParts;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::matcher::Matcher;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriter;
use crate::LayoutWriterContext;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::list::ListLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SequenceId;
use crate::strategy::LayoutWriterActor;

/// Strategy for writing list-typed arrays, with a fallback for non-list dtypes.
///
/// This is a *structural* writer that decomposes a list column into independent `elements`,
/// `offsets`, and (when nullable) `validity` sub-columns, each written through its own downstream
/// strategy, producing a single [`ListLayout`].
///
/// For list-typed input the strategy transposes the whole column stream into three sub-streams:
///  1. Each chunk is canonicalized to a [`ListArray`] (rebuilding a [`ListView`] via
///     [`list_from_list_view`] when necessary).
///  2. `offsets` are rebased to global `u64` positions (cumulative across chunks) so the single
///     `offsets` child indexes into the concatenated `elements` child.
///  3. `elements`, `offsets`, and `validity` are streamed to their child strategies concurrently.
///
/// For input whose dtype is not [`DType::List`], the stream is forwarded unchanged to the
/// configured `fallback` strategy.
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

impl LayoutStrategy for ListLayoutStrategy {
    fn new_writer(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        dtype: DType,
        session: &VortexSession,
    ) -> VortexResult<Box<dyn LayoutWriter>> {
        if !dtype.is_list() {
            return self.fallback.new_writer(ctx, segment_sink, dtype, session);
        }

        let is_nullable = dtype.is_nullable();
        let element_dtype = dtype
            .as_list_element_opt()
            .vortex_expect("DType is List")
            .as_ref()
            .clone();
        // Global (whole-column) offsets are cumulative and may exceed the input offset width,
        // so definsively widen.
        let offsets_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);

        let mut child_specs = vec![
            (element_dtype, Arc::clone(&self.elements)),
            (offsets_dtype, Arc::clone(&self.offsets)),
        ];
        if is_nullable {
            child_specs.push((
                DType::Bool(Nullability::NonNullable),
                Arc::clone(&self.validity),
            ));
        }

        let buffered_bytes = ctx.buffered_bytes_tracker().clone();
        let handle = session.handle();
        let children = child_specs
            .into_iter()
            .map(|(child_dtype, strategy)| {
                let writer = strategy.new_writer(
                    ctx.clone(),
                    Arc::clone(&segment_sink),
                    child_dtype,
                    session,
                )?;
                Ok(LayoutWriterActor::spawn(
                    writer,
                    buffered_bytes.clone(),
                    &handle,
                ))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Box::new(ListLayoutWriter {
            dtype,
            is_nullable,
            children,
            exec_ctx: session.create_execution_ctx(),
            element_base: 0,
            first: true,
            saw_chunk: false,
        }))
    }
}

struct ListLayoutWriter {
    dtype: DType,
    is_nullable: bool,
    children: Vec<LayoutWriterActor>,
    exec_ctx: ExecutionCtx,
    element_base: u64,
    first: bool,
    saw_chunk: bool,
}

#[async_trait]
impl LayoutWriter for ListLayoutWriter {
    async fn write(&mut self, sequence_id: SequenceId, array: ArrayRef) -> VortexResult<()> {
        self.saw_chunk = true;
        let ListDataParts {
            elements,
            offsets,
            validity,
            ..
        } = canonicalize_to_list_parts(array, &mut self.exec_ctx)?;
        let n_elements = elements.len() as u64;
        let row_count = offsets.len().saturating_sub(1);
        let offsets = global_offsets(offsets, self.element_base, self.first, &mut self.exec_ctx)?;
        self.element_base += n_elements;
        self.first = false;

        let mut columns = vec![elements, offsets];
        if self.is_nullable {
            columns.push(
                validity
                    .execute_mask(row_count, &mut self.exec_ctx)?
                    .into_array(),
            );
        }

        let mut sequence = sequence_id.descend();
        let child_sequences = (0..self.children.len())
            .map(|_| sequence.advance())
            .collect::<Vec<_>>();
        try_join_all(
            self.children
                .iter_mut()
                .zip(columns)
                .zip(child_sequences)
                .map(|((writer, column), sequence_id)| writer.write(sequence_id, column)),
        )
        .await?;
        Ok(())
    }

    async fn finish(&mut self, sequence_id: SequenceId) -> VortexResult<()> {
        if !self.saw_chunk {
            vortex_bail!("ListLayoutStrategy needs at least one chunk");
        }
        let mut sequence = sequence_id.descend();
        let child_sequences = (0..self.children.len())
            .map(|_| sequence.advance())
            .collect::<Vec<_>>();
        try_join_all(
            self.children
                .iter_mut()
                .zip(child_sequences)
                .map(|(child, sequence_id)| child.finish(sequence_id)),
        )
        .await?;
        Ok(())
    }

    async fn close(self: Box<Self>) -> VortexResult<LayoutRef> {
        let Self {
            dtype,
            is_nullable,
            children,
            ..
        } = *self;
        let mut child_layouts = Vec::with_capacity(children.len());
        for mut writer in children {
            child_layouts.push(writer.take_layout()?);
        }
        let mut layouts = child_layouts.into_iter();
        let elements = layouts.next().vortex_expect("elements layout present");
        let offsets = layouts.next().vortex_expect("offsets layout present");
        let validity = is_nullable.then(|| layouts.next().vortex_expect("validity layout present"));
        Ok(ListLayout::new(dtype, elements, offsets, validity).into_layout())
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

/// Rebase a chunk's local `offsets` into global `u64` positions for the whole-column `offsets`
/// child. Each chunk's offsets are shifted by `element_base` (the number of elements already
/// emitted) so they index into the concatenated `elements`. The duplicated boundary offset is
/// dropped on every chunk after the first, so the concatenation of all chunks' contributions is a
/// single monotonic `[0, .., total_elements]` array of length `row_count + 1`.
fn global_offsets(
    offsets: ArrayRef,
    element_base: u64,
    first: bool,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let widened = offsets.cast(DType::Primitive(PType::U64, Nullability::NonNullable))?;
    let based = if element_base == 0 {
        widened
    } else {
        let base = ConstantArray::new(element_base, widened.len()).into_array();
        widened.binary(base, Operator::Add)?
    };
    let based = if first {
        based
    } else {
        based.slice(1..based.len())?
    };
    // Materialize so the child sub-stream carries a concrete array rather than a lazy expression.
    Ok(based.execute::<PrimitiveArray>(exec_ctx)?.into_array())
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
    use futures::StreamExt;
    use futures::stream;
    use vortex_array::ArrayContext;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;

    use super::*;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::table::TableStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::sequence::SequentialStreamAdapter;
    use crate::sequence::SequentialStreamExt;
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
        └── offsets: vortex.flat, dtype: u64, segment: 1
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
        ├── offsets: vortex.flat, dtype: u64, segment: 1
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
        │   ├── a: vortex.flat, dtype: i32, segment: 0
        │   └── b: vortex.flat, dtype: i32, segment: 1
        └── offsets: vortex.flat, dtype: u64, segment: 2
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
        │   ├── elements: vortex.flat, dtype: i32, segment: 0
        │   └── offsets: vortex.flat, dtype: u64, segment: 1
        └── offsets: vortex.flat, dtype: u64, segment: 2
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
        │   │   ├── elements: vortex.flat, dtype: i32, segment: 0
        │   │   └── offsets: vortex.flat, dtype: u64, segment: 1
        │   └── offsets: vortex.flat, dtype: u64, segment: 2
        └── offsets: vortex.flat, dtype: u64, segment: 3
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
        │   └── offsets: vortex.flat, dtype: u64, segment: 1
        └── [1]: vortex.list, dtype: list(i32), children: 2
            ├── elements: vortex.flat, dtype: i32, segment: 2
            └── offsets: vortex.flat, dtype: u64, segment: 3
        ");
        Ok(())
    }
}
