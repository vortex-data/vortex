// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_channel::Receiver;
use async_channel::Sender;
use async_channel::bounded;
use async_trait::async_trait;
use futures::StreamExt;
use futures::future::join;
use futures::future::try_join_all;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::demand::list_elements_demand;
use vortex_array::aggregate_fn::demand::valid_demand;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::List;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListView;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::arrays::list::ListDataParts;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::matcher::Matcher;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::list::ListLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialChunk;
use crate::sequence::SequentialStream;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;
use crate::sequence::fully_demanded;

/// Item carried on each child sub-stream: a sequenced, materialized chunk.
type ChildChunk = VortexResult<SequentialChunk>;

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

        // Empty nested element streams are valid below empty or all-null lists.
        // As in the flat writer, no input chunks require no child segments.
        let first_chunk = match stream.next().await {
            Some(chunk) => chunk?,
            None => {
                return Ok(ChunkedLayout::new(
                    0,
                    dtype,
                    OwnedLayoutChildren::layout_children(vec![]),
                )
                .into_layout());
            }
        };

        let is_nullable = dtype.is_nullable();
        let element_dtype = dtype
            .as_list_element_opt()
            .vortex_expect("DType is List")
            .as_ref()
            .clone();
        // Global (whole-column) offsets are cumulative and may exceed the input offset width,
        // so definsively widen.
        let offsets_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);

        // One bounded sub-stream per child: elements, offsets, and (when nullable) validity.
        let (elements_tx, elements_rx) = bounded::<ChildChunk>(1);
        let (offsets_tx, offsets_rx) = bounded::<ChildChunk>(1);
        let (validity_tx, validity_rx) = if is_nullable {
            let (tx, rx) = bounded::<ChildChunk>(1);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        // The elements are normalised to those of the lists its ancestors demand and that are
        // valid, so every element written belongs to the logical elements column, and every row
        // of every child is demanded.
        // Transpose the list column into its child sub-streams and rebase offsets to global
        // positions. Kept joined with the child writers below so producer errors surface rather
        // than being hidden as an early channel close.
        let fanout_fut = transpose_list_column(
            first_chunk,
            stream,
            session.clone(),
            elements_tx,
            offsets_tx,
            validity_tx,
        );

        // Spawn a writer per child sub-stream, concurrently.
        let handle = session.handle();
        let mut child_specs: Vec<(DType, Arc<dyn LayoutStrategy>, Receiver<ChildChunk>)> = vec![
            (element_dtype, Arc::clone(&self.elements), elements_rx),
            (offsets_dtype, Arc::clone(&self.offsets), offsets_rx),
        ];
        if let Some(validity_rx) = validity_rx {
            child_specs.push((
                DType::Bool(Nullability::NonNullable),
                Arc::clone(&self.validity),
                validity_rx,
            ));
        }

        let layout_futures: Vec<_> = child_specs
            .into_iter()
            .map(|(child_dtype, strategy, rx)| {
                let child_stream = SequentialStreamAdapter::new(child_dtype, rx.boxed()).sendable();
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

        // A child writer that fails drops its stream, which fails the transpose too. Report the
        // child writer's error, which is the cause, ahead of the transpose's.
        let (fanout, layouts) = join(fanout_fut, try_join_all(layout_futures)).await;
        let layouts = layouts?;
        fanout?;
        let mut layouts = layouts.into_iter();
        let elements_layout = layouts.next().vortex_expect("elements layout present");
        let offsets_layout = layouts.next().vortex_expect("offsets layout present");
        let validity_layout =
            is_nullable.then(|| layouts.next().vortex_expect("validity layout present"));

        Ok(ListLayout::new(dtype, elements_layout, offsets_layout, validity_layout).into_layout())
    }
}

/// Transpose a list column into its `elements`, `offsets`, and (when present) `validity` child
/// sub-streams, rebasing each chunk's local `offsets` to global `u64` positions so the single
/// `offsets` child indexes into the concatenated `elements` child.
///
/// `first_chunk` establishes nonempty input; `stream` contains the remaining chunks.
/// `validity_tx` is `Some` exactly when the list is nullable. Errors surface to the caller, which
/// joins this against the child writers, rather than being hidden as an early channel close.
async fn transpose_list_column(
    first_chunk: SequentialChunk,
    stream: SendableSequentialStream,
    session: VortexSession,
    elements_tx: Sender<ChildChunk>,
    offsets_tx: Sender<ChildChunk>,
    validity_tx: Option<Sender<ChildChunk>>,
) -> VortexResult<()> {
    let mut exec_ctx = session.create_execution_ctx();
    let mut element_base: u64 = 0;
    let mut first = true;
    let mut chunks = futures::stream::iter([Ok(first_chunk)]).chain(stream);
    while let Some(chunk) = chunks.next().await {
        let (sequence_id, array, demand) = chunk?;
        let mut sp = sequence_id.descend();
        let list = canonicalize_to_list(array, &mut exec_ctx)?;
        let list = drop_absent_elements(list, &demand, &mut exec_ctx)?;
        let ListDataParts {
            elements,
            offsets,
            validity,
            ..
        } = list.into_data_parts();
        let n_elements = elements.len() as u64;
        let row_count = offsets.len().saturating_sub(1);
        let offsets = global_offsets(offsets, element_base, first, &mut exec_ctx)?;
        element_base += n_elements;
        first = false;

        if elements_tx
            .send(Ok(fully_demanded(sp.advance(), elements)))
            .await
            .is_err()
            || offsets_tx
                .send(Ok(fully_demanded(sp.advance(), offsets)))
                .await
                .is_err()
        {
            vortex_bail!("list child writer finished before all chunks were sent");
        }
        if let Some(validity_tx) = &validity_tx {
            let validity = validity
                .execute_mask(row_count, &mut exec_ctx)?
                .into_array();
            if validity_tx
                .send(Ok(fully_demanded(sp.advance(), validity)))
                .await
                .is_err()
            {
                vortex_bail!("list validity writer finished before all chunks were sent");
            }
        }
    }
    Ok(())
}

/// Drop the elements that do not belong to the logical elements column: those of null lists, of
/// lists `demand` does not select, and elements no list references. Dropped lists become empty.
fn drop_absent_elements(
    list: ListArray,
    demand: &Mask,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ListArray> {
    let validity = list.validity()?;
    let lists = valid_demand(&validity, demand, exec_ctx)?;
    let elements = list_elements_demand(&list, &lists, exec_ctx)?;
    if elements.all_true() {
        return Ok(list);
    }

    let offsets = list
        .offsets()
        .clone()
        .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
        .execute::<PrimitiveArray>(exec_ctx)?;
    let mut kept = BufferMut::<u64>::with_capacity(offsets.len());
    kept.push(0);
    let mut end = 0u64;
    for (row, bounds) in offsets.as_slice::<u64>().windows(2).enumerate() {
        if lists.value(row) {
            end += bounds[1] - bounds[0];
        }
        kept.push(end);
    }
    // Writers serialize what they receive, so the filter is executed rather than left lazy.
    let elements = list
        .elements()
        .filter(elements)?
        .execute::<Canonical>(exec_ctx)?
        .into_array();
    ListArray::try_new(elements, kept.freeze().into_array(), validity, exec_ctx)
}

/// Canonicalize a list-dtype array into a [`ListArray`].
fn canonicalize_to_list(array: ArrayRef, exec_ctx: &mut ExecutionCtx) -> VortexResult<ListArray> {
    let canonical = array.execute_until::<AnyList>(exec_ctx)?;
    if let Some(list) = canonical.as_opt::<List>() {
        Ok(list.into_owned())
    } else if let Some(view) = canonical.as_opt::<ListView>() {
        list_from_list_view(view.into_owned(), exec_ctx)
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
    use std::num::NonZeroUsize;
    use std::sync::LazyLock;

    use futures::stream;
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::fns::null_count::NullCount;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSession;
    use vortex_session::VortexSession;

    use super::*;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::table::TableStrategy;
    use crate::layouts::zoned::Zoned;
    use crate::layouts::zoned::writer::ZonedLayoutOptions;
    use crate::layouts::zoned::writer::ZonedStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::session::LayoutSession;
    use crate::test::FailingStrategy;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);
    use crate::test::new_session;
    use crate::test::read_zone_stats;

    fn layout_test_session() -> VortexSession {
        VortexSession::empty()
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
        let mut ctx = SESSION.create_execution_ctx();
        ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5].into_array(),
            buffer![0u32, 2, 5, 5].into_array(),
            validity,
            &mut ctx,
        )
        .unwrap()
        .into_array()
    }

    fn zoned_leaf() -> Arc<dyn LayoutStrategy> {
        Arc::new(ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: NonZeroUsize::new(3).vortex_expect("non zero"),
                ..Default::default()
            },
        ))
    }

    /// Write `array` through `strategy` on a single-threaded runtime, returning the layout and the
    /// segments it was written to.
    fn write_blocking<S: LayoutStrategy>(
        strategy: S,
        array: ArrayRef,
    ) -> VortexResult<(LayoutRef, Arc<TestSegments>)> {
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        let segments = Arc::new(TestSegments::default());
        let segments2 = Arc::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ArrayContext::empty().into(),
                    segments2,
                    stream,
                    eof,
                    &session,
                )
                .await
        })?;
        Ok((layout, segments))
    }

    /// The per-zone minimum and maximum of a zoned elements layout.
    fn zone_bounds(
        elements: &LayoutRef,
        segments: &Arc<TestSegments>,
    ) -> VortexResult<(ArrayRef, ArrayRef)> {
        assert!(elements.is::<Zoned>(), "{elements} has no zone map");
        Ok((
            read_zone_stats(elements, segments, &Min.bind(Default::default()))?,
            read_zone_stats(elements, segments, &Max.bind(Default::default()))?,
        ))
    }

    /// The elements of a null list are dropped, so the elements' zone map covers only the
    /// elements of valid lists, and does not count the null list's null element.
    #[test]
    fn elements_of_null_lists_are_dropped() -> VortexResult<()> {
        let list = ListArray::try_new(
            PrimitiveArray::from_option_iter([Some(1i32), Some(100), None, Some(2), Some(3)])
                .into_array(),
            buffer![0u32, 1, 3, 5].into_array(),
            Validity::from_iter([true, false, true]),
            &mut SESSION.create_execution_ctx(),
        )?
        .into_array();

        let (layout, segments) = write_blocking(
            ListLayoutStrategy::default().with_elements(zoned_leaf()),
            list,
        )?;

        let elements = &layout.children()?[0];
        assert_eq!(elements.row_count(), 3);
        let (min, max) = zone_bounds(elements, &segments)?;
        let null_count = read_zone_stats(elements, &segments, &NullCount.bind(EmptyOptions))?;
        let mut ctx = new_session().create_execution_ctx();
        assert_arrays_eq!(
            min,
            PrimitiveArray::from_option_iter([Some(1i32)]),
            &mut ctx
        );
        assert_arrays_eq!(
            max,
            PrimitiveArray::from_option_iter([Some(3i32)]),
            &mut ctx
        );
        assert_arrays_eq!(
            null_count,
            PrimitiveArray::from_option_iter([Some(0u64)]),
            &mut ctx
        );
        Ok(())
    }

    /// Lists that are already normalised are written unchanged.
    #[test]
    fn normalised_lists_keep_their_elements() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let list = ListArray::try_new(
            buffer![1i32, 2, 3].into_array(),
            buffer![0u32, 1, 1, 3].into_array(),
            Validity::from_iter([true, false, true]),
            &mut ctx,
        )?;
        let normalised = drop_absent_elements(list.clone(), &Mask::new_true(3), &mut ctx)?;
        assert_arrays_eq!(normalised, list, &mut ctx);
        assert_eq!(normalised.elements().len(), 3);
        Ok(())
    }

    /// A list zoned in its outer-row space under a nullable struct hands the struct's demand on
    /// to the list writer, which keeps only the elements of the struct's valid rows.
    #[test]
    fn list_under_nullable_struct_keeps_elements_of_valid_rows() -> VortexResult<()> {
        let list = ListArray::try_new(
            buffer![1i32, 60, 50, 2].into_array(),
            buffer![0u32, 1, 3, 4].into_array(),
            Validity::NonNullable,
            &mut SESSION.create_execution_ctx(),
        )?;
        let array = StructArray::try_from_iter_with_validity(
            [("l", list.into_array())],
            Validity::from_iter([true, false, true]),
        )?
        .into_array();

        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        let strategy = TableStrategy::new(Arc::clone(&flat), Arc::clone(&flat))
            .with_list_layout_factory(move |list_layout: ListLayoutStrategy| {
                Arc::new(ZonedStrategy::new(
                    list_layout.with_elements(zoned_leaf()),
                    Arc::clone(&flat),
                    ZonedLayoutOptions {
                        block_size: NonZeroUsize::new(3).vortex_expect("non zero"),
                        ..Default::default()
                    },
                )) as Arc<dyn LayoutStrategy>
            });
        let (layout, segments) = write_blocking(strategy, array)?;

        // Skip the struct's validity child, then the list's zone map to reach the list layout.
        let list_layout = Arc::clone(&layout.children()?[1]);
        let list_layout = if list_layout.is::<Zoned>() {
            Arc::clone(&list_layout.children()?[0])
        } else {
            list_layout
        };
        // The list in the struct's null row is written empty, without its 60 and 50.
        assert_eq!(list_layout.children()?[0].row_count(), 2);
        let (min, max) = zone_bounds(&list_layout.children()?[0], &segments)?;
        let mut ctx = new_session().create_execution_ctx();
        assert_arrays_eq!(
            min,
            PrimitiveArray::from_option_iter([Some(1i32)]),
            &mut ctx
        );
        assert_arrays_eq!(
            max,
            PrimitiveArray::from_option_iter([Some(2i32)]),
            &mut ctx
        );
        Ok(())
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

    #[rstest]
    #[case::non_nullable(false)]
    #[case::nullable(true)]
    #[tokio::test]
    async fn empty_stream_preserves_dtype_and_row_count(
        #[case] nullable: bool,
    ) -> VortexResult<()> {
        let segments = Arc::new(TestSegments::default());
        let (_, eof) = SequenceId::root().split();
        let empty = stream::empty::<VortexResult<SequentialChunk>>().boxed();
        let stream = SequentialStreamAdapter::new(i32_list_dtype(nullable), empty).sendable();
        let session = layout_test_session();

        let layout = flat_list_strategy()
            .write_stream(
                ArrayContext::empty().into(),
                segments,
                stream,
                eof,
                &session,
            )
            .await?;
        assert_eq!(layout.dtype(), &i32_list_dtype(nullable));
        assert_eq!(layout.row_count(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn list_of_struct_tree() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
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
            &mut ctx,
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
        └── offsets: vortex.flat, dtype: u64, segment: 0
        ");
        Ok(())
    }

    #[tokio::test]
    async fn list_of_list_tree() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let inner_list = ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            buffer![0u32, 2, 5, 5, 6].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let list = ListArray::try_new(
            inner_list,
            buffer![0u32, 2, 4].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();

        let writer =
            ListLayoutStrategy::default().with_elements(Arc::new(ListLayoutStrategy::default()));
        let layout = write(&writer, list).await?;
        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.list, dtype: list(list(i32)), children: 2
        ├── elements: vortex.list, dtype: list(i32), children: 2
        │   ├── elements: vortex.flat, dtype: i32, segment: 1
        │   └── offsets: vortex.flat, dtype: u64, segment: 2
        └── offsets: vortex.flat, dtype: u64, segment: 0
        ");
        Ok(())
    }

    #[tokio::test]
    async fn list_of_list_of_list_tree() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let innermost = ListArray::try_new(
            buffer![1i32, 2, 3, 4].into_array(),
            buffer![0u32, 2, 4].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let middle = ListArray::try_new(
            innermost,
            buffer![0u32, 2].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let outer = ListArray::try_new(
            middle,
            buffer![0u32, 1].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
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
        │   │   └── offsets: vortex.flat, dtype: u64, segment: 3
        │   └── offsets: vortex.flat, dtype: u64, segment: 1
        └── offsets: vortex.flat, dtype: u64, segment: 0
        ");
        Ok(())
    }

    #[tokio::test]
    async fn chunked_list_input_with_chunked_strategy_succeeds() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let chunk0 = ListArray::try_new(
            buffer![1i32, 2, 3].into_array(),
            buffer![0u32, 2, 3].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )
        .unwrap()
        .into_array();
        let chunk1 = ListArray::try_new(
            buffer![4i32, 5, 6, 7].into_array(),
            buffer![0u32, 1, 4].into_array(),
            Validity::NonNullable,
            &mut ctx,
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

    /// A child writer's own error is reported, not the transpose's failure to send it more chunks.
    #[tokio::test]
    async fn reports_child_writer_error() -> VortexResult<()> {
        let strategy = flat_list_strategy().with_elements(Arc::new(FailingStrategy));
        let chunks = (0..4)
            .map(|_| create_basic_list(Validity::NonNullable))
            .collect::<Vec<_>>();
        let dtype = chunks[0].dtype().clone();
        let array = ChunkedArray::try_new(chunks, dtype)?.into_array();
        let err = write(&strategy, array).await.unwrap_err();
        assert!(err.to_string().contains(FailingStrategy::MESSAGE), "{err}");
        Ok(())
    }
}
