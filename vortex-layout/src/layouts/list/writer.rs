// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroU64;
use std::sync::Arc;

use async_channel::Receiver;
use async_channel::Sender;
use async_channel::bounded;
use async_trait::async_trait;
use futures::StreamExt;
use futures::future::join;
use futures::future::try_join_all;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::List;
use vortex_array::arrays::ListView;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::list::ListArrayExt;
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
use crate::LayoutWriterContext;
use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::list::ElementSamples;
use crate::layouts::list::ListLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStream;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Item carried on each child sub-stream: a sequenced, materialized chunk.
type ChildChunk = VortexResult<(SequenceId, ArrayRef)>;

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
    element_stride: Option<NonZeroU64>,
}

/// The most [`ElementSamples`] a list layout records: past it, every other sample is dropped and
/// the stride doubles, so a long column's samples stay a small part of the footer.
const MAX_ELEMENT_SAMPLES: usize = 16_384;

/// Records [`ElementSamples`] as the offsets of a list column stream past.
struct SampleRecorder {
    stride: u64,
    /// The element row of the next sample.
    next: u64,
    lists: Vec<u64>,
}

impl SampleRecorder {
    fn new(stride: NonZeroU64) -> Self {
        Self {
            stride: stride.get(),
            next: 0,
            lists: Vec::new(),
        }
    }

    /// Samples the elements of a chunk whose global `offsets` start at list `list_base`.
    fn record(&mut self, offsets: &[u64], list_base: u64) {
        let Some(&end) = offsets.last() else {
            return;
        };
        while self.next < end {
            // The last list starting at or before the sampled row holds it.
            let starts = offsets.partition_point(|&offset| offset <= self.next) as u64;
            self.lists.push(list_base + starts - 1);
            self.next += self.stride;
        }
    }

    fn finish(mut self, list_rows: u64, element_rows: u64) -> VortexResult<ElementSamples> {
        while self.lists.len() > MAX_ELEMENT_SAMPLES {
            self.lists = self.lists.iter().step_by(2).copied().collect();
            self.stride *= 2;
        }
        ElementSamples::try_new(self.stride, self.lists, list_rows, element_rows)
    }
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
            element_stride: None,
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

    /// Records the list holding every `stride`-th element row, so a reader maps lists to their
    /// elements without reading the offsets. See [`ElementSamples`].
    pub fn with_element_samples(mut self, stride: NonZeroU64) -> Self {
        self.element_stride = Some(stride);
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

        // Transpose the list column into its child sub-streams and rebase offsets to global
        // positions. Kept joined with the child writers below so producer errors surface rather
        // than being hidden as an early channel close.
        let fanout_fut = transpose_list_column(
            first_chunk,
            self.element_stride.map(SampleRecorder::new),
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
        let samples = fanout?;
        let mut layouts = layouts.into_iter();
        let elements_layout = layouts.next().vortex_expect("elements layout present");
        let offsets_layout = layouts.next().vortex_expect("offsets layout present");
        let validity_layout =
            is_nullable.then(|| layouts.next().vortex_expect("validity layout present"));

        let element_samples = samples
            .map(|samples| {
                samples.finish(
                    offsets_layout.row_count().saturating_sub(1),
                    elements_layout.row_count(),
                )
            })
            .transpose()?;
        Ok(ListLayout::new_with_element_samples(
            dtype,
            elements_layout,
            offsets_layout,
            validity_layout,
            element_samples,
        )
        .into_layout())
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
    first_chunk: (SequenceId, ArrayRef),
    mut samples: Option<SampleRecorder>,
    stream: SendableSequentialStream,
    session: VortexSession,
    elements_tx: Sender<ChildChunk>,
    offsets_tx: Sender<ChildChunk>,
    validity_tx: Option<Sender<ChildChunk>>,
) -> VortexResult<Option<SampleRecorder>> {
    let mut exec_ctx = session.create_execution_ctx();
    let mut element_base: u64 = 0;
    let mut list_base: u64 = 0;
    let mut first = true;
    let mut chunks = futures::stream::iter([Ok(first_chunk)]).chain(stream);
    while let Some(chunk) = chunks.next().await {
        let (sequence_id, array) = chunk?;
        let mut sp = sequence_id.descend();
        let ListDataParts {
            elements,
            offsets,
            validity,
            ..
        } = canonicalize_to_list_parts(array, &mut exec_ctx)?;
        let n_elements = elements.len() as u64;
        let row_count = offsets.len().saturating_sub(1);
        let offsets = global_offsets(offsets, element_base, &mut exec_ctx)?;
        if let Some(samples) = &mut samples {
            samples.record(offsets.as_slice::<u64>(), list_base);
        }
        // The single offsets child holds each chunk's end once: as the next chunk's start.
        let offsets = if first {
            offsets.into_array()
        } else {
            offsets.into_array().slice(1..row_count + 1)?
        };
        element_base += n_elements;
        list_base += row_count as u64;
        first = false;

        if elements_tx
            .send(Ok((sp.advance(), elements)))
            .await
            .is_err()
            || offsets_tx.send(Ok((sp.advance(), offsets))).await.is_err()
        {
            vortex_bail!("list child writer finished before all chunks were sent");
        }
        if let Some(validity_tx) = &validity_tx {
            let validity = validity
                .execute_mask(row_count, &mut exec_ctx)?
                .into_array();
            if validity_tx
                .send(Ok((sp.advance(), validity)))
                .await
                .is_err()
            {
                vortex_bail!("list validity writer finished before all chunks were sent");
            }
        }
    }
    Ok(samples)
}

/// Canonicalize a list-dtype array into [`ListDataParts`].
fn canonicalize_to_list_parts(
    array: ArrayRef,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ListDataParts> {
    let canonical = array.execute_until::<AnyList>(exec_ctx)?;
    if let Some(list) = canonical.as_opt::<List>() {
        // A sliced list keeps its parent's whole `elements`, but `global_offsets` assumes each
        // chunk's offsets span exactly `[0, elements.len()]`.
        Ok(list.reset_offsets(false, exec_ctx)?.into_data_parts())
    } else if let Some(view) = canonical.as_opt::<ListView>() {
        Ok(list_from_list_view(view.into_owned(), exec_ctx)?.into_data_parts())
    } else {
        unreachable!("AnyList matcher guarantees List or ListView")
    }
}

/// Rebase a chunk's local `offsets` into global `u64` positions for the whole-column `offsets`
/// child. Each chunk's offsets are shifted by `element_base` (the number of elements already
/// emitted) so they index into the concatenated `elements`. The caller drops the duplicated
/// boundary offset of every chunk after the first, so the concatenation of all chunks'
/// contributions is a single monotonic `[0, .., total_elements]` array of length
/// `row_count + 1`.
fn global_offsets(
    offsets: ArrayRef,
    element_base: u64,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let widened = offsets.cast(DType::Primitive(PType::U64, Nullability::NonNullable))?;
    let based = if element_base == 0 {
        widened
    } else {
        let base = ConstantArray::new(element_base, widened.len()).into_array();
        widened.binary(base, Operator::Add)?
    };
    // Materialize so the child sub-stream carries a concrete array rather than a lazy expression.
    based.execute::<PrimitiveArray>(exec_ctx)
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
    use std::sync::LazyLock;

    use futures::stream;
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_io::session::RuntimeSession;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::LayoutBuildContext;
    use crate::LayoutDeserialization;
    use crate::LayoutPlugin;
    use crate::layout_children;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::list::ListLayoutEncoding;
    use crate::layouts::table::TableStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::session::LayoutSession;
    use crate::test::FailingStrategy;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

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
        let empty = stream::empty::<VortexResult<(SequenceId, ArrayRef)>>().boxed();
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

    /// A list strategy whose children keep every chunk, so a multi-chunk stream reaches the
    /// transpose, recording samples every `stride` elements.
    fn sampled_list_strategy(stride: u64) -> ListLayoutStrategy {
        let chunked: Arc<dyn LayoutStrategy> =
            Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()));
        ListLayoutStrategy::default()
            .with_elements(Arc::clone(&chunked))
            .with_offsets(Arc::clone(&chunked))
            .with_validity(chunked)
            .with_element_samples(NonZeroU64::new(stride).vortex_expect("stride is nonzero"))
    }

    /// A chunk of lists of `lengths`, its elements counting up from zero.
    fn list_chunk(lengths: &[u32]) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let offsets = std::iter::once(0)
            .chain(lengths.iter().scan(0u32, |end, &len| {
                *end += len;
                Some(*end)
            }))
            .collect::<Buffer<u32>>();
        let elements = (0..i32::try_from(offsets[offsets.len() - 1])?).collect::<Buffer<i32>>();
        Ok(ListArray::try_new(
            elements.into_array(),
            offsets.into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array())
    }

    /// The list holding every `stride`-th element row of lists of `lengths`, found by walking
    /// the lists.
    fn holding_lists(lengths: &[u32], stride: usize) -> Vec<u64> {
        let mut holders = Vec::new();
        for (list, &len) in lengths.iter().enumerate() {
            holders.extend(std::iter::repeat_n(list as u64, len as usize));
        }
        holders.into_iter().step_by(stride).collect()
    }

    /// Samples follow the global element rows across chunks, through empty lists at the chunk
    /// boundaries.
    #[rstest]
    #[tokio::test]
    async fn element_samples_hold_every_stride_th_element(
        #[values(1, 2, 3, 5)] stride: u64,
    ) -> VortexResult<()> {
        let chunks: [&[u32]; 3] = [&[3, 0, 2], &[0, 0, 4], &[1, 5, 0]];
        let array = ChunkedArray::try_new(
            chunks
                .iter()
                .map(|lengths| list_chunk(lengths))
                .collect::<VortexResult<Vec<_>>>()?,
            i32_list_dtype(false),
        )?
        .into_array();

        let layout = write(&sampled_list_strategy(stride), array).await?;
        let samples = layout
            .as_::<ListLayoutEncoding>()
            .element_samples()
            .vortex_expect("samples were recorded");
        assert_eq!(samples.stride(), stride);
        assert_eq!(
            samples.lists(),
            holding_lists(&chunks.concat(), usize::try_from(stride)?)
        );
        Ok(())
    }

    /// Past the cap, every other sample is dropped and the stride doubles until they fit.
    #[tokio::test]
    async fn element_samples_halve_past_the_cap() -> VortexResult<()> {
        let lengths = vec![2u32; 20_000];
        let layout = write(&sampled_list_strategy(1), list_chunk(&lengths)?).await?;
        let samples = layout
            .as_::<ListLayoutEncoding>()
            .element_samples()
            .vortex_expect("samples were recorded");
        assert_eq!(samples.stride(), 4);
        assert_eq!(samples.lists(), holding_lists(&lengths, 4));
        Ok(())
    }

    /// The samples survive the layout's metadata, which checks them against its children.
    #[tokio::test]
    async fn element_samples_round_trip_through_metadata() -> VortexResult<()> {
        let lengths = [3u32, 0, 2, 4, 1];
        let layout = write(&sampled_list_strategy(2), list_chunk(&lengths)?).await?;
        let list = layout.as_::<ListLayoutEncoding>();
        let children = layout_children(vec![list.elements()?, list.offsets()?]);
        let read_ctx = ReadContext::new([]);
        let build_ctx = LayoutBuildContext {
            session: &crate::test::SESSION,
            array_read_ctx: &read_ctx,
        };
        let metadata = layout.metadata();
        let read = LayoutPlugin::deserialize(
            &ListLayoutEncoding,
            LayoutDeserialization {
                serialized_id: LayoutPlugin::id(&ListLayoutEncoding),
                dtype: layout.dtype(),
                row_count: layout.row_count(),
                metadata: &metadata,
                segment_ids: vec![],
                children: children.as_ref(),
            },
            &build_ctx,
        )?;
        assert_eq!(
            read.as_::<ListLayoutEncoding>().element_samples(),
            list.element_samples()
        );
        Ok(())
    }
}
