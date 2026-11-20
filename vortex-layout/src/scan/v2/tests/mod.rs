// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod lists;

use std::future::poll_fn;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_schema::Field;
use async_trait::async_trait;
use futures::Stream;
use futures::TryStreamExt;
use futures::stream;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::cast;
use vortex_array::expr::checked_add;
use vortex_array::expr::dynamic;
use vortex_array::expr::eq;
use vortex_array::expr::fill_null;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::gt_eq;
use vortex_array::expr::is_nan;
use vortex_array::expr::is_null;
use vortex_array::expr::list_contains;
use vortex_array::expr::list_length;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::not;
use vortex_array::expr::not_eq;
use vortex_array::expr::or;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowSessionExt;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_scan::execution::SegmentLocation;
use vortex_scan::planning::driver::Batch;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use super::ChunkPlanner;
use super::PreparedScan;
use super::PrunePlanner;
use super::ROW_BLOCK_SIZE;
use super::ScanBuilder;
use super::ScanFile;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::VTable as _;
use crate::layouts::chunked::Chunked;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::flat::Flat;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::repartition::RepartitionStrategy;
use crate::layouts::repartition::RepartitionWriterOptions;
use crate::layouts::row_idx::row_idx;
use crate::layouts::table::TableStrategy;
use crate::layouts::zoned::LegacyStats;
use crate::layouts::zoned::Zoned;
use crate::layouts::zoned::writer::ZonedLayoutOptions;
use crate::layouts::zoned::writer::ZonedStrategy;
use crate::scan::arrow::to_record_batch;
use crate::scan::scan_builder::ScanBuilder as V1ScanBuilder;
use crate::segments::CountingSegmentSource;
use crate::segments::SegmentSinkRef;
use crate::segments::SegmentSource;
use crate::segments::TestSegments;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt as _;
use crate::test::new_session;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

const CHUNK_ROWS: usize = 1000;
const CHUNKS: usize = 4;

/// A table chunk of `CHUNK_ROWS` rows starting at row `start`: an integer, a nullable string and a
/// list column.
fn table_chunk(start: usize) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let rows = start..start + CHUNK_ROWS;
    let ints = Buffer::from_iter(
        rows.clone()
            .map(|row| i32::try_from(row).unwrap_or(i32::MAX)),
    );
    let strings = VarBinArray::from_iter(
        rows.map(|row| (row % 3 != 0).then(|| format!("row {row}"))),
        DType::Utf8(true.into()),
    );
    let elements = Buffer::from_iter((0..2 * CHUNK_ROWS).map(|i| i64::try_from(i).unwrap_or(0)));
    let offsets = Buffer::from_iter((0..=CHUNK_ROWS).map(|i| u32::try_from(2 * i).unwrap_or(0)));
    let lists = ListArray::try_new(
        elements.into_array(),
        offsets.into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?;
    Ok(StructArray::from_fields(&[
        ("ints", ints.into_array()),
        ("strings", strings.into_array()),
        ("lists", lists.into_array()),
    ])?
    .into_array())
}

/// Writes `CHUNKS` table chunks with `strategy`.
async fn write(
    session: &VortexSession,
    strategy: &dyn LayoutStrategy,
) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef)> {
    let chunks = (0..CHUNKS)
        .map(|chunk| table_chunk(chunk * CHUNK_ROWS))
        .collect::<VortexResult<Vec<_>>>()?;
    write_chunks(session, strategy, chunks).await
}

async fn write_chunks(
    session: &VortexSession,
    strategy: &dyn LayoutStrategy,
    chunks: Vec<ArrayRef>,
) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef)> {
    let dtype = chunks
        .first()
        .ok_or_else(|| vortex_err!("test layout requires at least one chunk"))?
        .dtype()
        .clone();
    let segments = Arc::new(TestSegments::default());
    let (mut sequence_id, eof) = SequenceId::root().split();
    let chunks = chunks
        .into_iter()
        .map(move |chunk| Ok((sequence_id.advance(), chunk)));
    let layout = strategy
        .write_stream(
            ArrayContext::empty().into(),
            Arc::<TestSegments>::clone(&segments),
            SequentialStreamAdapter::new(dtype, stream::iter(chunks)).sendable(),
            eof,
            session,
        )
        .await?;
    Ok((segments, layout))
}

fn chunked_flat() -> Arc<dyn LayoutStrategy> {
    Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()))
}

/// Every column is shredded into its own chunked leaves.
fn table() -> Arc<dyn LayoutStrategy> {
    Arc::new(TableStrategy::new(chunked_flat(), chunked_flat()))
}

/// A chunked leaf with one zone per chunk.
fn zoned_leaf() -> ZonedStrategy {
    ZonedStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        FlatLayoutStrategy::default(),
        ZonedLayoutOptions {
            block_size: NonZeroUsize::new(CHUNK_ROWS).unwrap_or(NonZeroUsize::MIN),
            ..Default::default()
        },
    )
}

/// Every column is shredded into chunked leaves with zone statistics.
fn zoned_table() -> Arc<dyn LayoutStrategy> {
    Arc::new(TableStrategy::new(chunked_flat(), Arc::new(zoned_leaf())))
}

/// A [`zoned_leaf`] that first repartitions its input into chunks of `CHUNK_ROWS` rows, as the file
/// writer does. A list layout writes its elements in chunks as long as the lists they hold, and a
/// zoned leaf records one zone per chunk, so those zones would not match the zone length.
fn repartitioned_zoned_leaf() -> Arc<dyn LayoutStrategy> {
    Arc::new(RepartitionStrategy::new(
        zoned_leaf(),
        RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: CHUNK_ROWS,
            block_size_target: None,
            canonicalize: false,
        },
    ))
}

/// A [`ScanFile`] over test segments. They have no byte offsets, and the scan fetches them by id,
/// so each location is a placeholder whose offset is its id.
fn scan_file(segments: &Arc<dyn SegmentSource>, layout: &LayoutRef) -> VortexResult<ScanFile> {
    let mut count = 0;
    for layout in layout.depth_first_traversal() {
        for id in layout?.segment_ids() {
            count = count.max(*id as usize + 1);
        }
    }
    let locations = (0..count as u64)
        .map(|offset| SegmentLocation {
            offset,
            length: 0,
            alignment: Alignment::none(),
        })
        .collect();
    Ok(ScanFile {
        layout: Arc::clone(layout),
        locations,
        segments: Arc::clone(segments),
        cache: None,
    })
}

fn builder(
    session: &VortexSession,
    segments: &Arc<dyn SegmentSource>,
    layout: &LayoutRef,
) -> VortexResult<ScanBuilder> {
    Ok(ScanBuilder::new(
        session.clone(),
        scan_file(segments, layout)?,
    ))
}

/// Composes a scan's planners the way an engine does: the pruning plans first when there is a filter,
/// continuing into the chunks of the rows they keep.
fn compose(scan: &Arc<PreparedScan>, rows: Range<u64>) -> Box<dyn Planner> {
    let chunks = {
        let scan = Arc::clone(scan);
        let rows = rows.clone();
        move |kept: Option<Mask>| -> Box<dyn Planner> {
            let selection = scan.selection().clone();
            Box::new(ChunkPlanner::new(scan, rows, selection, kept))
        }
    };
    if scan.filter().is_some() {
        Box::new(PrunePlanner::new(Arc::clone(scan), rows, Box::new(chunks)))
    } else {
        chunks(None)
    }
}

/// Runs `rows` of a prepared scan on one planning run, every spawned chunk admitted to it, and
/// returns its batches in row order, each with the rows it stands for.
async fn run_batches(scan: &Arc<PreparedScan>, rows: Range<u64>) -> VortexResult<Vec<Batch>> {
    let io = scan.io();
    let mut run = Run::new();
    run.admit(
        compose(scan, rows.clone()),
        WorkScope {
            file_ordinal: 0,
            rows,
        },
        io.session(),
    );
    let mut batches = Vec::new();
    loop {
        match run.advance()? {
            Progress::Batch(batch) => batches.push(batch),
            Progress::Spawn(scope, chunk) => {
                run.admit(chunk, scope, io.session());
            }
            Progress::RootDone(_) if run.live_roots() > 0 => {}
            Progress::RootDone(_) | Progress::Idle => break,
            Progress::Waiting => poll_fn(|cx| run.poll_completion(cx)).await?,
        }
    }
    batches.sort_by_key(|batch| batch.scope.rows.start);
    Ok(batches)
}

/// Runs `rows` of a prepared scan and returns its arrays.
async fn run_rows(scan: &Arc<PreparedScan>, rows: Range<u64>) -> VortexResult<Vec<ArrayRef>> {
    Ok(run_batches(scan, rows)
        .await?
        .into_iter()
        .map(|batch| batch.array)
        .collect())
}

/// Prepares and runs a whole scan, returning its arrays.
async fn run_scan(builder: ScanBuilder) -> VortexResult<Vec<ArrayRef>> {
    let scan = Arc::new(builder.prepare()?);
    let rows = scan.rows().clone();
    run_rows(&scan, rows).await
}

async fn collect(
    dtype: DType,
    stream: impl Stream<Item = VortexResult<ArrayRef>>,
) -> VortexResult<ArrayRef> {
    let chunks: Vec<ArrayRef> = stream.try_collect().await?;
    Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
}

/// Prepares and runs a whole scan, returning its rows as one chunked array.
async fn collect_scan(dtype: DType, builder: ScanBuilder) -> VortexResult<ArrayRef> {
    Ok(ChunkedArray::try_new(run_scan(builder).await?, dtype)?.into_array())
}

/// Runs `builder` on both executors and checks the planner-driven one returns the same rows.
async fn assert_same_as_v1(
    session: &VortexSession,
    make: impl Fn() -> VortexResult<ScanBuilder>,
) -> VortexResult<()> {
    let builder = make()?;
    let dtype = builder.dtype().clone();
    let reader = builder.file.layout.new_reader(
        "".into(),
        Arc::clone(&builder.file.segments),
        session,
        &Default::default(),
    )?;
    let mut v1 = V1ScanBuilder::new(session.clone(), reader)
        .with_projection(builder.projection.clone())
        .with_some_filter(builder.filter.clone())
        .with_selection(builder.selection.clone())
        .with_row_offset(builder.row_offset);
    if let Some(rows) = builder.row_range.clone() {
        v1 = v1.with_row_range(rows);
    }
    let expected = collect(dtype.clone(), v1.into_stream()?).await?;
    let actual = collect_scan(dtype, builder).await?;
    assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Strategy {
    Flat,
    Table,
    ZonedTable,
}

#[rstest]
#[case::dense(-1)]
#[case::sparse(200_000)]
#[case::empty(300_000)]
#[tokio::test]
async fn large_filtered_chunks_match_v1(#[case] cutoff: i32) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let segments = Arc::new(TestSegments::default());
    let (mut sequence, eof) = SequenceId::root().split();
    let values = PrimitiveArray::from_iter(0i32..210_000).into_array();
    let dtype = values.dtype().clone();
    let layout = chunked_flat()
        .write_stream(
            ArrayContext::empty().into(),
            Arc::<TestSegments>::clone(&segments),
            SequentialStreamAdapter::new(
                dtype.clone(),
                stream::iter([Ok((sequence.advance(), values))]),
            )
            .sendable(),
            eof,
            &session,
        )
        .await?;
    let segments: Arc<dyn SegmentSource> = segments;
    assert_same_as_v1(&session, || {
        Ok(builder(&session, &segments, &layout)?
            .with_some_filter(Some(gt(root(), lit(cutoff)).bind(&dtype)?)))
    })
    .await
}

#[rstest]
#[case::range(and(
    gt(get_item("ints", root()), lit(995i32)),
    lt(get_item("ints", root()), lit(2050i32))
))]
#[case::nullable(eq(get_item("strings", root()), lit("row 1001")))]
#[case::nulls(is_null(get_item("strings", root())))]
#[case::no_matches(gt(get_item("ints", root()), lit(10_000i32)))]
#[case::row_index(gt(row_idx(), lit(1500u64)))]
#[tokio::test]
async fn filter_matches_v1(
    #[values(Strategy::Flat, Strategy::Table, Strategy::ZonedTable)] strategy: Strategy,
    #[case] filter: Expression,
    #[values(false, true)] sparse: bool,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, strategy.get().as_ref()).await?;
    assert_same_as_v1(&session, || {
        let mut scan = builder(&session, &segments, &layout)?
            .with_projection(
                pack([("strings", get_item("strings", root()))], false.into())
                    .bind(layout.dtype())?,
            )
            .with_some_filter(Some(filter.bind(layout.dtype())?))
            .with_row_range(900..3100)
            .with_row_offset(100);
        if sparse {
            scan = scan.with_row_indices(StrictSortedBuffer::try_new(Buffer::from_iter(
                (0..4000u64).step_by(7),
            ))?);
        }
        Ok(scan)
    })
    .await
}

impl Strategy {
    fn get(self) -> Arc<dyn LayoutStrategy> {
        match self {
            Strategy::Flat => chunked_flat(),
            Strategy::Table => table(),
            Strategy::ZonedTable => zoned_table(),
        }
    }
}

#[rstest]
#[case::whole(None, None)]
#[case::row_range(Some(250..3_100), None)]
#[case::empty_row_range(Some(10..10), None)]
#[case::row_indices(None, Some(&[0, 1, 999, 1000, 2500, 3999][..]))]
#[case::row_range_and_indices(Some(900..2600), Some(&[0, 999, 1000, 2500, 3999][..]))]
#[tokio::test]
async fn matches_v1(
    #[values(Strategy::Flat, Strategy::Table, Strategy::ZonedTable)] strategy: Strategy,
    #[case] row_range: Option<Range<u64>>,
    #[case] rows: Option<&'static [u64]>,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, strategy.get().as_ref()).await?;
    assert_same_as_v1(&session, || {
        let mut builder = builder(&session, &segments, &layout)?;
        if let Some(row_range) = row_range.clone() {
            builder = builder.with_row_range(row_range);
        }
        if let Some(rows) = rows {
            builder =
                builder.with_row_indices(StrictSortedBuffer::try_new(Buffer::copy_from(rows))?);
        }
        Ok(builder)
    })
    .await
}

#[rstest]
#[tokio::test]
async fn projection_matches_v1(
    #[values(Strategy::Flat, Strategy::Table, Strategy::ZonedTable)] strategy: Strategy,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, strategy.get().as_ref()).await?;
    let dtype = layout.dtype().clone();
    assert_same_as_v1(&session, || {
        let projection = pack(
            [
                ("strings", get_item("strings", root())),
                ("idx", row_idx()),
                ("big", gt(get_item("ints", root()), lit(1500i32))),
            ],
            false.into(),
        );
        Ok(builder(&session, &segments, &layout)?
            .with_projection(projection.bind(&dtype)?)
            .with_row_offset(100))
    })
    .await
}

#[tokio::test]
async fn arrow_stream_projects_across_chunks() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, table().as_ref()).await?;
    let projection =
        pack([("value", get_item("ints", root()))], false.into()).bind(layout.dtype())?;
    let scan = builder(&session, &segments, &layout)?
        .with_projection(projection)
        .with_row_range(990..1010);
    let schema = session.arrow().to_arrow_schema(scan.dtype())?;
    let field = Field::new_struct("", schema.fields().clone(), false);
    let mut ctx = session.create_execution_ctx();
    let batches = run_scan(scan)
        .await?
        .into_iter()
        .map(|chunk| to_record_batch(chunk, &field, &mut ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    let values: Vec<_> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_primitive::<Int32Type>()
                .values()
                .iter()
                .copied()
        })
        .collect();
    assert_eq!(values, (990_i32..1010).collect::<Vec<_>>());
    Ok(())
}

/// List layouts scan the same on both executors, over row ranges, row indices and filters, whose
/// rows of lists map to the elements the offsets of those lists give.
#[rstest]
#[case::whole(root(), None, None, None)]
#[case::row_range(get_item("lists", root()), None, Some(250..3_100), None)]
#[case::row_indices(
    get_item("lists", root()),
    None,
    None,
    Some(&[0, 1, 999, 1000, 2500, 3999][..])
)]
#[case::filter(
    get_item("lists", root()),
    Some(gt(get_item("ints", root()), lit(1500i32))),
    Some(900..3_100),
    None
)]
#[case::filter_on_lists(
    get_item("ints", root()),
    Some(list_contains(get_item("lists", root()), lit(1501i64))),
    None,
    None
)]
#[case::lengths(list_length(get_item("lists", root())), None, Some(10..20), None)]
#[tokio::test]
async fn list_layouts_match_v1(
    #[values(false, true)] zoned: bool,
    #[case] projection: Expression,
    #[case] filter: Option<Expression>,
    #[case] row_range: Option<Range<u64>>,
    #[case] rows: Option<&'static [u64]>,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let leaf = if zoned {
        repartitioned_zoned_leaf()
    } else {
        chunked_flat()
    };
    let strategy = TableStrategy::new(chunked_flat(), leaf).with_list_layout();
    let (segments, layout) = write(&session, &strategy).await?;
    assert_same_as_v1(&session, || {
        let mut builder = builder(&session, &segments, &layout)?
            .with_projection(projection.bind(layout.dtype())?)
            .with_some_filter(
                filter
                    .as_ref()
                    .map(|filter| filter.bind(layout.dtype()))
                    .transpose()?,
            );
        if let Some(row_range) = row_range.clone() {
            builder = builder.with_row_range(row_range);
        }
        if let Some(rows) = rows {
            builder =
                builder.with_row_indices(StrictSortedBuffer::try_new(Buffer::copy_from(rows))?);
        }
        Ok(builder)
    })
    .await
}

#[tokio::test]
async fn projection_skips_unsupported_unselected_layouts() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let strategy = TableStrategy::new(chunked_flat(), chunked_flat()).with_list_layout();
    let (segments, layout) = write(&session, &strategy).await?;
    assert_same_as_v1(&session, || {
        Ok(builder(&session, &segments, &layout)?
            .with_projection(get_item("ints", root()).bind(layout.dtype())?))
    })
    .await
}

/// Each ancestor masks a different row, and the nullable field has a null of its own.
fn nullable_struct_with_list(session: &VortexSession) -> VortexResult<ArrayRef> {
    let inner = StructArray::try_new(
        FieldNames::from(["value", "optional"]),
        [
            buffer![10i32, 20, 30, 40].into_array(),
            PrimitiveArray::from_option_iter([Some(11i32), Some(21), Some(31), None]).into_array(),
        ],
        4,
        Validity::from_iter([true, true, false, true]),
    )?;
    let items = ListArray::try_new(
        buffer![1i32, 2, 3, 4].into_array(),
        buffer![0u32, 1, 2, 3, 4].into_array(),
        Validity::NonNullable,
        &mut session.create_execution_ctx(),
    )?;
    Ok(StructArray::try_new(
        FieldNames::from(["inner", "items"]),
        [inner.into_array(), items.into_array()],
        4,
        Validity::from_iter([true, false, true, true]),
    )?
    .into_array())
}

#[rstest]
#[case::nested_nullable_field(get_item("optional", get_item("inner", root())), true, Some(3))]
#[case::whole_input(
    pack([("input", root()), ("marker", lit(7i32))], false.into()),
    false,
    None,
)]
#[tokio::test]
async fn nullable_struct_projection_preserves_validity(
    #[case] projection: Expression,
    #[case] list_layout: bool,
    #[case] expected_requests: Option<usize>,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let strategy = TableStrategy::new(chunked_flat(), chunked_flat());
    let strategy = if list_layout {
        strategy.with_list_layout()
    } else {
        strategy
    };
    let input = nullable_struct_with_list(&session)?;
    let (segments, layout) = write_chunks(&session, &strategy, vec![input.clone()]).await?;

    let source = Arc::new(CountingSegmentSource::new(segments));
    let scan_source: Arc<dyn SegmentSource> = Arc::<CountingSegmentSource>::clone(&source);
    let projection = projection.bind(layout.dtype())?;
    let scan = builder(&session, &scan_source, &layout)?.with_projection(projection.clone());
    let actual = collect_scan(scan.dtype().clone(), scan).await?;

    let expected = input.apply_bound(&projection)?;
    assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
    if let Some(expected_requests) = expected_requests {
        assert_eq!(source.requests(), expected_requests);
    }

    Ok(())
}

/// The zone map layout that [`StoredValueZones`] writes.
#[derive(Clone, Copy, Debug)]
enum ZoneMapLayout {
    /// The `vortex.zoned` layout.
    Zoned,
    /// The legacy `vortex.stats` layout, which older releases wrote.
    LegacyStats,
}

/// Writes each column's zone map over its stored values, even beneath a nullable struct.
///
/// Releases up to 0.88.0 wrote zone maps this way, so the null count of a struct field misses the
/// rows where the struct is null. Readers must still prune those files correctly. Develop writes
/// no zone map beneath a nullable struct, so the tests need this writer to produce one.
struct StoredValueZones {
    layout: ZoneMapLayout,
    zoned: ZonedStrategy,
}

#[async_trait]
impl LayoutStrategy for StoredValueZones {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        // NB: A fresh context drops the nullable-ancestor flag that the struct writer set. The old
        // writers had no such flag.
        let stored_ctx = LayoutWriterContext::new(ctx.array_ctx().clone());
        let stored_ctx = match self.layout {
            ZoneMapLayout::Zoned => stored_ctx,
            ZoneMapLayout::LegacyStats => stored_ctx.with_allowed_layouts(HashSet::from_iter([
                LegacyStats.id(),
                Chunked.id(),
                Flat.id(),
            ])),
        };

        self.zoned
            .write_stream(stored_ctx, segment_sink, stream, eof, session)
            .await
    }
}

/// Every column is shredded into chunked leaves with zone maps over their stored values.
fn stored_zones_table(layout: ZoneMapLayout) -> Arc<dyn LayoutStrategy> {
    let zoned = zoned_leaf();
    Arc::new(TableStrategy::new(
        chunked_flat(),
        Arc::new(StoredValueZones { layout, zoned }),
    ))
}

/// The number of leaf columns in a [`nullable_zone`].
const NULLABLE_ZONE_LEAVES: usize = 5;

/// A table whose zone maps include values that nullable structs hide.
///
/// Each scan gets a fresh reader and cache so its segment count includes every required read.
struct NullableZones {
    session: VortexSession,
    input: ArrayRef,
    segments: Arc<dyn SegmentSource>,
    layout: LayoutRef,
}

impl NullableZones {
    async fn write(zone_maps: ZoneMapLayout, with_nans: bool) -> VortexResult<Self> {
        let session = new_session().with_tokio();
        let chunks = (0..CHUNKS)
            .map(|zone| nullable_zone(zone, with_nans))
            .collect::<VortexResult<Vec<_>>>()?;
        let strategy = stored_zones_table(zone_maps);
        let (segments, layout) = write_chunks(&session, strategy.as_ref(), chunks.clone()).await?;
        assert_stored_zone_maps(&layout, zone_maps)?;
        let input = ChunkedArray::try_new(chunks, layout.dtype().clone())?.into_array();

        Ok(Self {
            session,
            input,
            segments,
            layout,
        })
    }

    async fn assert_scan(&self, use_v2: bool, predicate: Expression) -> VortexResult<usize> {
        let projection = get_item("id", root());
        let bound_projection = projection.bind(self.layout.dtype())?;
        let dtype = bound_projection.dtype().clone();
        let filter = predicate.bind(self.layout.dtype())?;
        let source = Arc::new(CountingSegmentSource::new(Arc::clone(&self.segments)));
        let segments: Arc<dyn SegmentSource> = Arc::<CountingSegmentSource>::clone(&source);

        let actual = if use_v2 {
            let scan = builder(&self.session, &segments, &self.layout)?
                .with_projection(bound_projection)
                .with_some_filter(Some(filter));
            collect_scan(dtype, scan).await?
        } else {
            let reader =
                self.layout
                    .new_reader("".into(), segments, &self.session, &Default::default())?;
            let scan = V1ScanBuilder::new(self.session.clone(), reader)
                .with_projection(bound_projection)
                .with_filter(filter);
            collect(dtype, scan.into_stream()?).await?
        };

        let mut ctx = self.session.create_execution_ctx();
        let keep = self
            .input
            .clone()
            .apply(&fill_null(predicate, lit(false)))?
            .execute::<Mask>(&mut ctx)?;
        let expected = self.input.clone().filter(keep)?.apply(&projection)?;
        assert_arrays_eq!(actual, expected, &mut ctx);

        Ok(source.requests())
    }
}

/// Asserts that every leaf column has a zone map in the `zone_maps` layout with a null count.
/// Beneath a nullable struct, that count misses the rows where the struct is null, so the scans
/// must never trust it.
#[track_caller]
fn assert_stored_zone_maps(layout: &LayoutRef, zone_maps: ZoneMapLayout) -> VortexResult<()> {
    let mut written = 0;
    for layout in layout.depth_first_traversal() {
        let layout = layout?;
        let aggregate_fns = if let Some(zoned) = layout.as_opt::<Zoned>() {
            assert!(matches!(zone_maps, ZoneMapLayout::Zoned));
            zoned.aggregate_fns()
        } else if let Some(legacy) = layout.as_opt::<LegacyStats>() {
            assert!(matches!(zone_maps, ZoneMapLayout::LegacyStats));
            legacy.aggregate_fns()
        } else {
            continue;
        };
        assert!(
            aggregate_fns
                .iter()
                .any(|aggregate_fn| aggregate_fn.is::<NullCount>())
        );
        written += 1;
    }

    assert_eq!(written, NULLABLE_ZONE_LEAVES);
    Ok(())
}

/// Builds zone `zone` of a table with an `id` column and a nullable struct `nested`.
///
/// `nested` hides zone 1 and the odd rows of zone 2. Its nullable struct field `deep` also hides
/// zone 3. Hidden integers and strings sort after every visible value, so they widen the stored
/// bounds. The optional NaNs include a visible value in zone 2 and values hidden by `nested` in
/// zone 1.
fn nullable_zone(zone: usize, with_nans: bool) -> VortexResult<ArrayRef> {
    let first_row = i32::try_from(zone * CHUNK_ROWS)?;
    let rows = first_row..first_row + i32::try_from(CHUNK_ROWS)?;
    let visible = |row: i32| zone != 1 && (zone != 2 || row % 2 == 0);
    let deep_visible = |row: i32| visible(row) && zone != 3;

    let ids = Buffer::from_iter(rows.clone());
    let ints = Buffer::from_iter(rows.clone().map(|row| stored_int(row, visible(row))));
    let deep_ints = Buffer::from_iter(rows.clone().map(|row| stored_int(row, deep_visible(row))));
    let floats = PrimitiveArray::from_option_iter(rows.clone().map(|row| {
        let value = if with_nans && (zone == 1 || (zone == 2 && row == first_row)) {
            f64::NAN
        } else if visible(row) {
            f64::from(row)
        } else {
            1e9_f64
        };
        (zone != 3 || row != first_row).then_some(value)
    }));
    let texts = VarBinArray::from_iter(
        rows.clone().map(|row| {
            Some(if visible(row) {
                format!("row {row:04}")
            } else {
                "~hidden".to_string()
            })
        }),
        DType::Utf8(Nullability::NonNullable),
    );

    let deep = StructArray::try_new(
        FieldNames::from(["int"]),
        [deep_ints.into_array()],
        CHUNK_ROWS,
        Validity::from_iter(rows.clone().map(|_| zone != 3)),
    )?;
    let nested = StructArray::try_new(
        FieldNames::from(["int", "float", "text", "deep"]),
        [
            ints.into_array(),
            floats.into_array(),
            texts.into_array(),
            deep.into_array(),
        ],
        CHUNK_ROWS,
        Validity::from_iter(rows.map(visible)),
    )?;

    Ok(
        StructArray::from_fields(&[("id", ids.into_array()), ("nested", nested.into_array())])?
            .into_array(),
    )
}

/// A hidden integer would overflow when adding one. A fallible filter must still see its row as
/// null after pruning.
fn stored_int(row: i32, visible: bool) -> i64 {
    if visible { i64::from(row) } else { i64::MAX }
}

fn nested_field(field: &str) -> Expression {
    get_item(field, get_item("nested", root()))
}

fn deep_int() -> Expression {
    get_item("int", nested_field("deep"))
}

/// `nested.int < cutoff()`, which is true for every row while `cutoff()` returns no value.
fn int_below_dynamic(
    cutoff: impl Fn() -> Option<ScalarValue> + Send + Sync + 'static,
) -> Expression {
    dynamic(
        CompareOperator::Lt,
        cutoff,
        DType::Primitive(PType::I64, Nullability::NonNullable),
        true,
        nested_field("int"),
    )
}

/// The segments that a scan of `id` reads when it filters on a field beneath `depth` nullable
/// structs: the field's zone map, then the field, each struct's validity, and the IDs of each zone
/// it keeps. The classic scan also reads the IDs of the zones it skips.
fn expected_reads(use_v2: bool, depth: usize, kept_zones: usize) -> usize {
    let per_kept_zone = 1 + depth + 1;
    let skipped_ids = if use_v2 { 0 } else { CHUNKS - kept_zones };

    1 + kept_zones * per_kept_zone + skipped_ids
}

/// Only zone 0 holds visible values below each cutoff, and the stored bounds of the other zones
/// still rule them out.
#[rstest]
#[case::int_bound(lt(nested_field("int"), lit(100i64)), 1)]
#[case::int_range(
    and(
        gt_eq(nested_field("int"), lit(0i64)),
        lt(nested_field("int"), lit(100i64)),
    ),
    1
)]
#[case::float_bound(lt(nested_field("float"), lit(100.0f64)), 1)]
#[case::text_bound(lt(nested_field("text"), lit("row 0100")), 1)]
#[case::text_range(
    and(
        gt_eq(nested_field("text"), lit("row 0000")),
        lt(nested_field("text"), lit("row 0100")),
    ),
    1
)]
#[case::deep_bound(lt(deep_int(), lit(100i64)), 2)]
#[tokio::test]
async fn bounds_under_nullable_structs_skip_zone_reads(
    #[values(false, true)] use_v2: bool,
    #[values(ZoneMapLayout::Zoned, ZoneMapLayout::LegacyStats)] zone_maps: ZoneMapLayout,
    #[case] filter: Expression,
    #[case] depth: usize,
) -> VortexResult<()> {
    let table = NullableZones::write(zone_maps, false).await?;
    let reads = table.assert_scan(use_v2, filter).await?;
    assert_eq!(reads, expected_reads(use_v2, depth, 1));
    Ok(())
}

/// Only the classic scan prunes a dynamic filter on a scan this small. The V2 scan skips the
/// pruning plans of dynamic conjuncts on scans of at most one filtered chunk.
#[rstest]
#[tokio::test]
async fn dynamic_bounds_under_nullable_structs_skip_classic_zone_reads(
    #[values(ZoneMapLayout::Zoned, ZoneMapLayout::LegacyStats)] zone_maps: ZoneMapLayout,
) -> VortexResult<()> {
    let table = NullableZones::write(zone_maps, false).await?;
    let reads = table
        .assert_scan(false, int_below_dynamic(|| Some(100i64.into())))
        .await?;
    assert_eq!(reads, expected_reads(false, 1, 1));
    Ok(())
}

/// Zones 0 and 3 store no NaNs. Zone 1 stores NaNs hidden by `nested`, so a scan must read it.
#[rstest]
#[tokio::test]
async fn nan_counts_under_nullable_structs_skip_zone_reads(
    #[values(false, true)] use_v2: bool,
    #[values(ZoneMapLayout::Zoned, ZoneMapLayout::LegacyStats)] zone_maps: ZoneMapLayout,
) -> VortexResult<()> {
    let table = NullableZones::write(zone_maps, true).await?;
    let reads = table
        .assert_scan(use_v2, is_nan(nested_field("float")))
        .await?;
    assert_eq!(reads, expected_reads(use_v2, 1, 2));
    Ok(())
}

#[rstest]
#[case::parent_nulls(is_null(nested_field("int")))]
#[case::deep_parent_nulls(is_null(deep_int()))]
#[case::loose_bound(lt(nested_field("int"), lit(2050i64)))]
#[case::loose_range(and(
    gt_eq(nested_field("int"), lit(10i64)),
    lt(nested_field("int"), lit(2050i64)),
))]
#[case::nulls_or_bounds(or(
    is_null(nested_field("float")),
    lt(nested_field("float"), lit(10.0f64)),
))]
#[case::text_not_eq(not_eq(nested_field("text"), lit("row 0000")))]
#[case::deep_or_text(or(lt(deep_int(), lit(10i64)), eq(nested_field("text"), lit("row 3000")),))]
#[case::narrow_cast(gt(
    cast(
        nested_field("int"),
        DType::Primitive(PType::I16, Nullability::Nullable)
    ),
    lit(10i16),
))]
#[case::overflow(gt(checked_add(nested_field("int"), lit(1i64)), lit(10i64)))]
#[case::dynamic_bound(int_below_dynamic(|| Some(2050i64.into())))]
#[case::dynamic_without_value(int_below_dynamic(|| None))]
#[case::nan(is_nan(nested_field("float")))]
#[case::not_nan(not(is_nan(nested_field("float"))))]
#[case::float_bounds_with_nans(lt(nested_field("float"), lit(100.0f64)))]
#[tokio::test]
async fn pruning_under_nullable_structs_preserves_matching_rows(
    #[values(false, true)] use_v2: bool,
    #[values(ZoneMapLayout::Zoned, ZoneMapLayout::LegacyStats)] zone_maps: ZoneMapLayout,
    #[case] filter: Expression,
) -> VortexResult<()> {
    NullableZones::write(zone_maps, true)
        .await?
        .assert_scan(use_v2, filter)
        .await?;
    Ok(())
}

#[test]
fn partition_rows_tile_the_file_on_row_blocks() {
    let (file_size, row_count) = (1_000u64, 100_000u64);
    let bytes = [0..300, 300..600, 600..1_000];
    let rows: Vec<_> = bytes
        .iter()
        .map(|bytes| super::partition_row_range(bytes.clone(), file_size, row_count))
        .collect();
    assert_eq!(rows[0].start, 0);
    assert_eq!(rows[2].end, row_count);
    for pair in rows.windows(2) {
        assert_eq!(pair[0].end, pair[1].start);
        assert_eq!(pair[0].end % ROW_BLOCK_SIZE as u64, 0);
    }
    assert!(rows.iter().all(|rows| !rows.is_empty()));
    assert_eq!(
        super::partition_row_range(1..2, file_size, row_count),
        0..0,
        "a range inside one block owns no rows"
    );
}
