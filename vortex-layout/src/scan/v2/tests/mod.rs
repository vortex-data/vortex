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
use futures::Stream;
use futures::TryStreamExt;
use futures::stream;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::expr::Expression;
use vortex_array::expr::Lambda;
use vortex_array::expr::and;
use vortex_array::expr::cast;
use vortex_array::expr::checked_add;
use vortex_array::expr::eq;
use vortex_array::expr::ext_storage;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::is_not_null;
use vortex_array::expr::is_null;
use vortex_array::expr::list_contains;
use vortex_array::expr::list_length;
use vortex_array::expr::list_transform;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::not;
use vortex_array::expr::or;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::var;
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

use super::ChunkPlanner;
use super::PreparedScan;
use super::PrunePlanner;
use super::ROW_BLOCK_SIZE;
use super::ScanBuilder;
use super::ScanFile;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::repartition::RepartitionStrategy;
use crate::layouts::repartition::RepartitionWriterOptions;
use crate::layouts::row_idx::row_idx;
use crate::layouts::table::TableStrategy;
use crate::layouts::zoned::writer::ZonedLayoutOptions;
use crate::layouts::zoned::writer::ZonedStrategy;
use crate::scan::arrow::to_record_batch;
use crate::scan::scan_builder::ScanBuilder as V1ScanBuilder;
use crate::segments::CountingSegmentSource;
use crate::segments::SegmentSource;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt as _;
use crate::test::TestPoint;
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

/// A chunk of `CHUNK_ROWS` rows starting at row `start`: an integer column and a nullable
/// extension column whose struct storage holds the row and ten times the row.
fn point_chunk(start: usize) -> VortexResult<ArrayRef> {
    points_chunk(start, true, Nullability::Nullable)
}

/// [`point_chunk`] with a non-nullable extension column.
fn non_nullable_point_chunk(start: usize) -> VortexResult<ArrayRef> {
    points_chunk(start, true, Nullability::NonNullable)
}

/// [`point_chunk`] with the storage struct in place of the extension column.
fn plain_point_chunk(start: usize) -> VortexResult<ArrayRef> {
    points_chunk(start, false, Nullability::Nullable)
}

fn points_chunk(start: usize, extension: bool, nullability: Nullability) -> VortexResult<ArrayRef> {
    let rows = start..start + CHUNK_ROWS;
    let ids = Buffer::from_iter(
        rows.clone()
            .map(|row| i32::try_from(row).unwrap_or(i32::MAX)),
    );
    let tens = Buffer::from_iter(
        rows.clone()
            .map(|row| i64::try_from(10 * row).unwrap_or(i64::MAX)),
    );
    let validity = match nullability {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable => Validity::from_iter(rows.map(|row| row % 5 != 0)),
    };
    let storage = StructArray::try_new(
        FieldNames::from(["a", "b"]),
        [ids.clone().into_array(), tens.into_array()],
        CHUNK_ROWS,
        validity,
    )?;
    let points = if extension {
        ExtensionArray::try_new(TestPoint::dtype(nullability), storage.into_array())?.into_array()
    } else {
        storage.into_array()
    };
    Ok(StructArray::from_fields(&[("id", ids.into_array()), ("point", points)])?.into_array())
}

/// A chunk of `CHUNK_ROWS` rows starting at row `start`, shaped like a protobuf message column:
/// `{id, msg: {x, inner: {y}?, list}?}`, where `msg` is null at every fifth row and `inner` at
/// every seventh.
///
/// `x` is the row, `y` ten times the row and `list` holds the row twice. The fields keep their
/// values in rows where a struct above them is null, except that a list there holds `i64::MAX` in
/// place of its second row, which overflows when added to.
fn message_chunk(start: usize) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let rows = start..start + CHUNK_ROWS;
    let ids = Buffer::from_iter(
        rows.clone()
            .map(|row| i32::try_from(row).unwrap_or(i32::MAX)),
    );
    let tens = Buffer::from_iter(
        rows.clone()
            .map(|row| i64::try_from(10 * row).unwrap_or(i64::MAX)),
    );
    let inner = StructArray::try_new(
        FieldNames::from(["y"]),
        [tens.into_array()],
        CHUNK_ROWS,
        Validity::from_iter(rows.clone().map(|row| row % 7 != 0)),
    )?;
    let elements = Buffer::from_iter(rows.clone().flat_map(|row| {
        let value = i64::try_from(row).unwrap_or(i64::MAX);
        [
            value,
            if row.is_multiple_of(5) {
                i64::MAX
            } else {
                value
            },
        ]
    }));
    let offsets = Buffer::from_iter((0..=CHUNK_ROWS).map(|i| u32::try_from(2 * i).unwrap_or(0)));
    let list = ListArray::try_new(
        elements.into_array(),
        offsets.into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?;
    let msg = StructArray::try_new(
        FieldNames::from(["x", "inner", "list"]),
        [
            ids.clone().into_array(),
            inner.into_array(),
            list.into_array(),
        ],
        CHUNK_ROWS,
        Validity::from_iter(rows.map(|row| row % 5 != 0)),
    )?;
    Ok(
        StructArray::from_fields(&[("id", ids.into_array()), ("msg", msg.into_array())])?
            .into_array(),
    )
}

/// A chunk of `CHUNK_ROWS` rows starting at row `start`: `{id, s: {a}?}` with a non-nullable `a`
/// that holds values in rows where `s` is null, like a proto3 scalar in an absent message.
///
/// Chunk `k` is the `k`-th zone of a zoned layout. `a` is the row wherever `s` is valid. `s` is null
/// in all of zone 1 and in the odd rows of zone 2, where `a` holds 1,000,000 plus the row: values
/// that no read through `s` returns but that the zone map of `a` sees.
fn hidden_values_chunk(start: usize) -> VortexResult<ArrayRef> {
    let rows = start..start + CHUNK_ROWS;
    let valid = |row: usize| match row / CHUNK_ROWS {
        1 => false,
        2 => row.is_multiple_of(2),
        _ => true,
    };
    let ids = Buffer::from_iter(
        rows.clone()
            .map(|row| i32::try_from(row).unwrap_or(i32::MAX)),
    );
    let values = Buffer::from_iter(rows.clone().map(|row| {
        let value = if valid(row) { row } else { 1_000_000 + row };
        i64::try_from(value).unwrap_or(i64::MAX)
    }));
    let s = StructArray::try_new(
        FieldNames::from(["a"]),
        [values.into_array()],
        CHUNK_ROWS,
        Validity::from_iter(rows.map(valid)),
    )?;
    Ok(StructArray::from_fields(&[("id", ids.into_array()), ("s", s.into_array())])?.into_array())
}

/// A table chunk of `id` and a nullable struct `s` of `{small: i64, x: i64, y: i64, f: f64}`, which
/// is null where it is in [`hidden_values_chunk`].
///
/// Where `s` is valid, `small` and `x` hold `row % 100`, `y` holds 50 and `f` holds `row + 0.5`.
/// Where it is null, they hold values that no read through `s` returns: `small` holds `i64::MAX`,
/// which is outside the range of `i8` and overflows when added to, `x` is below every `y`, and `f`
/// is above every visible `f`.
fn masked_fields_chunk(start: usize) -> VortexResult<ArrayRef> {
    let rows = start..start + CHUNK_ROWS;
    let valid = |row: usize| match row / CHUNK_ROWS {
        1 => false,
        2 => row.is_multiple_of(2),
        _ => true,
    };
    let field = |visible: fn(i64) -> i64, hidden: fn(i64) -> i64| {
        Buffer::from_iter(rows.clone().map(|row| {
            let value = i64::try_from(row).unwrap_or(i64::MAX);
            if valid(row) {
                visible(value)
            } else {
                hidden(value)
            }
        }))
        .into_array()
    };
    let f = Buffer::from_iter(rows.clone().map(|row| {
        let value = f64::from(u32::try_from(row).unwrap_or(u32::MAX));
        if valid(row) { value + 0.5 } else { 1e9 + value }
    }));
    let s = StructArray::try_new(
        FieldNames::from(["small", "x", "y", "f"]),
        [
            field(|row| row % 100, |_| i64::MAX),
            field(|row| row % 100, |_| -1_000_000),
            field(|_| 50, |_| 1_000_000),
            f.into_array(),
        ],
        CHUNK_ROWS,
        Validity::from_iter(rows.clone().map(valid)),
    )?;
    let ids = Buffer::from_iter(rows.map(|row| i32::try_from(row).unwrap_or(i32::MAX)));
    Ok(StructArray::from_fields(&[("id", ids.into_array()), ("s", s.into_array())])?.into_array())
}

/// Writes `CHUNKS` table chunks with `strategy`.
async fn write(
    session: &VortexSession,
    strategy: &dyn LayoutStrategy,
) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef)> {
    write_chunks(session, strategy, make_chunks(table_chunk)?).await
}

/// Makes `CHUNKS` chunks with `chunk`, called with each chunk's first row.
fn make_chunks(chunk: fn(usize) -> VortexResult<ArrayRef>) -> VortexResult<Vec<ArrayRef>> {
    (0..CHUNKS).map(|index| chunk(index * CHUNK_ROWS)).collect()
}

/// Writes `chunks`, which must hold at least one chunk, with `strategy`.
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
fn table_strategy() -> TableStrategy {
    TableStrategy::new(chunked_flat(), chunked_flat())
}

/// Every column is shredded into chunked leaves with zone statistics.
///
/// Leaves are repartitioned into zones of `CHUNK_ROWS` rows first, as the file writer does, since a
/// list layout writes its elements in chunks as long as the lists they hold.
fn zoned_table_strategy() -> TableStrategy {
    let zoned = ZonedStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        FlatLayoutStrategy::default(),
        ZonedLayoutOptions {
            block_size: NonZeroUsize::new(CHUNK_ROWS).unwrap_or(NonZeroUsize::MIN),
            ..Default::default()
        },
    );
    let repartition = RepartitionStrategy::new(
        zoned,
        RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: CHUNK_ROWS,
            block_size_target: None,
            canonicalize: false,
        },
    );
    TableStrategy::new(chunked_flat(), Arc::new(repartition))
}

fn table() -> Arc<dyn LayoutStrategy> {
    Arc::new(table_strategy())
}

fn zoned_table() -> Arc<dyn LayoutStrategy> {
    Arc::new(zoned_table_strategy())
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
    Some(list_contains(get_item("lists", root()), lit(4001i64))),
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
    let strategy = if zoned {
        zoned_table_strategy()
    } else {
        table_strategy()
    }
    .with_list_layout();
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

fn point_storage(field: &str) -> Expression {
    get_item(field, ext_storage(get_item("point", root())))
}

/// Extension columns written with the extension layout scan the same on both executors, and as
/// the expressions evaluate over the input.
#[rstest]
#[case::field(point_storage("a"), None)]
#[case::values(get_item("point", root()), None)]
#[case::is_null(is_null(get_item("point", root())), None)]
#[case::mixed(
    pack(
        [("a", point_storage("a")), ("point", get_item("point", root()))],
        Nullability::NonNullable,
    ),
    None,
)]
#[case::filter_field(get_item("id", root()), Some(gt(point_storage("a"), lit(2500i32))))]
#[case::filter_values(
    get_item("point", root()),
    Some(and(
        gt(point_storage("a"), lit(995i32)),
        lt(point_storage("b"), lit(20_500i64)),
    ))
)]
#[case::filter_is_null(point_storage("b"), Some(is_null(get_item("point", root()))))]
#[tokio::test]
async fn extension_layout_matches_v1(
    #[values(false, true)] zoned: bool,
    #[case] projection: Expression,
    #[case] filter: Option<Expression>,
) -> VortexResult<()> {
    let strategy = if zoned {
        zoned_table_strategy()
    } else {
        table_strategy()
    }
    .with_extension_layout([TestPoint.id()]);
    assert_matches_v1_and_input(&strategy, point_chunk, &projection, filter.as_ref()).await
}

fn msg(field: &str) -> Expression {
    get_item(field, get_item("msg", root()))
}

/// A nullable struct with a list field, written with the list layout, scans the same on both
/// executors, also where the scan reads the list. Its fields are null wherever a struct above them
/// is.
#[rstest]
#[case::sibling_of_list(msg("x"), None)]
#[case::list(msg("list"), None)]
#[case::list_length(list_length(msg("list")), None)]
#[case::list_transform(
    list_transform(msg("list"), Lambda::new(["x"], gt(var("x"), lit(2500i64)))?),
    None
)]
#[case::fallible_list_transform(
    list_transform(
        msg("list"),
        Lambda::new(["x"], checked_add(var("x"), lit(1i64)))?,
    ),
    None
)]
#[case::filter_on_list(get_item("id", root()), Some(list_contains(msg("list"), lit(2500i64))))]
#[case::nested_nullable(get_item("y", msg("inner")), None)]
#[case::struct_validity(is_null(get_item("msg", root())), None)]
#[case::pack(
    pack(
        [("id", get_item("id", root())), ("x", msg("x"))],
        Nullability::NonNullable,
    ),
    None,
)]
#[case::filter_field(get_item("id", root()), Some(gt(msg("x"), lit(2500i32))))]
#[case::filter_nested(msg("x"), Some(lt(get_item("y", msg("inner")), lit(20_500i64))))]
#[case::filter_nested_null(msg("x"), Some(is_null(get_item("y", msg("inner")))))]
#[tokio::test]
async fn nullable_struct_beside_a_list_matches_v1(
    #[values(false, true)] zoned: bool,
    #[case] projection: Expression,
    #[case] filter: Option<Expression>,
) -> VortexResult<()> {
    let strategy = if zoned {
        zoned_table_strategy()
    } else {
        table_strategy()
    }
    .with_list_layout();
    assert_matches_v1_and_input(&strategy, message_chunk, &projection, filter.as_ref()).await
}

fn s_a() -> Expression {
    get_item("a", get_item("s", root()))
}

/// Zone maps beneath a nullable struct prune filters on its fields without losing rows.
///
/// The zones of `s.a` see the values it holds where `s` is null, which no read through `s`
/// returns. Bounds over them are only looser, so `s.a > 2500` reads zone 1 and finds no row
/// there. A null count over them would be 0 in zones 1 and 2, where `is_null(s.a)` holds, so the
/// zone map records none (#9205).
#[rstest]
#[case::gt(gt(s_a(), lit(2500i64)))]
#[case::lt(lt(s_a(), lit(100i64)))]
#[case::hidden_value(eq(s_a(), lit(1_001_500i64)))]
#[case::not(not(gt(s_a(), lit(500i64))))]
#[case::is_null(is_null(s_a()))]
#[case::is_not_null(is_not_null(s_a()))]
#[case::is_null_or_lt(or(is_null(s_a()), lt(s_a(), lit(10i64))))]
#[tokio::test]
async fn pruning_beneath_a_nullable_struct_keeps_every_row(
    #[case] filter: Expression,
) -> VortexResult<()> {
    assert_matches_v1_and_input(
        &zoned_table_strategy(),
        hidden_values_chunk,
        &get_item("id", root()),
        Some(&filter),
    )
    .await
}

fn s_field(name: &str) -> Expression {
    get_item(name, get_item("s", root()))
}

/// Pruning beneath a nullable struct keeps every row, also where the zone map statistics of a
/// field feed a function that fails on the values the struct hides, where a filter compares two
/// fields of the struct, and for a float field, whose bounds prune only with a proof that the zone
/// holds no NaN.
#[rstest]
#[case::cast(gt(
    cast(s_field("small"), DType::Primitive(PType::I8, Nullability::Nullable)),
    lit(5i8)
))]
#[case::cast_lt(lt(
    cast(s_field("small"), DType::Primitive(PType::I8, Nullability::Nullable)),
    lit(5i8)
))]
#[case::overflow(gt(checked_add(s_field("small"), lit(1i64)), lit(95i64)))]
#[case::two_fields(gt(s_field("x"), s_field("y")))]
#[case::float_gt(gt(s_field("f"), lit(1500.0f64)))]
#[case::float_lt(lt(s_field("f"), lit(100.0f64)))]
#[case::float_not(not(gt(s_field("f"), lit(500.0f64))))]
#[tokio::test]
async fn pruning_beneath_a_nullable_struct_keeps_rows_of_any_filter(
    #[case] filter: Expression,
) -> VortexResult<()> {
    assert_matches_v1_and_input(
        &zoned_table_strategy(),
        masked_fields_chunk,
        &get_item("id", root()),
        Some(&filter),
    )
    .await
}

/// A filter on a field beneath a nullable struct reads only the zones its bounds keep.
///
/// A filter the bounds of the field cannot prune reads its zone map and, in each of the four
/// zones, the field, the validity of `s` and `id`: 13 segments. The pruned filter holds only in
/// zone 0, so it reads the three there alone. The classic scan still reads `id` in every zone. A
/// float field prunes too: its zone map records the NaN count of each zone, and a count of zero
/// proves what the bounds of a float need, that the zone holds no NaN.
#[rstest]
#[case::v1(false, hidden_values_chunk, s_a(), lit(100i64), lit(10_000_000i64), 7)]
#[case::v2(true, hidden_values_chunk, s_a(), lit(100i64), lit(10_000_000i64), 4)]
#[case::v1_float(
    false,
    masked_fields_chunk,
    s_field("f"),
    lit(100.0f64),
    lit(1e12f64),
    7
)]
#[case::v2_float(
    true,
    masked_fields_chunk,
    s_field("f"),
    lit(100.0f64),
    lit(1e12f64),
    4
)]
#[tokio::test]
async fn pruning_beneath_a_nullable_struct_skips_zones(
    #[case] v2: bool,
    #[case] chunk: fn(usize) -> VortexResult<ArrayRef>,
    #[case] field: Expression,
    #[case] pruned: Expression,
    #[case] unpruned: Expression,
    #[case] pruned_requests: usize,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) =
        write_chunks(&session, &zoned_table_strategy(), make_chunks(chunk)?).await?;
    let mut requests = Vec::new();
    for filter in [lt(field.clone(), pruned), lt(field, unpruned)] {
        let source = Arc::new(CountingSegmentSource::new(Arc::clone(&segments)));
        let projection = get_item("id", root()).bind(layout.dtype())?;
        let filter = filter.bind(layout.dtype())?;
        let dtype = projection.dtype().clone();
        if v2 {
            let segments: Arc<dyn SegmentSource> = Arc::<CountingSegmentSource>::clone(&source);
            let scan = builder(&session, &segments, &layout)?
                .with_projection(projection)
                .with_some_filter(Some(filter));
            collect_scan(dtype, scan).await?;
        } else {
            let reader = layout.new_reader(
                "".into(),
                Arc::<CountingSegmentSource>::clone(&source),
                &session,
                &Default::default(),
            )?;
            let scan = V1ScanBuilder::new(session.clone(), reader)
                .with_projection(projection)
                .with_filter(filter);
            collect(dtype, scan.into_stream()?).await?;
        }
        requests.push(source.requests());
    }
    assert_eq!(requests, [pruned_requests, 13]);
    Ok(())
}

/// Writes `CHUNKS` chunks made by `chunk` with `strategy`, then scans them with `projection` and
/// `filter` on both executors. Checks both return what the expressions return over the input.
async fn assert_matches_v1_and_input(
    strategy: &dyn LayoutStrategy,
    chunk: fn(usize) -> VortexResult<ArrayRef>,
    projection: &Expression,
    filter: Option<&Expression>,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_chunks(&session, strategy, make_chunks(chunk)?).await?;
    let dtype = layout.dtype().clone();
    let make = || -> VortexResult<ScanBuilder> {
        Ok(builder(&session, &segments, &layout)?
            .with_projection(projection.bind(&dtype)?)
            .with_some_filter(filter.map(|filter| filter.bind(&dtype)).transpose()?))
    };
    assert_same_as_v1(&session, make).await?;

    let input = ChunkedArray::try_new(
        (0..CHUNKS)
            .map(|index| chunk(index * CHUNK_ROWS))
            .collect::<VortexResult<Vec<_>>>()?,
        layout.dtype().clone(),
    )?
    .into_array();
    let mut ctx = session.create_execution_ctx();
    let input = match filter {
        Some(filter) => {
            let mask = input
                .clone()
                .apply(filter)?
                .fill_null(false)?
                .execute::<Mask>(&mut ctx)?;
            input.filter(mask)?
        }
        None => input,
    };
    let expected = input.apply(projection)?;
    let scan = make()?;
    let actual = collect_scan(scan.dtype().clone(), scan).await?;
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

/// Filtering a storage field through `ext_storage` returns the same rows and reads the same
/// segments as filtering the same field of a plain struct column.
#[rstest]
#[tokio::test]
async fn extension_filter_prunes_like_struct(#[values(false, true)] v2: bool) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let strategy = zoned_table_strategy().with_extension_layout([TestPoint.id()]);
    let mut scans = Vec::new();
    for (chunk, point) in [
        (
            point_chunk as fn(usize) -> VortexResult<ArrayRef>,
            ext_storage(get_item("point", root())),
        ),
        (plain_point_chunk, get_item("point", root())),
    ] {
        let (segments, layout) = write_chunks(&session, &strategy, make_chunks(chunk)?).await?;
        let source = Arc::new(CountingSegmentSource::new(segments));
        let projection = get_item("id", root()).bind(layout.dtype())?;
        let filter = gt(get_item("a", point), lit(3500i32)).bind(layout.dtype())?;
        let dtype = projection.dtype().clone();
        let rows = if v2 {
            let segments: Arc<dyn SegmentSource> = Arc::<CountingSegmentSource>::clone(&source);
            let scan = builder(&session, &segments, &layout)?
                .with_projection(projection)
                .with_some_filter(Some(filter));
            collect_scan(dtype, scan).await?
        } else {
            let reader = layout.new_reader(
                "".into(),
                Arc::<CountingSegmentSource>::clone(&source),
                &session,
                &Default::default(),
            )?;
            let scan = V1ScanBuilder::new(session.clone(), reader)
                .with_projection(projection)
                .with_filter(filter);
            collect(dtype, scan.into_stream()?).await?
        };
        scans.push((rows, source.requests()));
    }

    let mut ctx = session.create_execution_ctx();
    assert_arrays_eq!(scans[0].0, scans[1].0, &mut ctx);
    assert_eq!(scans[0].1, scans[1].1);
    Ok(())
}

/// Projecting one storage field of an extension column reads the segments of that field, plus
/// those of the storage validity when the extension is nullable. A plain nullable struct column
/// reads the same segments, so the extension layout adds no reads of its own.
#[rstest]
#[case::v1(false, non_nullable_point_chunk, point_storage("a"), 1)]
#[case::v1_nullable(false, point_chunk, point_storage("a"), 2)]
#[case::v1_nullable_struct(false, plain_point_chunk, get_item("a", get_item("point", root())), 2)]
#[case::v2(true, non_nullable_point_chunk, point_storage("a"), 1)]
#[case::v2_nullable(true, point_chunk, point_storage("a"), 2)]
#[case::v2_nullable_struct(true, plain_point_chunk, get_item("a", get_item("point", root())), 2)]
#[tokio::test]
async fn extension_field_projection_reads(
    #[case] v2: bool,
    #[case] chunk: fn(usize) -> VortexResult<ArrayRef>,
    #[case] projection: Expression,
    #[case] columns_read: usize,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let strategy = table_strategy().with_extension_layout([TestPoint.id()]);
    let (segments, layout) = write_chunks(&session, &strategy, make_chunks(chunk)?).await?;
    let source = Arc::new(CountingSegmentSource::new(segments));
    let projection = projection.bind(layout.dtype())?;
    let dtype = projection.dtype().clone();
    let actual = if v2 {
        let segments: Arc<dyn SegmentSource> = Arc::<CountingSegmentSource>::clone(&source);
        let scan = builder(&session, &segments, &layout)?.with_projection(projection);
        collect_scan(dtype, scan).await?
    } else {
        let reader = layout.new_reader(
            "".into(),
            Arc::<CountingSegmentSource>::clone(&source),
            &session,
            &Default::default(),
        )?;
        let scan = V1ScanBuilder::new(session.clone(), reader).with_projection(projection);
        collect(dtype, scan.into_stream()?).await?
    };

    assert_eq!(actual.len(), CHUNKS * CHUNK_ROWS);
    assert_eq!(source.requests(), columns_read * CHUNKS);
    Ok(())
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
