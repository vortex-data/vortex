// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_schema::Field;
use futures::TryStreamExt;
use futures::stream;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::dynamic;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowSessionExt;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::request::Completion;
use vortex_io::request::IoBatch;
use vortex_io::request::IoIntent;
use vortex_io::request::IoOwnerId;
use vortex_io::request::IoSource;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_scan::planning::driver::Batch;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;

use super::AccessMode;
use super::ChunkPlanner;
use super::PreparedScan;
use super::PrunePlanner;
use super::ROW_BLOCK_SIZE;
use super::ScanBuilder;
use super::ScanCache;
use super::ScanFile;
use super::filtered_rows_per_chunk;
use super::rows_per_chunk;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::dict::DictLayout;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::row_idx::row_idx;
use crate::layouts::table::TableStrategy;
use crate::layouts::zoned::writer::ZonedLayoutOptions;
use crate::layouts::zoned::writer::ZonedStrategy;
use crate::scan::arrow::to_record_batch;
use crate::scan::execution::SegmentLocation;
use crate::scan::execution::row_chunks;
use crate::scan::execution::scan_chunks;
use crate::scan::scan_builder::ScanBuilder as V1ScanBuilder;
use crate::segments::CountingSegmentSource;
use crate::segments::SegmentSource;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
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
    let segments = Arc::new(TestSegments::default());
    let (mut sequence_id, eof) = SequenceId::root().split();
    let chunks = (0..CHUNKS)
        .map(|chunk| Ok((sequence_id.advance(), table_chunk(chunk * CHUNK_ROWS)?)))
        .collect::<Vec<_>>();
    let dtype = table_chunk(0)?.dtype().clone();
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

/// Every column is shredded into chunked leaves with zone statistics.
fn zoned_table() -> Arc<dyn LayoutStrategy> {
    let zoned = ZonedStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        FlatLayoutStrategy::default(),
        ZonedLayoutOptions {
            block_size: NonZeroUsize::new(CHUNK_ROWS).unwrap_or(NonZeroUsize::MIN),
            ..Default::default()
        },
    );
    Arc::new(TableStrategy::new(chunked_flat(), Arc::new(zoned)))
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

#[rstest]
#[case::count_integer(lit(true), gt(get_item("ints", root()), lit(10)), 1_048_576)]
#[case::project_integer(
    get_item("ints", root()),
    gt(get_item("ints", root()), lit(10)),
    1_048_576
)]
#[case::filter_string(lit(true), eq(get_item("strings", root()), lit("row 10")), 131_072)]
#[case::project_string(
    get_item("strings", root()),
    gt(get_item("ints", root()), lit(10)),
    131_072
)]
#[case::wide_projection(root(), gt(get_item("ints", root()), lit(10)), 131_072)]
fn filtered_chunk_sizing(
    #[case] projection: Expression,
    #[case] filter: Expression,
    #[case] expected: usize,
) -> VortexResult<()> {
    let array = table_chunk(0)?;
    let dtype = array.dtype();
    assert_eq!(
        filtered_rows_per_chunk(dtype, &projection.bind(dtype)?, &filter.bind(dtype)?)?,
        expected,
    );
    Ok(())
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
            Progress::Waiting => std::future::poll_fn(|cx| run.poll_completion(cx)).await?,
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
    stream: impl futures::Stream<Item = VortexResult<ArrayRef>>,
) -> VortexResult<ArrayRef> {
    let chunks: Vec<ArrayRef> = stream.try_collect().await?;
    Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
}

/// Prepares and runs a whole scan, returning its rows as one chunked array.
async fn collect_scan(dtype: DType, builder: ScanBuilder) -> VortexResult<ArrayRef> {
    Ok(ChunkedArray::try_new(run_scan(builder).await?, dtype)?.into_array())
}

#[rstest]
#[tokio::test]
async fn sparse_row_lookup_matches_scan_with_filters_and_nested_values(
    #[values(None, Some(AccessMode::Scan), Some(AccessMode::RandomAccess))] mode: Option<
        AccessMode,
    >,
    #[values(false, true)] roaring: bool,
    #[values(false, true)] filtered: bool,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, table().as_ref()).await?;
    let indices = vec![0, 100, 999, 1001, 2000, 3999];
    let selection = if roaring {
        vortex_scan::selection::Selection::IncludeRoaring(indices.into_iter().collect())
    } else {
        vortex_scan::selection::Selection::IncludeByIndex(StrictSortedBuffer::try_new(
            Buffer::from(indices),
        )?)
    };
    let mut builder = builder(&session, &segments, &layout)?
        .with_row_range(200..2500)
        .with_selection(selection);
    if let Some(mode) = mode {
        builder = builder.with_access_mode(mode);
    }
    if filtered {
        builder = builder.with_some_filter(Some(
            gt(get_item("ints", root()), lit(1000i32)).bind(layout.dtype())?,
        ));
    }
    let scan = Arc::new(builder.prepare()?);
    assert_eq!(scan.access_mode(), mode.unwrap_or(AccessMode::RandomAccess));
    let actual = ChunkedArray::try_new(run_rows(&scan, 200..2500).await?, layout.dtype().clone())?;
    let source = ChunkedArray::try_new(
        (0..CHUNKS)
            .map(|i| table_chunk(i * CHUNK_ROWS))
            .collect::<VortexResult<Vec<_>>>()?,
        layout.dtype().clone(),
    )?
    .into_array();
    let expected = if filtered {
        vec![1001u64, 2000]
    } else {
        vec![999u64, 1001, 2000]
    };
    assert_arrays_eq!(
        actual,
        source.take(Buffer::from(expected).into_array())?,
        &mut session.create_execution_ctx()
    );
    Ok(())
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

#[rstest]
#[case::whole(0..150_000, vec![0..65_536, 65_536..131_072, 131_072..150_000])]
#[case::unaligned(10..140_000, vec![10..65_536, 65_536..131_072, 131_072..140_000])]
#[case::boundary(65_536..131_072, vec![65_536..131_072])]
#[case::empty(5..5, vec![])]
#[case::near_max(u64::MAX - 3..u64::MAX, vec![u64::MAX - 3..u64::MAX])]
fn fixed_row_chunks(#[case] rows: Range<u64>, #[case] expected: Vec<Range<u64>>) {
    assert_eq!(row_chunks(rows, 65_536), expected);
}

/// A scan joins a chunk at either edge of its rows that is under half a chunk to its neighbour,
/// so a partition whose edges fall off the chunk grid plans no sliver of a chunk.
#[rstest]
#[case::partition_tail(0..90_112, 81_920, vec![0..90_112])]
#[case::partition_head(90_112..180_224, 81_920, vec![90_112..180_224])]
#[case::long_head_kept(10..140_000, 65_536, vec![10..65_536, 65_536..140_000])]
#[case::both_edges(60_000..140_000, 65_536, vec![60_000..140_000])]
#[case::interior_kept(60_000..270_000, 65_536, vec![60_000..131_072, 131_072..196_608, 196_608..270_000])]
#[case::single(5..100, 65_536, vec![5..100])]
#[case::empty(5..5, 65_536, vec![])]
fn scans_join_short_edge_chunks(
    #[case] rows: Range<u64>,
    #[case] rows_per_chunk: usize,
    #[case] expected: Vec<Range<u64>>,
) {
    assert_eq!(scan_chunks(rows, rows_per_chunk), expected);
}

#[rstest]
#[case::empty(0, 0, 1_048_576)]
#[case::single_chunk(1000, 100_000_000, 1_048_576)]
#[case::no_bytes(100_000, 0, 1_048_576)]
#[case::small_rows(1_000_000, 64_000_000, 131_072)]
#[case::wide_small_file(65_536, 128_000_000, 8192)]
#[case::large_rows(1_000_000, 384_000_000, 16_384)]
#[case::minimum(1_000_000, 2_000_000_000, 8192)]
fn chunk_byte_budget(#[case] rows: u64, #[case] bytes: u32, #[case] expected: usize) {
    let locations = [SegmentLocation {
        offset: 0,
        length: bytes,
        alignment: Alignment::none(),
    }];
    assert_eq!(rows_per_chunk(rows, &locations), expected);
}

#[test]
fn smaller_chunks_keep_file_alignment() {
    assert_eq!(
        row_chunks(7500..33_000, 16_384),
        [7500..16_384, 16_384..32_768, 32_768..33_000]
    );
}

#[tokio::test]
async fn dictionary_values_are_reused_between_splits() -> VortexResult<()> {
    let session = new_session().with_tokio();
    const STORED_CHUNK_ROWS: usize = 70_000;
    const STORED_CHUNKS: usize = 3;
    let segments = Arc::new(TestSegments::default());
    let (mut sequence, eof) = SequenceId::root().split();
    let arrays = (0..STORED_CHUNKS)
        .map(|index| {
            let start = i32::try_from(index * STORED_CHUNK_ROWS)?;
            let end = start + i32::try_from(STORED_CHUNK_ROWS)?;
            Ok((
                sequence.advance(),
                PrimitiveArray::from_iter(start..end).into_array(),
            ))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let dtype = DType::Primitive(vortex_array::dtype::PType::I32, false.into());
    let ints = chunked_flat()
        .write_stream(
            ArrayContext::empty().into(),
            Arc::<TestSegments>::clone(&segments),
            SequentialStreamAdapter::new(dtype, stream::iter(arrays.into_iter().map(Ok)))
                .sendable(),
            eof,
            &session,
        )
        .await?;
    // Codes index the same values across fixed work chunks and stored chunk boundaries.
    let layout = DictLayout::new(Arc::clone(&ints), ints).into_layout();
    let counted = Arc::new(CountingSegmentSource::new(segments));
    let segments = Arc::clone(&counted) as Arc<dyn SegmentSource>;
    let mut file = scan_file(&segments, &layout)?;
    // Without retained segments, so a chunk reads every codes chunk it overlaps.
    file.cache = Some(ScanCache::with_retained_bytes(0));
    let mut scan = ScanBuilder::new(session.clone(), file).prepare()?;
    // Exercise reuse across work chunks even when the default work chunk holds the whole file.
    scan.chunk_rows = 8 * ROW_BLOCK_SIZE;
    let scan = Arc::new(scan);
    let chunks = row_chunks(scan.rows().clone(), scan.chunk_rows);
    assert_eq!(
        chunks.len(),
        (STORED_CHUNKS * STORED_CHUNK_ROWS).div_ceil(scan.chunk_rows)
    );
    for (index, rows) in chunks.into_iter().enumerate() {
        let before = counted.requests();
        let arrays = run_rows(&scan, rows.clone()).await?;
        let reads = counted.requests() - before;
        let code_chunks =
            (rows.end - 1) / STORED_CHUNK_ROWS as u64 - rows.start / STORED_CHUNK_ROWS as u64 + 1;
        // The first chunk reads every values chunk once for the whole scan; its codes are one of
        // them, and the session shares that read. Later chunks read only their codes.
        assert_eq!(
            reads,
            if index == 0 {
                STORED_CHUNKS
            } else {
                usize::try_from(code_chunks)?
            }
        );
        let expected =
            PrimitiveArray::from_iter(i32::try_from(rows.start)?..i32::try_from(rows.end)?);
        assert_arrays_eq!(
            ChunkedArray::try_new(arrays, layout.dtype().clone())?.into_array(),
            expected,
            &mut session.create_execution_ctx()
        );
    }
    Ok(())
}

/// Work chunks that each overlap half of a stored segment read it once while the file's cache
/// retains segments, and once per chunk when it retains nothing.
#[rstest]
#[case::retained(None, 1)]
#[case::not_retained(Some(0), 2)]
#[tokio::test]
async fn chunks_sharing_a_segment_read_it_once(
    #[case] retained_bytes: Option<u64>,
    #[case] reads_per_segment: usize,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, chunked_flat().as_ref()).await?;
    let counted = Arc::new(CountingSegmentSource::new(segments));
    let segments = Arc::clone(&counted) as Arc<dyn SegmentSource>;
    let mut file = scan_file(&segments, &layout)?;
    file.cache =
        Some(retained_bytes.map_or_else(ScanCache::default, ScanCache::with_retained_bytes));
    let projection = get_item("ints", root()).bind(layout.dtype())?;
    let mut scan = ScanBuilder::new(session, file)
        .with_projection(projection)
        .prepare()?;
    let chunk_rows = CHUNK_ROWS / 2;
    scan.chunk_rows = chunk_rows;
    let scan = Arc::new(scan);

    // One chunk at a time, so no read is still in flight when the next chunk starts.
    for rows in row_chunks(scan.rows().clone(), chunk_rows) {
        run_rows(&scan, rows).await?;
    }
    assert_eq!(counted.requests(), CHUNKS * reads_per_segment);
    Ok(())
}

/// V2 rejects list layouts: their element ranges are only known once their offsets are read.
#[tokio::test]
async fn list_layouts_return_an_error() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let strategy = TableStrategy::new(chunked_flat(), chunked_flat()).with_list_layout();
    let (segments, layout) = write(&session, &strategy).await?;
    let error = run_scan(builder(&session, &segments, &layout)?)
        .await
        .expect_err("V2 must reject an unsupported layout");
    assert!(error.to_string().contains("vortex.plan.list_pack"));
    Ok(())
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

#[tokio::test]
async fn every_row_is_returned() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, table().as_ref()).await?;
    let chunks = run_scan(builder(&session, &segments, &layout)?).await?;
    let rows: usize = chunks.iter().map(|chunk| chunk.len()).sum();
    assert_eq!(rows, CHUNKS * CHUNK_ROWS);
    Ok(())
}

#[tokio::test]
async fn primitive_column() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, table().as_ref()).await?;
    let dtype = layout.dtype().clone();
    let actual = collect_scan(
        DType::Primitive(vortex_array::dtype::PType::I32, false.into()),
        builder(&session, &segments, &layout)?
            .with_projection(get_item("ints", root()).bind(&dtype)?)
            .with_row_range(990..1010),
    )
    .await?;
    assert_arrays_eq!(
        actual,
        PrimitiveArray::from_iter(990i32..1010),
        &mut session.create_execution_ctx()
    );
    Ok(())
}

/// Stored batches stream within one work chunk without being concatenated for Arrow conversion.
#[rstest]
#[tokio::test]
async fn stored_batches_share_one_work_chunk(
    #[values(Strategy::Flat, Strategy::Table, Strategy::ZonedTable)] strategy: Strategy,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, strategy.get().as_ref()).await?;
    let scan = Arc::new(builder(&session, &segments, &layout)?.prepare()?);
    assert_eq!(
        row_chunks(scan.rows().clone(), scan.chunk_rows),
        vec![0..(CHUNKS * CHUNK_ROWS) as u64]
    );
    let batches = run_batches(&scan, scan.rows().clone()).await?;
    assert_eq!(batches.len(), CHUNKS);
    assert!(batches.iter().all(|batch| batch.scope.rows == *scan.rows()));
    let arrays = batches
        .into_iter()
        .map(|batch| batch.array)
        .collect::<Vec<_>>();
    let expected = (0..CHUNKS)
        .map(|chunk| table_chunk(chunk * CHUNK_ROWS))
        .collect::<VortexResult<Vec<_>>>()?;
    assert_arrays_eq!(
        ChunkedArray::try_new(arrays, layout.dtype().clone())?,
        ChunkedArray::try_new(expected, layout.dtype().clone())?,
        &mut session.create_execution_ctx()
    );
    Ok(())
}

#[tokio::test]
async fn large_stored_batch_is_not_subdivided_for_projection() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let segments = Arc::new(TestSegments::default());
    let (mut sequence, eof) = SequenceId::root().split();
    let expected = PrimitiveArray::from_iter(0..150_000i32).into_array();
    let layout = FlatLayoutStrategy::default()
        .write_stream(
            ArrayContext::empty().into(),
            Arc::<TestSegments>::clone(&segments),
            SequentialStreamAdapter::new(
                expected.dtype().clone(),
                stream::iter([Ok((sequence.advance(), expected.clone()))]),
            )
            .sendable(),
            eof,
            &session,
        )
        .await?;
    let segments = segments as Arc<dyn SegmentSource>;
    let scan = Arc::new(builder(&session, &segments, &layout)?.prepare()?);
    let batches = run_batches(&scan, scan.rows().clone()).await?;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].scope.rows, 0..150_000);
    assert_arrays_eq!(
        batches[0].array,
        expected,
        &mut session.create_execution_ctx()
    );
    Ok(())
}

/// Writes `0..30` as three stored chunks of ten rows.
async fn chunked_ints(
    session: &VortexSession,
    segments: &Arc<TestSegments>,
) -> VortexResult<LayoutRef> {
    let (mut sequence, eof) = SequenceId::root().split();
    let arrays = (0..3i32)
        .map(|index| {
            Ok((
                sequence.advance(),
                PrimitiveArray::from_iter(index * 10..index * 10 + 10).into_array(),
            ))
        })
        .collect::<Vec<_>>();
    let dtype = DType::Primitive(vortex_array::dtype::PType::I32, false.into());
    chunked_flat()
        .write_stream(
            ArrayContext::empty().into(),
            Arc::<TestSegments>::clone(segments),
            SequentialStreamAdapter::new(dtype, stream::iter(arrays)).sendable(),
            eof,
            session,
        )
        .await
}

/// A shared [`ScanCache`] keeps dictionary values between scans; without one, each scan reads
/// them again.
#[rstest]
#[case::cached(true, 1)]
#[case::uncached(false, 4)]
#[tokio::test]
async fn scan_cache_reuses_values_between_scans(
    #[case] cached: bool,
    #[case] second_scan_reads: usize,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let segments = Arc::new(TestSegments::default());
    let values = chunked_ints(&session, &segments).await?;
    let codes = chunked_ints(&session, &segments).await?;
    let layout = DictLayout::new(values, codes).into_layout();
    let counted = Arc::new(CountingSegmentSource::new(segments));
    let segments = Arc::clone(&counted) as Arc<dyn SegmentSource>;
    let mut file = scan_file(&segments, &layout)?;
    // Without retained segments, so only the values come from the cache.
    file.cache = cached.then(|| ScanCache::with_retained_bytes(0));

    let mut reads = Vec::new();
    for _ in 0..2 {
        let before = counted.requests();
        let scan = ScanBuilder::new(session.clone(), file.clone())
            .with_row_indices(StrictSortedBuffer::try_new(Buffer::from(vec![12u64, 15]))?);
        let actual = collect_scan(layout.dtype().clone(), scan).await?;
        assert_arrays_eq!(
            actual,
            PrimitiveArray::from_iter([12i32, 15]),
            &mut session.create_execution_ctx()
        );
        reads.push(counted.requests() - before);
    }
    // Every values chunk and the one selected codes chunk, then only the codes chunk if cached.
    assert_eq!(reads, vec![4, second_scan_reads]);
    Ok(())
}

#[tokio::test]
async fn selected_split_scope_is_bounded_before_projection() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, table().as_ref()).await?;
    let scan = Arc::new(
        builder(&session, &segments, &layout)?
            .with_row_indices(StrictSortedBuffer::try_new(Buffer::from(vec![7, 2001]))?)
            .prepare()?,
    );
    let mut chunks = ChunkPlanner::new(
        Arc::clone(&scan),
        scan.rows().clone(),
        scan.selection().clone(),
        None,
    );
    let PlannerOutput::Spawn(scope, _) = chunks.compute()? else {
        vortex_bail!("the selected rows were not planned");
    };
    assert_eq!(scope.rows, 7..2002);
    assert!(matches!(chunks.compute()?, PlannerOutput::Done));
    Ok(())
}

/// A repeated projection reuses its optimized plan, so later scans find the dictionary values the
/// first one kept instead of reading them again.
#[tokio::test]
async fn repeated_projections_reuse_their_values() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let segments = Arc::new(TestSegments::default());
    let values = chunked_ints(&session, &segments).await?;
    let codes = chunked_ints(&session, &segments).await?;
    let layout = DictLayout::new(values, codes).into_layout();
    let counted = Arc::new(CountingSegmentSource::new(segments));
    let segments = Arc::clone(&counted) as Arc<dyn SegmentSource>;
    let mut file = scan_file(&segments, &layout)?;
    // Without retained segments, so only the values come from the cache.
    file.cache = Some(ScanCache::with_retained_bytes(0));

    let projection = gt(root(), lit(15i32)).bind(layout.dtype())?;
    let mut reads = Vec::new();
    for _ in 0..3 {
        let before = counted.requests();
        let scan = ScanBuilder::new(session.clone(), file.clone())
            .with_projection(projection.clone())
            .with_row_indices(StrictSortedBuffer::try_new(Buffer::from(vec![12u64, 18]))?);
        let actual = collect_scan(projection.dtype().clone(), scan).await?;
        assert_arrays_eq!(
            actual,
            vortex_array::arrays::BoolArray::from_iter([false, true]),
            &mut session.create_execution_ctx()
        );
        reads.push(counted.requests() - before);
    }
    assert_eq!(reads, vec![4, 1, 1]);
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

/// Pruning runs once before any chunk is planned: rows every pruning plan excludes are never
/// given a chunk, and a scan whose zones all fail plans nothing.
#[rstest]
#[case::tail(2_999, Some(3_000..4_000))]
#[case::nothing(10_000, None)]
#[tokio::test]
async fn pruned_rows_get_no_chunk(
    #[case] above: i32,
    #[case] expected: Option<Range<u64>>,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, zoned_table().as_ref()).await?;
    let scan = Arc::new(
        builder(&session, &segments, &layout)?
            .with_some_filter(Some(
                gt(get_item("ints", root()), lit(above)).bind(layout.dtype())?,
            ))
            .prepare()?,
    );
    let batches = run_batches(&scan, scan.rows().clone()).await?;
    assert_eq!(
        batches
            .iter()
            .map(|batch| batch.scope.rows.clone())
            .collect::<Vec<_>>(),
        expected.into_iter().collect::<Vec<_>>()
    );
    Ok(())
}

/// Runs only the pruning plans of `scan` over `rows`, returning the rows they kept.
async fn prune(scan: &Arc<PreparedScan>, rows: Range<u64>) -> VortexResult<Option<Mask>> {
    let io = scan.io();
    let kept = Arc::new(parking_lot::Mutex::new(None));
    let next = {
        let scan = Arc::clone(scan);
        let kept = Arc::clone(&kept);
        move |rows_kept: Option<Mask>| -> Box<dyn Planner> {
            *kept.lock() = rows_kept;
            let selection = scan.selection().clone();
            Box::new(ChunkPlanner::new(scan, 0..0, selection, None))
        }
    };
    let mut run = Run::new();
    run.admit(
        Box::new(PrunePlanner::new(
            Arc::clone(scan),
            rows.clone(),
            Box::new(next),
        )),
        WorkScope {
            file_ordinal: 0,
            rows,
        },
        io.session(),
    );
    loop {
        match run.advance()? {
            Progress::Batch(_) => vortex_bail!("pruning alone produced a batch"),
            Progress::Spawn(scope, planner) => {
                run.admit(planner, scope, io.session());
            }
            Progress::RootDone(_) if run.live_roots() > 0 => {}
            Progress::RootDone(_) | Progress::Idle => return Ok(kept.lock().take()),
            Progress::Waiting => std::future::poll_fn(|cx| run.poll_completion(cx)).await?,
        }
    }
}

/// Pruning some of a scan's rows again, as an engine does once a dynamic filter has tightened,
/// prunes with the tightened value from the zone maps the first pruning read: it reads no segment,
/// even when nothing is retained.
#[tokio::test]
async fn pruning_again_reuses_the_first_reads() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, zoned_table().as_ref()).await?;
    let counted = Arc::new(CountingSegmentSource::new(segments));
    let segments = Arc::clone(&counted) as Arc<dyn SegmentSource>;
    let mut file = scan_file(&segments, &layout)?;
    file.cache = Some(ScanCache::with_retained_bytes(0));
    let bound = Arc::new(AtomicI32::new(-1));
    let ints = get_item("ints", root());
    let filter = {
        let bound = Arc::clone(&bound);
        dynamic(
            CompareOperator::Gt,
            move || Some(bound.load(Ordering::Relaxed).into()),
            ints.return_dtype(layout.dtype())?,
            true,
            ints,
        )
    };
    let scan = Arc::new(
        ScanBuilder::new(session.clone(), file)
            .with_some_filter(Some(filter.bind(layout.dtype())?))
            .prepare()?,
    );

    let all = prune(&scan, scan.rows().clone()).await?;
    let reads = counted.requests();
    assert!(reads > 0, "the first pruning reads the zone maps");
    assert_eq!(all.map(|kept| kept.true_count()), Some(4_000));

    bound.store(2_999, Ordering::Relaxed);
    let part = prune(&scan, 1_000..4_000).await?;
    assert_eq!(part.map(|kept| kept.true_count()), Some(1_000));
    assert_eq!(counted.requests(), reads);
    Ok(())
}

/// Scans of one file that share a cache share their filter plans, so the pruning plans run once:
/// the second scan with the same filter re-reads the data it needs but never the zone maps. Each
/// scan binds its own filter and projection, as each partition of an engine does.
#[tokio::test]
async fn shared_cache_prunes_zones_once() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, zoned_table().as_ref()).await?;
    let counted = Arc::new(CountingSegmentSource::new(segments));
    let segments = Arc::clone(&counted) as Arc<dyn SegmentSource>;
    let mut file = scan_file(&segments, &layout)?;
    // Without retained segments, so only the proved zones come from the cache.
    let cache = ScanCache::with_retained_bytes(0);
    file.cache = Some(cache.clone());
    let filter = || gt(get_item("ints", root()), lit(2_999i32)).bind(layout.dtype());
    let projection = || get_item("ints", root()).bind(layout.dtype());
    assert!(
        cache
            .filter_plans(&filter()?, || vortex_bail!("the cache should be empty"))
            .is_err()
    );

    let mut reads = Vec::new();
    for _ in 0..2 {
        let before = counted.requests();
        let projection = projection()?;
        let scan = ScanBuilder::new(session.clone(), file.clone())
            .with_projection(projection.clone())
            .with_some_filter(Some(filter()?));
        let actual = collect_scan(projection.dtype().clone(), scan).await?;
        assert_eq!(actual.len(), 1_000);
        reads.push(counted.requests() - before);
    }
    let plans = cache.filter_plans(&filter()?, || {
        vortex_bail!("the first scan should have kept its filter plans")
    })?;
    assert_eq!(plans.pruning().count(), 1);
    // Each scan reads the one chunk pruning keeps; only the first reads the zone map.
    assert_eq!(reads, vec![2, 1]);
    Ok(())
}

/// Fetch only demanded segments so these tests distinguish avoided reads from eager hints.
struct DemandOnlyIo(Arc<dyn IoSource>);

impl IoSource for DemandOnlyIo {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        self.0.submit(
            owner,
            batch
                .into_iter()
                .filter(|request| request.intent == IoIntent::Fetch)
                .collect(),
        )
    }
    fn poll(&self) -> VortexResult<Option<Completion>> {
        self.0.poll()
    }
    fn poll_completion(&self, cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        self.0.poll_completion(cx)
    }
    fn wait(&self) -> VortexResult<Completion> {
        self.0.wait()
    }
    fn release(&self, owner: IoOwnerId) {
        self.0.release(owner);
    }
    fn clear(&self) {
        self.0.clear();
    }
}

#[rstest]
#[case::already_pruned(true, 1)]
#[case::not_yet_pruned(false, 2)]
#[tokio::test]
async fn dynamic_threshold_reprunes_a_queued_chunk(
    #[case] initially_pruned: bool,
    #[case] expected_reads: usize,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, zoned_table().as_ref()).await?;
    let counted = Arc::new(CountingSegments {
        inner: segments,
        reads: AtomicUsize::new(0),
    });
    let segments = Arc::clone(&counted) as Arc<dyn SegmentSource>;
    let threshold = Arc::new(AtomicI32::new(i32::MAX));
    let bound = Arc::clone(&threshold);
    let filter = dynamic(
        CompareOperator::Lt,
        move || Some(bound.load(Ordering::Relaxed).into()),
        DType::Primitive(vortex_array::dtype::PType::I32, false.into()),
        true,
        get_item("ints", root()),
    )
    .bind(layout.dtype())?;
    let scan = Arc::new(
        builder(&session, &segments, &layout)?
            .with_projection(get_item("ints", root()).bind(layout.dtype())?)
            .with_some_filter(Some(filter))
            .prepare()?,
    );
    let io = scan.io();
    let mut run = Run::new();
    let root = if initially_pruned {
        compose(&scan, scan.rows().clone())
    } else {
        Box::new(ChunkPlanner::new(
            Arc::clone(&scan),
            scan.rows().clone(),
            scan.selection().clone(),
            None,
        ))
    };
    run.admit(
        root,
        WorkScope {
            file_ordinal: 0,
            rows: scan.rows().clone(),
        },
        Arc::new(DemandOnlyIo(io.session())),
    );
    let (scope, chunk) = loop {
        match run.advance()? {
            Progress::Spawn(scope, chunk) => break (scope, chunk),
            Progress::Waiting => std::future::poll_fn(|cx| run.poll_completion(cx)).await?,
            Progress::RootDone(_) => {}
            _ => vortex_bail!("expected pruning to emit a chunk"),
        }
    };
    let before = counted.reads.load(Ordering::Relaxed);
    threshold.store(500, Ordering::Relaxed);
    run.admit(chunk, scope, Arc::new(DemandOnlyIo(io.session())));
    let mut arrays = Vec::new();
    loop {
        match run.advance()? {
            Progress::Batch(batch) => arrays.push(batch.array),
            Progress::Spawn(scope, chunk) => {
                run.admit(chunk, scope, Arc::new(DemandOnlyIo(io.session())));
            }
            Progress::Waiting => std::future::poll_fn(|cx| run.poll_completion(cx)).await?,
            Progress::RootDone(_) if run.live_roots() > 0 => {}
            Progress::RootDone(_) | Progress::Idle => break,
        }
    }
    assert_arrays_eq!(
        ChunkedArray::try_new(
            arrays,
            DType::Primitive(vortex_array::dtype::PType::I32, false.into())
        )?,
        PrimitiveArray::from_iter(0..500i32),
        &mut session.create_execution_ctx()
    );
    assert_eq!(
        counted.reads.load(Ordering::Relaxed) - before,
        expected_reads,
        "tightened zones must be excluded before reading data"
    );
    Ok(())
}

struct TighteningSegments {
    inner: Arc<dyn SegmentSource>,
    threshold: Arc<AtomicI32>,
    trigger: Vec<SegmentId>,
}

impl SegmentSource for TighteningSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        if self.trigger.contains(&id) {
            self.threshold.store(500, Ordering::Relaxed);
        }
        self.inner.request(id)
    }
}

#[tokio::test]
async fn dynamic_threshold_refilters_after_another_conjunct() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, zoned_table().as_ref()).await?;
    let strings = layout
        .slot(2)?
        .ok_or_else(|| vortex_err!("strings column missing"))?;
    let data = strings
        .slot(0)?
        .ok_or_else(|| vortex_err!("strings data missing"))?;
    let mut trigger = Vec::new();
    for child in data.depth_first_traversal() {
        trigger.extend(child?.segment_ids());
    }
    let threshold = Arc::new(AtomicI32::new(3500));
    let bound = Arc::clone(&threshold);
    let segments = Arc::new(TighteningSegments {
        inner: segments,
        threshold: Arc::clone(&threshold),
        trigger,
    }) as Arc<dyn SegmentSource>;
    let dynamic = dynamic(
        CompareOperator::Lt,
        move || Some(bound.load(Ordering::Relaxed).into()),
        DType::Primitive(vortex_array::dtype::PType::I32, false.into()),
        true,
        get_item("ints", root()),
    );
    let filter = and(dynamic, gt(get_item("strings", root()), lit(""))).bind(layout.dtype())?;
    let scan = Arc::new(
        builder(&session, &segments, &layout)?
            .with_projection(get_item("ints", root()).bind(layout.dtype())?)
            .with_some_filter(Some(filter))
            .prepare()?,
    );
    let io = scan.io();
    let mut run = Run::new();
    run.admit(
        compose(&scan, scan.rows().clone()),
        WorkScope {
            file_ordinal: 0,
            rows: scan.rows().clone(),
        },
        Arc::new(DemandOnlyIo(io.session())),
    );
    let mut arrays = Vec::new();
    loop {
        match run.advance()? {
            Progress::Batch(batch) => arrays.push(batch.array),
            Progress::Spawn(scope, chunk) => {
                run.admit(chunk, scope, Arc::new(DemandOnlyIo(io.session())));
            }
            Progress::Waiting => std::future::poll_fn(|cx| run.poll_completion(cx)).await?,
            Progress::RootDone(_) if run.live_roots() > 0 => {}
            Progress::RootDone(_) | Progress::Idle => break,
        }
    }
    assert_eq!(threshold.load(Ordering::Relaxed), 500);
    assert_arrays_eq!(
        ChunkedArray::try_new(
            arrays,
            DType::Primitive(vortex_array::dtype::PType::I32, false.into())
        )?,
        PrimitiveArray::from_iter((0..500i32).filter(|row| row % 3 != 0)),
        &mut session.create_execution_ctx()
    );
    Ok(())
}
