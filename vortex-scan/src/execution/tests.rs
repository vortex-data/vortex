// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::Ordering;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::dynamic;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::execution::SegmentLocation;
use crate::execution::row_chunks;
use crate::execution::scan_chunks;
use crate::execution::*;
use crate::plan::ConcatPlan;
use crate::plan::PlanRef;
use crate::plan::ZonedPlan;
use crate::planning::driver::Batch;
use crate::planning::driver::Progress;
use crate::planning::driver::Run;
use crate::planning::planner::Planner;
use crate::planning::planner::PlannerOutput;
use crate::planning::planner::WorkScope;
use crate::prepare::ScanBuilder;
use crate::prepare::filtered_rows_per_chunk;
use crate::source::ScanInput;
use crate::strict_sorted_buffer::StrictSortedBuffer;
use crate::test::SESSION;
use crate::test::Store;
use crate::test::dict;
use crate::test::struct_plan;

const CHUNK_ROWS: usize = 1000;
const CHUNKS: usize = 4;

fn new_session() -> VortexSession {
    VortexSession::empty()
}

#[derive(Clone, Copy, Debug)]
enum Strategy {
    Flat,
    Table,
    ZonedTable,
}

async fn source(
    session: &VortexSession,
    strategy: Strategy,
) -> VortexResult<(Arc<Store>, PlanRef)> {
    source_chunks(session, strategy, CHUNKS).await
}

async fn source_chunks(
    _session: &VortexSession,
    strategy: Strategy,
    count: usize,
) -> VortexResult<(Arc<Store>, PlanRef)> {
    let store = Arc::new(Store::default());
    let arrays = (0..count)
        .map(|chunk| table_chunk(chunk * CHUNK_ROWS))
        .collect::<VortexResult<Vec<_>>>()?;
    let dtype = arrays[0].dtype().clone();
    let plan = match strategy {
        Strategy::Flat => {
            let plans = arrays
                .iter()
                .map(|array| store.flat(array))
                .collect::<VortexResult<_>>()?;
            ConcatPlan::try_new(dtype, plans)?.into_plan()
        }
        Strategy::Table | Strategy::ZonedTable => {
            let fields = dtype.as_struct_fields();
            let columns = fields
                .names()
                .iter()
                .map(|name| {
                    let chunks = arrays
                        .iter()
                        .map(|array| {
                            let array = array
                                .clone()
                                .execute::<StructArray>(&mut SESSION.create_execution_ctx())?;
                            store.flat(array.unmasked_field_by_name(name)?)
                        })
                        .collect::<VortexResult<_>>()?;
                    let column_dtype = fields.field(name).expect("fixture field exists");
                    let data = ConcatPlan::try_new(column_dtype, chunks)?.into_plan();
                    if matches!(strategy, Strategy::ZonedTable) && name.as_ref() == "ints" {
                        let max = Max.bind(NumericalAggregateOpts::skip_nans());
                        let maxima = (0..count)
                            .map(|chunk| i32::try_from((chunk + 1) * CHUNK_ROWS - 1).map(Some))
                            .collect::<Result<Vec<_>, _>>()?;
                        let maxima = PrimitiveArray::from_option_iter(maxima).into_array();
                        let table =
                            StructArray::from_fields(&[(max.to_string().as_str(), maxima)])?
                                .into_array();
                        let zones = store.flat(&table)?;
                        Ok(
                            ZonedPlan::new(data, zones, CHUNK_ROWS as u64, vec![max].into())
                                .into_plan(),
                        )
                    } else {
                        Ok(data)
                    }
                })
                .collect::<VortexResult<Vec<_>>>()?;
            struct_plan((count * CHUNK_ROWS) as u64, dtype, columns)
        }
    };
    Ok((store, plan))
}

fn scan_input(store: &Arc<Store>, plan: &PlanRef) -> VortexResult<ScanInput> {
    Ok(store.input(plan.clone()))
}

fn builder(
    session: &VortexSession,
    store: &Arc<Store>,
    plan: &PlanRef,
) -> VortexResult<ScanBuilder> {
    Ok(ScanBuilder::new(session.clone(), store.input(plan.clone())))
}

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

/// Prepares and runs a whole scan, returning its rows as one chunked array.
async fn collect_scan(dtype: DType, builder: ScanBuilder) -> VortexResult<ArrayRef> {
    Ok(ChunkedArray::try_new(run_scan(builder).await?, dtype)?.into_array())
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
#[case::empty(0, 0, 65_536)]
#[case::single_chunk(1000, 100_000_000, 65_536)]
#[case::no_bytes(100_000, 0, 65_536)]
#[case::small_rows(1_000_000, 64_000_000, 65_536)]
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
    let segments = Arc::new(Store::default());
    let arrays = (0..STORED_CHUNKS)
        .map(|index| {
            let start = i32::try_from(index * STORED_CHUNK_ROWS)?;
            let end = start + i32::try_from(STORED_CHUNK_ROWS)?;
            segments.flat(&PrimitiveArray::from_iter(start..end).into_array())
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let dtype = DType::Primitive(vortex_array::dtype::PType::I32, false.into());
    let ints = ConcatPlan::try_new(dtype, arrays)?.into_plan();
    // Codes index the same values across fixed work chunks and stored chunk boundaries.
    let layout = dict(ints.clone(), ints);
    let counted = Arc::clone(&segments);
    let segments = Arc::clone(&counted);
    let mut file = scan_input(&segments, &layout)?;
    // Without retained segments, so a chunk reads every codes chunk it overlaps.
    file.cache = Some(ScanCache::with_retained_bytes(0));
    let scan = Arc::new(ScanBuilder::new(session.clone(), file).prepare()?);
    let chunks = row_chunks(scan.rows().clone(), scan.chunk_rows);
    assert_eq!(
        chunks.len(),
        (STORED_CHUNKS * STORED_CHUNK_ROWS).div_ceil(MAX_ROWS_PER_CHUNK)
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
    let (segments, layout) = source(&session, Strategy::Flat).await?;
    let counted = Arc::clone(&segments);
    let segments = Arc::clone(&counted);
    let mut file = scan_input(&segments, &layout)?;
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

#[tokio::test]
async fn every_row_is_returned() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = source(&session, Strategy::Table).await?;
    let chunks = run_scan(builder(&session, &segments, &layout)?).await?;
    let rows: usize = chunks.iter().map(|chunk| chunk.len()).sum();
    assert_eq!(rows, CHUNKS * CHUNK_ROWS);
    Ok(())
}

#[tokio::test]
async fn primitive_column() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = source(&session, Strategy::Table).await?;
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

/// Stored chunk boundaries do not subdivide a fixed work chunk.
#[rstest]
#[tokio::test]
async fn work_chunks_ignore_layout_boundaries(
    #[values(Strategy::Flat, Strategy::Table, Strategy::ZonedTable)] strategy: Strategy,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = source(&session, strategy).await?;
    let scan = Arc::new(builder(&session, &segments, &layout)?.prepare()?);
    assert_eq!(
        row_chunks(scan.rows().clone(), scan.chunk_rows),
        vec![0..(CHUNKS * CHUNK_ROWS) as u64]
    );
    let arrays = run_rows(&scan, scan.rows().clone()).await?;
    assert_eq!(arrays.len(), 1);
    assert_eq!(arrays[0].len(), CHUNKS * CHUNK_ROWS);
    Ok(())
}

async fn chunked_ints(_session: &VortexSession, segments: &Arc<Store>) -> VortexResult<PlanRef> {
    let arrays = (0..3i32)
        .map(|index| {
            segments.flat(&PrimitiveArray::from_iter(index * 10..index * 10 + 10).into_array())
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let dtype = DType::Primitive(vortex_array::dtype::PType::I32, false.into());
    Ok(ConcatPlan::try_new(dtype, arrays)?.into_plan())
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
    let segments = Arc::new(Store::default());
    let values = chunked_ints(&session, &segments).await?;
    let codes = chunked_ints(&session, &segments).await?;
    let layout = dict(values, codes);
    let counted = Arc::clone(&segments);
    let segments = Arc::clone(&counted);
    let mut file = scan_input(&segments, &layout)?;
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
    let (segments, layout) = source(&session, Strategy::Table).await?;
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
    let segments = Arc::new(Store::default());
    let values = chunked_ints(&session, &segments).await?;
    let codes = chunked_ints(&session, &segments).await?;
    let layout = dict(values, codes);
    let counted = Arc::clone(&segments);
    let segments = Arc::clone(&counted);
    let mut file = scan_input(&segments, &layout)?;
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
    let (segments, layout) = source(&session, Strategy::ZonedTable).await?;
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

/// A scan whose filter is `ints > bound`, a dynamic comparison, over `chunks` table chunks whose
/// segment reads are counted.
async fn dynamic_scan(
    chunks: usize,
    bound: &Arc<AtomicI32>,
) -> VortexResult<(Arc<PreparedScan>, Arc<Store>)> {
    let session = new_session().with_tokio();
    let (segments, layout) = source_chunks(&session, Strategy::ZonedTable, chunks).await?;
    let counted = Arc::clone(&segments);
    let segments = Arc::clone(&counted);
    let mut file = scan_input(&segments, &layout)?;
    file.cache = Some(ScanCache::with_retained_bytes(0));
    let ints = get_item("ints", root());
    let filter = {
        let bound = Arc::clone(bound);
        dynamic(
            CompareOperator::Gt,
            move || Some(bound.load(Ordering::Relaxed).into()),
            ints.return_dtype(layout.dtype())?,
            true,
            ints,
        )
    };
    let scan = ScanBuilder::new(session, file)
        .with_some_filter(Some(filter.bind(layout.dtype())?))
        .prepare()?;
    Ok((Arc::new(scan), counted))
}

/// Pruning some of a scan's rows again, as an engine does once a dynamic filter has tightened,
/// prunes with the tightened value from the zone maps the first pruning read: it reads no segment,
/// even when nothing is retained.
#[tokio::test]
async fn pruning_again_reuses_the_first_reads() -> VortexResult<()> {
    // Both prunings cover more rows than one filtered chunk, so neither skips the dynamic plan.
    let bound = Arc::new(AtomicI32::new(-1));
    let (scan, counted) = dynamic_scan(200, &bound).await?;

    let all = prune(&scan, scan.rows().clone()).await?;
    let reads = counted.requests();
    assert!(reads > 0, "the first pruning reads the zone maps");
    assert_eq!(all.map(|kept| kept.true_count()), Some(200_000));

    bound.store(2_999, Ordering::Relaxed);
    let part = prune(&scan, 1_000..151_000).await?;
    assert_eq!(part.map(|kept| kept.true_count()), Some(148_000));
    assert_eq!(counted.requests(), reads);
    Ok(())
}

/// A scan of at most one filtered chunk skips the pruning plan of a dynamic conjunct and reads no
/// zone map: its chunk evaluates the conjunct next.
#[tokio::test]
async fn small_scans_skip_dynamic_pruning() -> VortexResult<()> {
    let bound = Arc::new(AtomicI32::new(2_999));
    let (scan, counted) = dynamic_scan(CHUNKS, &bound).await?;

    assert!(prune(&scan, scan.rows().clone()).await?.is_none());
    assert_eq!(counted.requests(), 0);
    Ok(())
}

/// Runs a scan of the `strings` column of `TABLE_CHUNKS` table chunks, filtered on `ints > bound`
/// with a dynamic comparison when there is a bound, over `counted`. Returns the rows it kept and
/// the segment requests it made.
async fn strings_scan(
    session: &VortexSession,
    counted: &Arc<Store>,
    layout: &PlanRef,
    file: ScanInput,
    filter: Option<&Expression>,
    skip_unselective_dynamic: bool,
) -> VortexResult<(Arc<PreparedScan>, usize, usize)> {
    let projection = get_item("strings", root()).bind(layout.dtype())?;
    let scan = Arc::new(
        ScanBuilder::new(session.clone(), file)
            .with_projection(projection)
            .with_some_filter(filter.map(|f| f.bind(layout.dtype())).transpose()?)
            .with_skip_unselective_dynamic(skip_unselective_dynamic)
            .prepare()?,
    );
    let before = counted.requests();
    let rows = run_rows(&scan, scan.rows().clone())
        .await?
        .iter()
        .map(|array| array.len())
        .sum();
    Ok((scan, rows, counted.requests() - before))
}

/// A dynamic conjunct that keeps every row on enough chunks is switched off: later scans neither
/// prune its zones nor read its column, so they read no more than an unfiltered scan. A new dynamic
/// value switches it back on. Without the opt-in it keeps being evaluated.
#[rstest]
#[case::opted_in(true)]
#[case::default(false)]
#[tokio::test]
async fn unselective_dynamic_conjuncts_switch_off_until_their_value_changes(
    #[case] skip: bool,
) -> VortexResult<()> {
    // Five filtered chunks per scan.
    const TABLE_CHUNKS: usize = 600;
    let session = new_session().with_tokio();
    let (segments, layout) = source_chunks(&session, Strategy::ZonedTable, TABLE_CHUNKS).await?;
    let counted = Arc::clone(&segments);
    let segments = Arc::clone(&counted);
    let file = || -> VortexResult<ScanInput> {
        let mut file = scan_input(&segments, &layout)?;
        file.cache = Some(ScanCache::with_retained_bytes(0));
        Ok(file)
    };
    let shared = file()?;

    let (_, all, unfiltered_reads) =
        strings_scan(&session, &counted, &layout, file()?, None, skip).await?;
    assert_eq!(all, TABLE_CHUNKS * CHUNK_ROWS);

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

    // The first scan evaluates the conjunct on every chunk, which keeps every row.
    let (scan, rows, first_reads) = strings_scan(
        &session,
        &counted,
        &layout,
        shared.clone(),
        Some(&filter),
        skip,
    )
    .await?;
    assert_eq!(rows, all);
    assert!(first_reads > unfiltered_reads);
    let plans = scan.filter().expect("the scan has a filter");
    assert_eq!(plans.is_switched_off(0), skip);

    let (_, rows, reads) = strings_scan(
        &session,
        &counted,
        &layout,
        shared.clone(),
        Some(&filter),
        skip,
    )
    .await?;
    assert_eq!(rows, all);
    // Filtered and unfiltered scans cut chunks differently, so a switched-off scan can read a few
    // segments fewer than an unfiltered one, but never the filtered column.
    if skip {
        assert!(reads <= unfiltered_reads, "{reads} > {unfiltered_reads}");
    } else {
        assert!(reads > unfiltered_reads);
    }

    // A tightened bound is a new dynamic value, so the conjunct is evaluated again. It now
    // rejects half the rows, so it stays switched on.
    bound.store(299_999, Ordering::Relaxed);
    let (scan, rows, _) =
        strings_scan(&session, &counted, &layout, shared, Some(&filter), skip).await?;
    assert_eq!(rows, all - 300_000);
    let plans = scan.filter().expect("the scan has a filter");
    assert!(!plans.is_switched_off(0));
    Ok(())
}

/// Scans of one file that share a cache share their filter plans, so the pruning plans run once:
/// the second scan with the same filter re-reads the data it needs but never the zone maps. Each
/// scan binds its own filter and projection, as each partition of an engine does.
#[tokio::test]
async fn shared_cache_prunes_zones_once() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = source(&session, Strategy::ZonedTable).await?;
    let counted = Arc::clone(&segments);
    let segments = Arc::clone(&counted);
    let mut file = scan_input(&segments, &layout)?;
    // Without retained segments, so only the proved zones come from the cache.
    let cache = ScanCache::with_retained_bytes(0);
    file.cache = Some(cache.clone());
    let filter = || gt(get_item("ints", root()), lit(2_999i32)).bind(layout.dtype());
    let projection = || get_item("ints", root()).bind(layout.dtype());
    assert!(cache.filter_plans(&filter()?).is_none());

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
    let plans = cache
        .filter_plans(&filter()?)
        .expect("the first scan kept its filter plans");
    assert_eq!(plans.pruning().count(), 1);
    // Each scan reads the one chunk pruning keeps; only the first reads the zone map.
    assert_eq!(reads, vec![2, 1]);
    Ok(())
}
