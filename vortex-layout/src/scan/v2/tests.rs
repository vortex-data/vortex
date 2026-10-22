// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::FutureExt;
use futures::TryStreamExt;
use futures::channel::oneshot;
use futures::future::Shared;
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
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowSessionExt;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::planning::driver::Batch;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;

use super::MAX_ROWS_PER_CHUNK;
use super::ROW_BLOCK_SIZE;
use super::ScanBuilder;
use super::ScanCache;
use super::ScanFile;
use super::ScanRun;
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
use crate::scan::execution::SegmentLocation;
use crate::scan::execution::row_chunks;
use crate::scan::scan_builder::ScanBuilder as V1ScanBuilder;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
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
) -> VortexResult<ScanBuilder<ArrayRef>> {
    Ok(ScanBuilder::new(
        session.clone(),
        scan_file(segments, layout)?,
    ))
}

/// Runs `rows` of a prepared scan on a planning run of its own and returns its batches, in row
/// order, each with the rows it stands for.
async fn run_batches<A: 'static + Send>(
    scan: &ScanRun<A>,
    rows: Range<u64>,
) -> VortexResult<Vec<Batch>> {
    let mut run = Run::new();
    run.admit(
        scan.scan.planner(rows.clone(), scan.selection.clone()),
        WorkScope {
            file_ordinal: 0,
            rows,
        },
        scan.io.session(),
    );
    let mut batches = Vec::new();
    loop {
        match run.advance()? {
            Progress::Batch(batch) => batches.push(batch),
            Progress::RootDone(_) | Progress::Idle => break,
            Progress::Waiting => std::future::poll_fn(|cx| run.poll_completion(cx)).await?,
        }
    }
    batches.sort_by_key(|batch| batch.scope.rows.start);
    Ok(batches)
}

/// Runs `rows` of a prepared scan on a planning run of its own and returns its arrays.
async fn run_rows<A: 'static + Send>(
    scan: &ScanRun<A>,
    rows: Range<u64>,
) -> VortexResult<Vec<ArrayRef>> {
    Ok(run_batches(scan, rows)
        .await?
        .into_iter()
        .map(|batch| batch.array)
        .collect())
}

async fn collect(
    dtype: DType,
    stream: impl futures::Stream<Item = VortexResult<ArrayRef>>,
) -> VortexResult<ArrayRef> {
    let chunks: Vec<ArrayRef> = stream.try_collect().await?;
    Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
}

/// Runs `builder` on both executors and checks the planner-driven one returns the same rows.
async fn assert_same_as_v1(
    session: &VortexSession,
    make: impl Fn() -> VortexResult<ScanBuilder<ArrayRef>>,
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
        .with_row_offset(builder.row_offset)
        .with_ordered(builder.ordered);
    if let Some(rows) = builder.row_range.clone() {
        v1 = v1.with_row_range(rows);
    }
    let expected = collect(dtype.clone(), v1.into_stream()?).await?;
    let actual = collect(dtype, builder.into_stream()?).await?;
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
    let schema = Arc::new(session.arrow().to_arrow_schema(scan.dtype())?);
    let batches: Vec<_> = scan.into_record_batch_stream(schema)?.try_collect().await?;
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

struct CountingSegments {
    inner: Arc<dyn SegmentSource>,
    reads: AtomicUsize,
}

impl SegmentSource for CountingSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.inner.request(id)
    }
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
    let counted = Arc::new(CountingSegments {
        inner: segments,
        reads: AtomicUsize::new(0),
    });
    let segments = Arc::clone(&counted) as Arc<dyn SegmentSource>;
    let scan = ScanRun::try_new(builder(&session, &segments, &layout)?)?;
    let chunks = row_chunks(scan.rows.clone(), scan.scan.chunk_rows);
    assert_eq!(
        chunks.len(),
        (STORED_CHUNKS * STORED_CHUNK_ROWS).div_ceil(MAX_ROWS_PER_CHUNK)
    );
    for (index, rows) in chunks.into_iter().enumerate() {
        let before = counted.reads.load(Ordering::Relaxed);
        let arrays = run_rows(&scan, rows.clone()).await?;
        let reads = counted.reads.load(Ordering::Relaxed) - before;
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

/// V2 rejects list layouts: their element ranges are only known once their offsets are read.
#[tokio::test]
async fn list_layouts_return_an_error() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let strategy = TableStrategy::new(chunked_flat(), chunked_flat()).with_list_layout();
    let (segments, layout) = write(&session, &strategy).await?;
    let builder = builder(&session, &segments, &layout)?;
    let error = builder
        .into_stream()?
        .try_next()
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
async fn unordered_returns_every_row() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, table().as_ref()).await?;
    let scan = ScanRun::try_new(builder(&session, &segments, &layout)?.with_ordered(false))?;
    let chunks: Vec<ArrayRef> = scan.execute_stream()?.try_collect().await?;
    let rows: usize = chunks.iter().map(|chunk| chunk.len()).sum();
    assert_eq!(rows, CHUNKS * CHUNK_ROWS);
    Ok(())
}

struct GatedSegments {
    inner: Arc<dyn SegmentSource>,
    blocked: SegmentId,
    release: Shared<oneshot::Receiver<()>>,
}

impl SegmentSource for GatedSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let bytes = self.inner.request(id);
        if id != self.blocked {
            return bytes;
        }
        let release = self.release.clone();
        async move {
            let _ = release.await;
            bytes.await
        }
        .boxed()
    }
}

#[rstest]
#[timeout(Duration::from_secs(5))]
#[tokio::test]
async fn unordered_yields_before_the_split_finishes() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let segments = Arc::new(TestSegments::default());
    let (mut sequence, eof) = SequenceId::root().split();
    let arrays = (0..2)
        .map(|index| {
            let start = i32::try_from(index * MAX_ROWS_PER_CHUNK)?;
            Ok((
                sequence.advance(),
                PrimitiveArray::from_iter(start..start + i32::try_from(MAX_ROWS_PER_CHUNK)?)
                    .into_array(),
            ))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let dtype = arrays[0].1.dtype().clone();
    let layout = chunked_flat()
        .write_stream(
            ArrayContext::empty().into(),
            Arc::<TestSegments>::clone(&segments),
            SequentialStreamAdapter::new(dtype, stream::iter(arrays.into_iter().map(Ok)))
                .sendable(),
            eof,
            &session,
        )
        .await?;
    let blocked = *layout
        .slot(1)?
        .ok_or_else(|| vortex_err!("second chunk is missing"))?
        .segment_ids()
        .first()
        .ok_or_else(|| vortex_err!("second chunk has no segment"))?;
    let (release, gate) = oneshot::channel();
    let segments: Arc<dyn SegmentSource> = Arc::new(GatedSegments {
        inner: segments,
        blocked,
        release: gate.shared(),
    });
    let mut scan = ScanRun::try_new(builder(&session, &segments, &layout)?.with_ordered(false))?;
    // Put both projection morsels in one chunk, as a dense filtered scan would.
    Arc::get_mut(&mut scan.scan)
        .ok_or_else(|| vortex_err!("the prepared scan is not shared yet"))?
        .chunk_rows = usize::try_from(layout.row_count())?;
    let mut batches = scan.execute_stream()?;
    let first = batches.try_next().await;
    let _ = release.send(());
    let first = first?.ok_or_else(|| vortex_err!("first batch is missing"))?;
    assert_arrays_eq!(
        first,
        PrimitiveArray::from_iter(0i32..i32::try_from(MAX_ROWS_PER_CHUNK)?),
        &mut session.create_execution_ctx()
    );
    let rest: Vec<_> = batches.try_collect().await?;
    assert_eq!(
        rest.iter().map(|array| array.len()).sum::<usize>(),
        MAX_ROWS_PER_CHUNK
    );
    Ok(())
}

#[tokio::test]
async fn primitive_column() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, table().as_ref()).await?;
    let dtype = layout.dtype().clone();
    let scan = ScanRun::try_new(
        builder(&session, &segments, &layout)?
            .with_projection(get_item("ints", root()).bind(&dtype)?)
            .with_row_range(990..1010),
    )?;
    let actual = collect(
        DType::Primitive(vortex_array::dtype::PType::I32, false.into()),
        scan.execute_stream()?,
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
    let (segments, layout) = write(&session, strategy.get().as_ref()).await?;
    let scan = ScanRun::try_new(builder(&session, &segments, &layout)?)?;
    assert_eq!(
        row_chunks(scan.rows.clone(), scan.scan.chunk_rows),
        vec![0..(CHUNKS * CHUNK_ROWS) as u64]
    );
    let arrays: Vec<_> = scan.execute_stream()?.try_collect().await?;
    assert_eq!(arrays.len(), 1);
    assert_eq!(arrays[0].len(), CHUNKS * CHUNK_ROWS);
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
    let counted = Arc::new(CountingSegments {
        inner: segments,
        reads: AtomicUsize::new(0),
    });
    let segments = Arc::clone(&counted) as Arc<dyn SegmentSource>;
    let mut file = scan_file(&segments, &layout)?;
    file.cache = cached.then(ScanCache::default);

    let mut reads = Vec::new();
    for _ in 0..2 {
        let before = counted.reads.load(Ordering::Relaxed);
        let scan = ScanBuilder::new(session.clone(), file.clone())
            .with_row_indices(StrictSortedBuffer::try_new(Buffer::from(vec![12u64, 15]))?);
        let actual = collect(layout.dtype().clone(), scan.into_stream()?).await?;
        assert_arrays_eq!(
            actual,
            PrimitiveArray::from_iter([12i32, 15]),
            &mut session.create_execution_ctx()
        );
        reads.push(counted.reads.load(Ordering::Relaxed) - before);
    }
    // Every values chunk and the one selected codes chunk, then only the codes chunk if cached.
    assert_eq!(reads, vec![4, second_scan_reads]);
    Ok(())
}

#[tokio::test]
async fn selected_split_scope_is_bounded_before_projection() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, table().as_ref()).await?;
    let scan = ScanRun::try_new(
        builder(&session, &segments, &layout)?
            .with_row_indices(StrictSortedBuffer::try_new(Buffer::from(vec![7, 2001]))?),
    )?;
    let mut chunks = scan
        .scan
        .chunk_planner(scan.rows.clone(), scan.selection.clone(), None);
    let PlannerOutput::Planner(scope, _) = chunks.compute()? else {
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
    let counted = Arc::new(CountingSegments {
        inner: segments,
        reads: AtomicUsize::new(0),
    });
    let segments = Arc::clone(&counted) as Arc<dyn SegmentSource>;
    let mut file = scan_file(&segments, &layout)?;
    file.cache = Some(ScanCache::default());

    let projection = gt(root(), lit(15i32)).bind(layout.dtype())?;
    let mut reads = Vec::new();
    for _ in 0..3 {
        let before = counted.reads.load(Ordering::Relaxed);
        let scan = ScanBuilder::new(session.clone(), file.clone())
            .with_projection(projection.clone())
            .with_row_indices(StrictSortedBuffer::try_new(Buffer::from(vec![12u64, 18]))?);
        let actual = collect(projection.dtype().clone(), scan.into_stream()?).await?;
        assert_arrays_eq!(
            actual,
            vortex_array::arrays::BoolArray::from_iter([false, true]),
            &mut session.create_execution_ctx()
        );
        reads.push(counted.reads.load(Ordering::Relaxed) - before);
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

/// Zone proofs run once before any chunk is planned: rows every proof excludes are never
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
    let scan = ScanRun::try_new(
        builder(&session, &segments, &layout)?.with_some_filter(Some(
            gt(get_item("ints", root()), lit(above)).bind(layout.dtype())?,
        )),
    )?;
    let batches = run_batches(&scan, scan.rows.clone()).await?;
    assert_eq!(
        batches
            .iter()
            .map(|batch| batch.scope.rows.clone())
            .collect::<Vec<_>>(),
        expected.into_iter().collect::<Vec<_>>()
    );
    Ok(())
}
