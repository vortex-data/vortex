// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
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
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::Lambda;
use vortex_array::expr::and;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::is_null;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::var;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowSessionExt;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_functions::expr::list_length;
use vortex_functions::expr::list_sum;
use vortex_functions::expr::list_transform;
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
use vortex_utils::aliases::hash_map::HashMap;

use super::ChunkPlanner;
use super::PreparedScan;
use super::PrunePlanner;
use super::ROW_BLOCK_SIZE;
use super::ScanBuilder;
use super::ScanFile;
use crate::LayoutChildType;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::list::List;
use crate::layouts::list::writer::ListIndexOptions;
use crate::layouts::repartition::RepartitionStrategy;
use crate::layouts::repartition::RepartitionWriterOptions;
use crate::layouts::row_idx::row_idx;
use crate::layouts::table::TableStrategy;
use crate::layouts::zoned::writer::ZonedLayoutOptions;
use crate::layouts::zoned::writer::ZonedStrategy;
use crate::scan::arrow::to_record_batch;
use crate::scan::scan_builder::ScanBuilder as V1ScanBuilder;
use crate::segments::CountingSegmentSource;
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

/// A table chunk of `CHUNK_ROWS` rows starting at row `start`: an integer, a nullable list of 0
/// to 4 integers, every seventh null, and a list of 0 to 2 nullable lists of 0 to 3 integers.
fn list_chunk(start: usize) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let rows = start..start + CHUNK_ROWS;
    let ints = Buffer::from_iter(
        rows.clone()
            .map(|row| i32::try_from(row).unwrap_or(i32::MAX)),
    );
    let mut offsets = vec![0u32];
    for row in rows.clone() {
        offsets.push(offsets[offsets.len() - 1] + u32::try_from(row % 5).unwrap_or(0));
    }
    let elements = Buffer::from_iter((0..offsets[offsets.len() - 1]).map(|i| i64::from(i) * 3));
    let lists = ListArray::try_new(
        elements.into_array(),
        Buffer::from_iter(offsets).into_array(),
        Validity::from_iter(rows.clone().map(|row| row % 7 != 0)),
        &mut ctx,
    )?;
    let mut outer = vec![0u32];
    let mut inner = vec![0u32];
    let mut inner_valid = Vec::new();
    for row in rows {
        let count = u32::try_from(row % 3).unwrap_or(0);
        for j in 0..count as usize {
            inner.push(inner[inner.len() - 1] + u32::try_from((row + j) % 4).unwrap_or(0));
            inner_valid.push((row + j) % 5 != 0);
        }
        outer.push(outer[outer.len() - 1] + count);
    }
    let inner = ListArray::try_new(
        Buffer::from_iter((0..inner[inner.len() - 1]).map(i64::from)).into_array(),
        Buffer::from_iter(inner).into_array(),
        Validity::from_iter(inner_valid),
        &mut ctx,
    )?;
    let nested = ListArray::try_new(
        inner.into_array(),
        Buffer::from_iter(outer).into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?;
    Ok(StructArray::from_fields(&[
        ("ints", ints.into_array()),
        ("lists", lists.into_array()),
        ("nested", nested.into_array()),
    ])?
    .into_array())
}

/// Leaves repartitioned into zoned blocks of `block_len` rows, so a filter prunes to a block and
/// a list's elements span many segments.
fn small_zoned_leaves(block_len: usize) -> Arc<dyn LayoutStrategy> {
    Arc::new(RepartitionStrategy::new(
        ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: NonZeroUsize::new(block_len).unwrap_or(NonZeroUsize::MIN),
                ..Default::default()
            },
        ),
        RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: block_len,
            block_size_target: None,
            canonicalize: false,
        },
    ))
}

/// The rows of `input` a scan with these options returns.
fn expected_rows(
    input: &ArrayRef,
    row_range: Option<Range<u64>>,
    rows: Option<&[u64]>,
    filter: Option<&Expression>,
    column: Option<&str>,
) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let row_range = row_range.unwrap_or(0..input.len() as u64);
    let mut keep = Mask::from_iter((0..input.len() as u64).map(|row| {
        row_range.contains(&row) && rows.is_none_or(|rows| rows.binary_search(&row).is_ok())
    }));
    if let Some(filter) = filter {
        let matches = input
            .clone()
            .apply(filter)?
            .fill_null(false)?
            .execute::<Mask>(&mut ctx)?;
        keep = &keep & &matches;
    }
    let kept = input.filter(keep)?;
    match column {
        Some(column) => kept.apply(&get_item(column, root())),
        None => Ok(kept),
    }
}

/// The V2 scan reads list layouts, nested or not, under every way a scan selects rows.
#[rstest]
#[case::whole(None, None, None)]
#[case::row_range(Some(250..3_100), None, None)]
#[case::row_indices(None, Some(&[0, 1, 4, 999, 1000, 2500, 3999][..]), None)]
#[case::range_filter(
    None,
    None,
    Some(and(
        gt(get_item("ints", root()), lit(995i32)),
        lt(get_item("ints", root()), lit(2050i32))
    ))
)]
#[case::sparse_filter(None, None, Some(eq(get_item("ints", root()), lit(1001i32))))]
#[case::list_filter(None, None, Some(is_null(get_item("lists", root()))))]
#[case::no_matches(None, None, Some(gt(get_item("ints", root()), lit(10_000i32))))]
#[tokio::test]
async fn list_layouts_return_the_selected_rows(
    #[values(None, Some("lists"), Some("nested"))] column: Option<&str>,
    #[values(None, Some(64))] block_len: Option<usize>,
    #[values(false, true)] indexed: bool,
    #[case] row_range: Option<Range<u64>>,
    #[case] rows: Option<&'static [u64]>,
    #[case] filter: Option<Expression>,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let leaves = block_len.map_or_else(chunked_flat, small_zoned_leaves);
    // With `indexed`, every list is written with an index of its offsets, a window of 16
    // elements apart. Without, every list is one window and has none, as a list written before
    // lists had an index: a chunk then reads every element of the column.
    let strategy =
        TableStrategy::new(chunked_flat(), leaves).with_list_layout_factory(move |list| {
            Arc::new(list.with_index_options(ListIndexOptions {
                window: if indexed { 16 } else { u64::MAX },
                max_entries: usize::MAX,
            }))
        });
    let chunks = (0..CHUNKS)
        .map(|chunk| list_chunk(chunk * CHUNK_ROWS))
        .collect::<VortexResult<Vec<_>>>()?;
    let (segments, layout) = write_chunks(&session, &strategy, chunks).await?;
    let has_index: Vec<bool> = layout
        .depth_first_traversal()
        .filter_map(|layout| layout.ok())
        .filter_map(|layout| layout.as_opt::<List>().map(|list| list.index().is_some()))
        .collect();
    assert!(!has_index.is_empty());
    assert!(has_index.iter().all(|&has_index| has_index == indexed));
    let input = ChunkedArray::try_new(
        (0..CHUNKS)
            .map(|chunk| list_chunk(chunk * CHUNK_ROWS))
            .collect::<VortexResult<Vec<_>>>()?,
        layout.dtype().clone(),
    )?
    .into_array();

    let mut builder = builder(&session, &segments, &layout)?
        .with_some_filter(filter.clone().map(|f| f.bind(layout.dtype())).transpose()?);
    if let Some(column) = column {
        builder = builder.with_projection(get_item(column, root()).bind(layout.dtype())?);
    }
    if let Some(row_range) = row_range.clone() {
        builder = builder.with_row_range(row_range);
    }
    if let Some(rows) = rows {
        builder = builder.with_row_indices(StrictSortedBuffer::try_new(Buffer::copy_from(rows))?);
    }
    let dtype = builder.dtype().clone();
    let actual = collect_scan(dtype, builder).await?;
    let expected = expected_rows(&input, row_range, rows, filter.as_ref(), column)?;
    assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
    Ok(())
}

#[tokio::test]
async fn projection_skips_unselected_list_layouts() -> VortexResult<()> {
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

/// The column each segment of `layout` belongs to, by the path of its children.
fn segment_columns(
    layout: &LayoutRef,
    path: &str,
    columns: &mut HashMap<SegmentId, String>,
) -> VortexResult<()> {
    for id in layout.segment_ids() {
        columns.insert(id, path.to_string());
    }
    for (child_type, child) in layout.child_types().zip(layout.children()?) {
        let child_path = match child_type {
            LayoutChildType::Field(name) if path.is_empty() => name.to_string(),
            LayoutChildType::Field(name) => format!("{path}.{name}"),
            LayoutChildType::Auxiliary(name) => format!("{path}.{name}"),
            LayoutChildType::Chunk(_) | LayoutChildType::Transparent(_) => path.to_string(),
        };
        segment_columns(&child, &child_path, columns)?;
    }
    Ok(())
}

/// `{l: list<struct<a: i32, f: i32, g: list<struct<h: i32, i: i32>>>>, o: i32}`. The five lists
/// have lengths [2, 3, 0, 3, 2], and when `nullable` the second is null over elements that are
/// still stored.
fn transform_table(nullable: bool) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let inner = StructArray::from_fields(&[
        ("h", PrimitiveArray::from_iter(0..15_i32).into_array()),
        ("i", PrimitiveArray::from_iter(100..115_i32).into_array()),
    ])?
    .into_array();
    let g = ListArray::try_new(
        inner,
        Buffer::from_iter([0u32, 1, 3, 3, 4, 6, 6, 7, 9, 9, 15]).into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?
    .into_array();
    let elements = StructArray::from_fields(&[
        ("a", PrimitiveArray::from_iter(0..10_i32).into_array()),
        ("f", PrimitiveArray::from_iter(10..20_i32).into_array()),
        ("g", g),
    ])?
    .into_array();
    let validity = if nullable {
        Validity::from_iter([true, false, true, true, true])
    } else {
        Validity::NonNullable
    };
    let l = ListArray::try_new(
        elements,
        Buffer::from_iter([0u32, 2, 5, 5, 8, 10]).into_array(),
        validity,
        &mut ctx,
    )?
    .into_array();
    Ok(StructArray::from_fields(&[
        ("l", l),
        ("o", PrimitiveArray::from_iter(0..5_i32).into_array()),
    ])?
    .into_array())
}

fn lambda(param: &str, body: Expression) -> Lambda {
    Lambda::new([param], body).vortex_expect("lambda is valid")
}

/// `list_transform(l, x -> body)`.
fn transform(body: Expression) -> Expression {
    list_transform(col("l"), lambda("x", body))
}

/// Leaves repartitioned into two-row chunks, so a read of part of a column skips segments.
fn two_row_chunks() -> Arc<dyn LayoutStrategy> {
    Arc::new(RepartitionStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: 2,
            block_size_target: None,
            canonicalize: true,
        },
    ))
}

const EVERY_LIST_COLUMN: &[&str] = &[
    "l.offsets",
    "l.elements.a",
    "l.elements.f",
    "l.elements.g.offsets",
    "l.elements.g.elements.h",
    "l.elements.g.elements.i",
];

/// An expression over a list reads only the list children it needs: a null check its validity,
/// and its length its offsets. Anything else reads the lists whole, a `list_transform` too: its
/// lambda is evaluated over the lists a chunk assembles.
#[rstest]
#[case::is_null(is_null(col("l")), &[])]
#[case::length(list_length(col("l")), &["l.offsets"])]
#[case::length_in_an_expression(gt(list_length(col("l")), lit(2u64)), &["l.offsets"])]
#[case::length_beside_another_column(
    pack([("n", list_length(col("l"))), ("o", col("o"))], Nullability::NonNullable),
    &["l.offsets", "o"],
)]
#[case::whole(col("l"), EVERY_LIST_COLUMN)]
#[case::transform(transform(get_item("f", var("x"))), EVERY_LIST_COLUMN)]
#[case::transform_under_another_fn(
    list_sum(transform(get_item("f", var("x")))),
    EVERY_LIST_COLUMN
)]
#[case::length_beside_the_whole_list(
    pack([("n", list_length(col("l"))), ("l", col("l"))], Nullability::NonNullable),
    EVERY_LIST_COLUMN,
)]
#[tokio::test]
async fn list_expressions_read_only_what_they_use(
    #[case] expr: Expression,
    #[case] columns: &[&str],
    #[values(false, true)] nullable: bool,
    #[values(None, Some(&[1u64, 3, 4][..]))] rows: Option<&'static [u64]>,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let table = transform_table(nullable)?;
    let strategy = TableStrategy::new(two_row_chunks(), two_row_chunks()).with_list_layout();
    let (segments, layout) = write_chunks(&session, &strategy, vec![table.clone()]).await?;
    let source = Arc::new(CountingSegmentSource::new(segments));
    let scan_source: Arc<dyn SegmentSource> = Arc::<CountingSegmentSource>::clone(&source);

    let mut builder = builder(&session, &scan_source, &layout)?
        .with_projection(expr.bind(layout.dtype())?);
    if let Some(rows) = rows {
        builder = builder.with_row_indices(StrictSortedBuffer::try_new(Buffer::copy_from(rows))?);
    }
    let dtype = builder.dtype().clone();
    let actual = collect_scan(dtype, builder).await?;

    let keep = Mask::from_iter(
        (0..table.len() as u64).map(|row| rows.is_none_or(|rows| rows.contains(&row))),
    );
    let expected = table.filter(keep)?.apply(&expr)?;
    assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());

    let mut segment_column = HashMap::new();
    segment_columns(&layout, "", &mut segment_column)?;
    let read = source
        .requested()
        .iter()
        .map(|id| segment_column[id].as_str())
        .collect::<BTreeSet<_>>();
    let columns: &[&str] = if columns.is_empty() && !nullable {
        // A list that cannot be null has no validity to ask: its lengths answer instead.
        &["l.offsets"]
    } else {
        columns
    };
    let mut expected_read = columns.iter().copied().collect::<BTreeSet<_>>();
    if nullable {
        expected_read.insert("l.validity");
    }
    assert_eq!(read, expected_read);
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
