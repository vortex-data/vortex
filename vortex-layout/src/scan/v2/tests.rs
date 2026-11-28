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
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowSessionExt;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_expr::ArrayExprExt;
use vortex_expr::Expression;
use vortex_expr::and;
use vortex_expr::eq;
use vortex_expr::get_item;
use vortex_expr::gt;
use vortex_expr::is_not_null;
use vortex_expr::lit;
use vortex_expr::lt;
use vortex_expr::mask;
use vortex_expr::pack;
use vortex_expr::root;
use vortex_functions::expr::is_null;
use vortex_functions::expr::select;
use vortex_io::session::RuntimeSessionExt;
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
use super::Kept;
use super::PreparedScan;
use super::PrunePlanner;
use super::ROW_BLOCK_SIZE;
use super::ScanBuilder;
use super::ScanFile;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
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
        move |kept: Option<Kept>| -> Box<dyn Planner> {
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

/// `{id, s: {inner: {label, other, deep: {leaf, rest}}, x}}`. `nested_nullable` sets the
/// nullability of both `inner` and `deep`. Each nullable struct is null in a different row.
fn nested_struct_table(s_nullable: bool, nested_nullable: bool) -> VortexResult<ArrayRef> {
    let validity = |nullable: bool, null_row: usize| {
        if nullable {
            Validity::from_iter((0..4).map(|row| row != null_row))
        } else {
            Validity::NonNullable
        }
    };
    let ints = |start: i32| Buffer::from_iter(start..start + 4).into_array();

    let deep = StructArray::try_new(
        FieldNames::from(["leaf", "rest"]),
        vec![ints(0), ints(10)],
        4,
        validity(nested_nullable, 3),
    )?;
    let inner = StructArray::try_new(
        FieldNames::from(["label", "other", "deep"]),
        vec![ints(20), ints(30), deep.into_array()],
        4,
        validity(nested_nullable, 2),
    )?;
    let s = StructArray::try_new(
        FieldNames::from(["inner", "x"]),
        vec![inner.into_array(), ints(40)],
        4,
        validity(s_nullable, 1),
    )?;

    Ok(StructArray::from_fields(&[("id", ints(50)), ("s", s.into_array())])?.into_array())
}

/// Names each segment by the dotted path of the layout that owns it, such as `s.inner.validity`.
fn segment_paths(
    layout: &LayoutRef,
    path: &str,
    paths: &mut HashMap<SegmentId, String>,
) -> VortexResult<()> {
    for id in layout.segment_ids() {
        paths.insert(id, path.to_owned());
    }

    for (child, name) in layout.children()?.iter().zip(layout.child_names()) {
        let child_path = if path.is_empty() {
            name.to_string()
        } else {
            format!("{path}.{name}")
        };
        segment_paths(child, &child_path, paths)?;
    }

    Ok(())
}

fn s() -> Expression {
    get_item("s", root())
}

fn inner() -> Expression {
    get_item("inner", s())
}

fn masked_pack(fields: Vec<(&str, Expression)>, read: Expression) -> Expression {
    // Keep the projection dtype when optimizing away an all-true mask.
    mask(pack(fields, Nullability::Nullable), is_not_null(read))
}

/// The scan that reads the layout in a test.
#[derive(Clone, Copy, Debug)]
enum Executor {
    /// [`V1ScanBuilder`] over a reader of the layout.
    V1,
    /// The v2 scan, which plans the read with [`builder`].
    V2,
}

#[rstest]
#[case::leaf(
    get_item("label", inner()),
    &["s.inner.label"],
    &["s", "s.inner"],
)]
#[case::select_leaf(
    select(["label"], inner()),
    &["s.inner.label"],
    &["s", "s.inner"],
)]
#[case::is_not_null(is_not_null(inner()), &[], &["s", "s.inner"])]
#[case::rebuild_beside_whole_sibling(
    masked_pack(
        vec![
            (
                "inner",
                masked_pack(
                    vec![
                        ("label", get_item("label", inner())),
                        ("deep", select(["leaf"], get_item("deep", inner()))),
                    ],
                    inner(),
                ),
            ),
            ("x", get_item("x", s())),
        ],
        s(),
    ),
    &["s.inner.label", "s.inner.deep.leaf", "s.x"],
    &["s", "s.inner", "s.inner.deep"],
)]
#[tokio::test]
async fn nested_struct_projection_reads_only_used_fields(
    #[case] projection: Expression,
    #[case] field_paths: &[&str],
    #[case] struct_paths: &[&str],
    #[values((false, true), (true, false), (true, true))] nullability: (bool, bool),
    #[values(Executor::V1, Executor::V2)] executor: Executor,
) -> VortexResult<()> {
    let (s_nullable, nested_nullable) = nullability;
    let input = nested_struct_table(s_nullable, nested_nullable)?;
    let validity_paths: Vec<_> = struct_paths
        .iter()
        .copied()
        .filter(|&path| {
            if path == "s" {
                s_nullable
            } else {
                nested_nullable
            }
        })
        .map(|path| format!("{path}.validity"))
        .collect();
    let expected_paths = field_paths
        .iter()
        .copied()
        .chain(validity_paths.iter().map(String::as_str))
        .collect();

    assert_projection_reads(input, projection, executor, expected_paths).await
}

#[rstest]
#[tokio::test]
async fn non_nullable_nested_projection_reads_only_leaf(
    #[values(Executor::V1, Executor::V2)] executor: Executor,
) -> VortexResult<()> {
    let input = nested_struct_table(false, false)?;
    let projection = get_item("label", inner());
    assert_projection_reads(
        input,
        projection,
        executor,
        BTreeSet::from(["s.inner.label"]),
    )
    .await
}

async fn assert_projection_reads(
    input: ArrayRef,
    projection: Expression,
    executor: Executor,
    expected_paths: BTreeSet<&str>,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_chunks(&session, table().as_ref(), vec![input.clone()]).await?;
    let source = Arc::new(CountingSegmentSource::new(segments));
    let scan_source: Arc<dyn SegmentSource> = Arc::<CountingSegmentSource>::clone(&source);

    let projection = projection.bind(layout.dtype())?;
    let dtype = projection.dtype().clone();
    let scan_projection = projection.optimize_recursive()?;
    let expected = input.apply_bound(&projection)?;
    let actual = match executor {
        Executor::V1 => {
            let reader =
                layout.new_reader("".into(), scan_source, &session, &Default::default())?;
            let scan = V1ScanBuilder::new(session.clone(), reader).with_projection(scan_projection);
            collect(dtype, scan.into_stream()?).await?
        }
        Executor::V2 => {
            let scan = builder(&session, &scan_source, &layout)?.with_projection(scan_projection);
            collect_scan(dtype, scan).await?
        }
    };
    assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());

    let mut paths = HashMap::default();
    segment_paths(&layout, "", &mut paths)?;
    let read_paths: BTreeSet<&str> = source
        .requested_segments()
        .iter()
        .map(|id| paths[id].as_str())
        .collect();
    assert_eq!(read_paths, expected_paths);

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
