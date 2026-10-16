// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::TryStreamExt;
use futures::stream;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowSessionExt;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::MAX_ROWS_PER_CHUNK;
use super::PreparedScan;
use super::ScanBuilder;
use super::ScanFile;
use super::row_chunks;
use super::rows_per_chunk;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::dict::DictLayout;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::foreign::new_foreign_layout;
use crate::layouts::list::writer::ListLayoutStrategy;
use crate::layouts::row_idx::row_idx;
use crate::layouts::table::TableStrategy;
use crate::layouts::zoned::writer::ZonedLayoutOptions;
use crate::layouts::zoned::writer::ZonedStrategy;
use crate::scan::execution::SegmentLocation;
use crate::scan::scan_builder::ScanBuilder as V1ScanBuilder;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt as _;
use crate::test::new_session;

const CHUNK_ROWS: usize = 1000;
const CHUNKS: usize = 4;
static UNSUPPORTED_ID: CachedId = CachedId::new("test.unsupported");

/// A table chunk of `CHUNK_ROWS` rows starting at row `start`: an integer, a nullable string and a
/// list column.
fn table_chunk(start: usize) -> VortexResult<ArrayRef> {
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
        .with_selection(builder.selection.clone())
        .with_row_offset(builder.row_offset)
        .with_ordered(builder.ordered)
        .with_concurrency(builder.concurrency);
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
#[case::empty(0, 0, None, 65_536)]
#[case::single_chunk(1000, 100_000_000, None, 65_536)]
#[case::no_bytes(100_000, 0, None, 65_536)]
#[case::small_rows(1_000_000, 64_000_000, None, 65_536)]
#[case::large_rows(1_000_000, 384_000_000, None, 16_384)]
#[case::minimum(1_000_000, 2_000_000_000, None, 8192)]
#[case::sparse_small_rows(3_339_715, 49_808_116, Some(834_928), 524_288)]
#[case::sparse_medium_rows(1_000_000, 35_640_884, Some(250_000), 131_072)]
#[case::sparse_more_workers(3_339_715, 49_808_116, Some(27_830), 65_536)]
#[case::sparse_lists(1_000_000, 86_983_648, Some(250_000), 65_536)]
#[case::sparse_wide_rows(1_000_000, 3_354_532_688, Some(250_000), 8192)]
fn chunk_byte_budget(
    #[case] rows: u64,
    #[case] bytes: u32,
    #[case] sparse_target: Option<u64>,
    #[case] expected: usize,
) {
    let locations = [SegmentLocation {
        offset: 0,
        length: bytes,
        alignment: Alignment::none(),
    }];
    assert_eq!(rows_per_chunk(rows, &locations, sparse_target), expected);
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
async fn tiny_selection_shares_a_stored_segment() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let input = PrimitiveArray::from_iter(0_i32..1_048_576).into_array();
    let dtype = input.dtype().clone();
    let segments = Arc::new(TestSegments::default());
    let (mut sequence, eof) = SequenceId::root().split();
    let layout = FlatLayoutStrategy::default()
        .write_stream(
            ArrayContext::empty().into(),
            Arc::<TestSegments>::clone(&segments),
            SequentialStreamAdapter::new(
                dtype.clone(),
                stream::iter([Ok((sequence.advance(), input))]),
            )
            .sendable(),
            eof,
            &session,
        )
        .await?;
    let counted = Arc::new(CountingSegments {
        inner: segments,
        reads: AtomicUsize::new(0),
    });
    let segments: Arc<dyn SegmentSource> = Arc::<CountingSegments>::clone(&counted);
    // These rows share one stored segment but fall across the old worker-sized task boundary,
    // even on one core. Selection-aware sizing lets one projection share its read.
    let scan = builder(&session, &segments, &layout)?.with_row_indices(
        StrictSortedBuffer::try_new(Buffer::from(vec![10, 350_000]))?,
    );
    let actual = collect(dtype, scan.into_stream()?).await?;
    assert_arrays_eq!(
        actual,
        Buffer::from(vec![10_i32, 350_000]).into_array(),
        &mut session.create_execution_ctx()
    );
    assert_eq!(counted.reads.load(Ordering::Relaxed), 1);
    Ok(())
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
    let scan = PreparedScan::try_new(builder(&session, &segments, &layout)?)?;
    let splits = scan.split_plans();
    assert_eq!(
        splits.len(),
        (STORED_CHUNKS * STORED_CHUNK_ROWS).div_ceil(MAX_ROWS_PER_CHUNK)
    );
    for (index, split) in splits.into_iter().enumerate() {
        let rows = split.scope.rows.clone();
        let before = counted.reads.load(Ordering::Relaxed);
        let arrays = split.run(Arc::clone(&scan.io)).await?;
        let reads = counted.reads.load(Ordering::Relaxed) - before;
        let code_chunks =
            (rows.end - 1) / STORED_CHUNK_ROWS as u64 - rows.start / STORED_CHUNK_ROWS as u64 + 1;
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

#[tokio::test]
async fn list_layouts_match_v1() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let strategy = TableStrategy::new(chunked_flat(), chunked_flat()).with_list_layout();
    let (segments, layout) = write(&session, &strategy).await?;
    assert_same_as_v1(&session, || builder(&session, &segments, &layout)).await
}

#[rstest]
#[case::empty(vec![])]
#[case::null_only(vec![3, 6, 9])]
#[case::contiguous(vec![1, 2])]
#[case::contiguous_with_empty(vec![7, 8])]
#[case::mixed(vec![0, 1, 2, 3, 8, 31, 63])]
#[case::all((0..64).collect())]
#[tokio::test]
async fn selected_list_values_match_input(
    #[case] rows: Vec<u64>,
    #[values(false, true)] nested: bool,
    #[values(false, true)] nullable: bool,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let mut offsets = vec![0_u32];
    let mut total = 0;
    for row in 0..64 {
        total += row % 4;
        offsets.push(total);
    }
    let inner_strategy = ListLayoutStrategy::default()
        .with_elements(chunked_flat())
        .with_offsets(chunked_flat())
        .with_validity(chunked_flat());
    let (elements, element_strategy) = if nested {
        let elements = ListArray::try_new(
            Buffer::from_iter(0..2 * total).into_array(),
            Buffer::from_iter((0..=total).map(|row| 2 * row)).into_array(),
            Validity::Array(BoolArray::from_iter((0..total).map(|row| row % 3 != 0)).into_array()),
        )?
        .into_array();
        (
            elements,
            Arc::new(inner_strategy) as Arc<dyn LayoutStrategy>,
        )
    } else {
        (Buffer::from_iter(0..total).into_array(), chunked_flat())
    };
    let validity = if nullable {
        Validity::Array(BoolArray::from_iter((0..64).map(|row| row % 3 != 0)).into_array())
    } else {
        Validity::NonNullable
    };
    let input =
        ListArray::try_new(elements, Buffer::from(offsets).into_array(), validity)?.into_array();
    let dtype = input.dtype().clone();
    let expected = input.take(Buffer::from(rows.clone()).into_array())?;
    let strategy = ListLayoutStrategy::default()
        .with_elements(element_strategy)
        .with_offsets(chunked_flat())
        .with_validity(chunked_flat());
    let segments = Arc::new(TestSegments::default());
    let (mut id, eof) = SequenceId::root().split();
    let layout = strategy
        .write_stream(
            ArrayContext::empty().into(),
            Arc::<TestSegments>::clone(&segments),
            SequentialStreamAdapter::new(dtype.clone(), stream::iter([Ok((id.advance(), input))]))
                .sendable(),
            eof,
            &session,
        )
        .await?;
    let counted = Arc::new(CountingSegments {
        inner: segments,
        reads: AtomicUsize::new(0),
    });
    let segments: Arc<dyn SegmentSource> = Arc::<CountingSegments>::clone(&counted);
    let empty = rows.is_empty();
    let null_only = nullable && !empty && rows.iter().all(|row| row % 3 == 0);
    let scan = builder(&session, &segments, &layout)?
        .with_row_indices(StrictSortedBuffer::try_new(Buffer::from(rows))?);
    let actual = collect(dtype, scan.into_stream()?).await?;
    assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
    if empty {
        assert_eq!(counted.reads.load(Ordering::Relaxed), 0);
    } else if null_only {
        // Only offsets and validity are needed, even when null lists store nonempty elements.
        assert_eq!(counted.reads.load(Ordering::Relaxed), 2);
    }
    Ok(())
}

#[tokio::test]
async fn unordered_returns_every_row() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, table().as_ref()).await?;
    let scan = PreparedScan::try_new(builder(&session, &segments, &layout)?.with_ordered(false))?;
    let chunks: Vec<ArrayRef> = scan.execute_stream()?.try_collect().await?;
    let rows: usize = chunks.iter().map(|chunk| chunk.len()).sum();
    assert_eq!(rows, CHUNKS * CHUNK_ROWS);
    Ok(())
}

#[tokio::test]
async fn primitive_column() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write(&session, table().as_ref()).await?;
    let dtype = layout.dtype().clone();
    let scan = PreparedScan::try_new(
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
    let scan = PreparedScan::try_new(builder(&session, &segments, &layout)?)?;
    assert_eq!(scan.splits, vec![0..(CHUNKS * CHUNK_ROWS) as u64]);
    let arrays: Vec<_> = scan.execute_stream()?.try_collect().await?;
    assert_eq!(arrays.len(), 1);
    assert_eq!(arrays[0].len(), CHUNKS * CHUNK_ROWS);
    Ok(())
}

#[tokio::test]
async fn unknown_layout_returns_an_error() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, known) = write(&session, chunked_flat().as_ref()).await?;
    let layout = new_foreign_layout(
        *UNSUPPORTED_ID,
        known.dtype().clone(),
        known.row_count(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let error = builder(&session, &segments, &layout)?
        .into_stream()?
        .try_next()
        .await
        .expect_err("V2 must reject an unsupported layout");
    assert!(
        error
            .to_string()
            .contains("No physical plan implementation for layout 'test.unsupported'")
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn unsupported_chunk_is_checked_when_reached(
    #[values(false, true)] selected: bool,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, known) = write(&session, chunked_flat().as_ref()).await?;
    let rows = known.row_count();
    let unknown = new_foreign_layout(
        *UNSUPPORTED_ID,
        known.dtype().clone(),
        rows,
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let layout = ChunkedLayout::new(
        2 * rows,
        known.dtype().clone(),
        OwnedLayoutChildren::layout_children(vec![Arc::clone(&known), unknown]),
    )
    .into_layout();
    let range = if selected { rows..2 * rows } else { 0..rows };
    let result = collect(
        known.dtype().clone(),
        builder(&session, &segments, &layout)?
            .with_row_range(range)
            .into_stream()?,
    )
    .await;
    if selected {
        let error = result.expect_err("a reached unsupported chunk must fail");
        assert!(error.to_string().contains("test.unsupported"));
    } else {
        let expected = collect(
            known.dtype().clone(),
            builder(&session, &segments, &known)?.into_stream()?,
        )
        .await?;
        assert_arrays_eq!(result?, expected, &mut session.create_execution_ctx());
    }
    Ok(())
}
