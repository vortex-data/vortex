// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[path = "legacy_tests.rs"]
mod legacy;
#[path = "list_tests.rs"]
mod lists;

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Barrier;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::thread;

use futures::FutureExt;
use futures::TryStreamExt;
use futures::stream;
use parking_lot::Mutex;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::dict::dict_encode;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::byte_length;
use vortex_array::expr::cast;
use vortex_array::expr::dynamic;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::like;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::not_eq;
use vortex_array::expr::or;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;

use crate::LayoutChildren;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::chunked::Chunked;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::dict::Dict;
use crate::layouts::dict::writer::DictLayoutOptions;
use crate::layouts::dict::writer::DictStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::row_idx::row_idx;
use crate::layouts::struct_::StructLayout;
use crate::layouts::zoned::writer::ZonedLayoutOptions;
use crate::layouts::zoned::writer::ZonedStrategy;
use crate::plan::PlanRef;
use crate::scan::planning::SegmentLocation;
use crate::scan::scan_builder::ScanBuilder;
use crate::scan::v2;
use crate::scan::v2::FilePlans;
use crate::scan::v2::ScanFile;
use crate::scan::v2::file::shared_file;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt as _;
use crate::test::new_session;

const CHUNK_ROWS: i32 = 1000;
const DTYPE: DType = DType::Primitive(PType::I32, NonNullable);

#[derive(Clone)]
struct CountingChildren {
    inner: Arc<dyn LayoutChildren>,
    accessed: Arc<AtomicUsize>,
    row_counts: Arc<AtomicUsize>,
    hints: Arc<AtomicUsize>,
}

impl LayoutChildren for CountingChildren {
    fn to_arc(&self) -> Arc<dyn LayoutChildren> {
        Arc::new(self.clone())
    }

    fn child(&self, index: usize, dtype: &DType) -> VortexResult<LayoutRef> {
        self.accessed.fetch_add(1, Ordering::Relaxed);
        self.inner.child(index, dtype)
    }

    fn child_row_count(&self, index: usize) -> u64 {
        self.row_counts.fetch_add(1, Ordering::Relaxed);
        self.inner.child_row_count(index)
    }

    fn nchildren(&self) -> usize {
        self.inner.nchildren()
    }

    fn child_is_indivisible(&self, index: usize) -> bool {
        self.hints.fetch_add(1, Ordering::Relaxed);
        self.inner.child_is_indivisible(index)
    }
}

#[rstest]
#[case::empty(&[], 0)]
#[case::sparse(&[1, 2001], 2)]
#[case::same_chunk(&[1001, 1002], 1)]
#[case::last_row(&[3999], 1)]
#[tokio::test]
async fn identity_preparation_keeps_flat_chunks_lazy(
    #[case] indices: &'static [u64],
    #[case] selected_chunks: usize,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_layout(&session).await?;
    let mut expected_segments = BTreeSet::new();
    for &index in indices {
        let child = layout
            .slot(usize::try_from(index / CHUNK_ROWS as u64)?)?
            .ok_or_else(|| vortex_err!("Missing selected chunk"))?;
        expected_segments.extend(segment_ids(&child)?);
    }
    let registered = Arc::default();
    let segments: Arc<dyn SegmentSource> = Arc::new(RecordingSegments {
        inner: segments,
        registered: Arc::clone(&registered),
        reads: Arc::default(),
    });
    let accessed = Arc::new(AtomicUsize::new(0));
    let row_counts = Arc::new(AtomicUsize::new(0));
    let hints = Arc::new(AtomicUsize::new(0));
    let layout = ChunkedLayout::new(
        layout.row_count(),
        DTYPE,
        Arc::new(CountingChildren {
            inner: Arc::clone(layout.as_::<Chunked>().children()),
            accessed: Arc::clone(&accessed),
            row_counts: Arc::clone(&row_counts),
            hints: Arc::clone(&hints),
        }),
    )
    .into_layout();
    let file = scan_file(&segments, &layout)?;
    accessed.store(0, Ordering::Relaxed);
    row_counts.store(0, Ordering::Relaxed);
    hints.store(0, Ordering::Relaxed);
    let case = Case {
        rows: Some(indices),
        ..Default::default()
    };
    let scan = v2::prepare(builder(&session, &segments, &layout, &case)?, file)?;
    assert_eq!(accessed.load(Ordering::Relaxed), 0);
    assert_eq!(row_counts.load(Ordering::Relaxed), 0);
    assert_eq!(hints.load(Ordering::Relaxed), selected_chunks);

    let actual = await_tasks(DTYPE, scan.execute(None)?).await?;
    assert_eq!(*registered.lock(), expected_segments);
    let expected = indices
        .iter()
        .copied()
        .map(i32::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    assert_arrays_eq!(
        actual,
        Buffer::from(expected).into_array(),
        &mut session.create_execution_ctx()
    );
    Ok(())
}

/// Four chunks of consecutive integers, `0..4000`.
async fn write_layout(
    session: &VortexSession,
) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef)> {
    let segments = Arc::new(TestSegments::default());
    let (mut sequence_id, eof) = SequenceId::root().split();
    let chunks = (0..4)
        .map(|chunk| {
            let values = Buffer::from_iter(chunk * CHUNK_ROWS..(chunk + 1) * CHUNK_ROWS);
            Ok((sequence_id.advance(), values.into_array()))
        })
        .collect::<Vec<_>>();
    let layout = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default())
        .write_stream(
            ArrayContext::empty().into(),
            Arc::<TestSegments>::clone(&segments),
            SequentialStreamAdapter::new(DTYPE, stream::iter(chunks)).sendable(),
            eof,
            session,
        )
        .await?;
    Ok((segments, layout))
}

/// A [`ScanFile`] over test segments. They have no byte offsets, and the scan fetches them by id,
/// so each location is a placeholder whose offset is its id, with the alignment the test segments
/// already have.
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
        io: None,
        plans: None,
    })
}

#[derive(Clone, Default)]
struct Case {
    filter: bool,
    row_range: Option<Range<u64>>,
    limit: Option<u64>,
    rows: Option<&'static [u64]>,
    row_idx: bool,
    /// Adds a second conjunct to the filter, `$ < below`.
    below: Option<i32>,
}

fn builder(
    session: &VortexSession,
    segments: &Arc<dyn SegmentSource>,
    layout: &LayoutRef,
    case: &Case,
) -> VortexResult<ScanBuilder<ArrayRef>> {
    let reader = layout.new_reader(
        "".into(),
        Arc::clone(segments),
        session,
        &Default::default(),
    )?;
    let mut builder = ScanBuilder::new(session.clone(), reader);
    if case.filter {
        let filter = gt(root(), lit(1500_i32));
        let filter = match case.below {
            Some(below) => and(filter, lt(root(), lit(below))),
            None => filter,
        };
        builder = builder.with_filter(filter.bind(&DTYPE)?);
    }
    if let Some(row_range) = case.row_range.clone() {
        builder = builder.with_row_range(row_range);
    }
    if let Some(limit) = case.limit {
        builder = builder.with_limit(limit);
    }
    if let Some(rows) = case.rows {
        builder = builder.with_row_indices(StrictSortedBuffer::try_new(Buffer::copy_from(rows))?);
    }
    if case.row_idx {
        builder = builder
            .with_projection(row_idx().bind(&DTYPE)?)
            .with_row_offset(10_000);
    }
    Ok(builder)
}

async fn await_tasks(
    dtype: DType,
    tasks: Vec<futures::future::BoxFuture<'static, VortexResult<Option<ArrayRef>>>>,
) -> VortexResult<ArrayRef> {
    let mut chunks = Vec::new();
    for task in tasks {
        if let Some(chunk) = task.await? {
            chunks.push(chunk);
        }
    }
    Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
}

fn case(filter: bool, row_range: Option<Range<u64>>, limit: Option<u64>) -> Case {
    Case {
        filter,
        row_range,
        limit,
        ..Case::default()
    }
}

fn rows(filter: bool, rows: &'static [u64]) -> Case {
    Case {
        filter,
        rows: Some(rows),
        ..Case::default()
    }
}

fn conjuncts(below: i32) -> Case {
    Case {
        filter: true,
        below: Some(below),
        ..Case::default()
    }
}

fn row_idx_case(filter: bool) -> Case {
    Case {
        filter,
        row_idx: true,
        ..Case::default()
    }
}

#[rstest]
#[case::everything(case(false, None, None))]
#[case::filter(case(true, None, None))]
#[case::row_range(case(false, Some(500..2500), None))]
#[case::filter_and_row_range(case(true, Some(500..2500), None))]
#[case::limit(case(false, None, Some(1200)))]
#[case::limit_and_row_range(case(false, Some(900..3100), Some(1500)))]
#[case::selection(rows(false, &[0, 5, 999, 1000, 2500, 3999]))]
#[case::selection_and_filter(rows(true, &[0, 1499, 1501, 2500, 3999]))]
#[case::row_idx(row_idx_case(false))]
#[case::row_idx_and_filter(row_idx_case(true))]
#[case::two_conjuncts(conjuncts(3000))]
#[case::conjuncts_matching_nothing(conjuncts(1000))]
#[tokio::test(flavor = "multi_thread")]
async fn stream_matches_default(#[case] case: Case) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_layout(&session).await?;
    let dtype = builder(&session, &segments, &layout, &case)?.dtype()?;

    let expected = builder(&session, &segments, &layout, &case)?
        .into_stream()?
        .try_collect::<Vec<_>>()
        .await?;
    let actual = v2::into_stream(
        builder(&session, &segments, &layout, &case)?,
        scan_file(&segments, &layout)?,
    )?
    .try_collect::<Vec<_>>()
    .await?;

    assert_arrays_eq!(
        ChunkedArray::try_new(actual, dtype.clone())?,
        ChunkedArray::try_new(expected, dtype)?,
        &mut session.create_execution_ctx()
    );
    Ok(())
}

#[rstest]
#[case::single_split(250..750)]
#[case::empty_range(250..250)]
#[tokio::test(flavor = "multi_thread")]
async fn stream_without_parallel_splits(
    #[case] row_range: Range<u64>,
    #[values(true, false)] ordered: bool,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_layout(&session).await?;
    let case = case(false, Some(row_range), None);
    let dtype = builder(&session, &segments, &layout, &case)?.dtype()?;
    let expected = builder(&session, &segments, &layout, &case)?
        .with_ordered(ordered)
        .into_stream()?
        .try_collect::<Vec<_>>()
        .await?;
    let actual = v2::into_stream(
        builder(&session, &segments, &layout, &case)?.with_ordered(ordered),
        scan_file(&segments, &layout)?,
    )?
    .try_collect::<Vec<_>>()
    .await?;
    assert_arrays_eq!(
        ChunkedArray::try_new(actual, dtype.clone())?,
        ChunkedArray::try_new(expected, dtype)?,
        &mut session.create_execution_ctx()
    );
    Ok(())
}

/// The executor's own builder, configured directly rather than copied from a default builder,
/// returns what the default scan returns.
#[rstest]
#[case::everything(None, None)]
#[case::filter_and_row_range(Some(gt(root(), lit(1500_i32))), Some(500..2500))]
#[tokio::test(flavor = "multi_thread")]
async fn own_builder_matches_default(
    #[case] filter: Option<Expression>,
    #[case] row_range: Option<Range<u64>>,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_layout(&session).await?;
    let reader = || {
        layout.new_reader(
            "".into(),
            Arc::clone(&segments),
            &session,
            &Default::default(),
        )
    };
    let filter = filter.map(|filter| filter.bind(&DTYPE)).transpose()?;

    let mut default = ScanBuilder::new(session.clone(), reader()?).with_some_filter(filter.clone());
    let mut own = v2::ScanBuilder::new(session.clone(), reader()?, scan_file(&segments, &layout)?)
        .with_some_filter(filter);
    if let Some(row_range) = row_range {
        default = default.with_row_range(row_range.clone());
        own = own.with_row_range(row_range);
    }
    assert_eq!(own.dtype()?, default.dtype()?);

    let expected = default.into_stream()?.try_collect::<Vec<_>>().await?;
    let actual = own.into_stream()?.try_collect::<Vec<_>>().await?;
    assert_arrays_eq!(
        ChunkedArray::try_new(actual, DTYPE)?,
        ChunkedArray::try_new(expected, DTYPE)?,
        &mut session.create_execution_ctx()
    );
    Ok(())
}

#[rstest]
#[case::whole_scan(case(true, None, None), None)]
#[case::execute_range(case(true, None, None), Some(700..3300))]
#[case::both_ranges(case(false, Some(500..2500), None), Some(2000..4000))]
#[case::empty_range(case(true, None, None), Some(1200..1200))]
#[tokio::test(flavor = "multi_thread")]
async fn execute_matches_default(
    #[case] case: Case,
    #[case] execute_range: Option<Range<u64>>,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_layout(&session).await?;

    let default = builder(&session, &segments, &layout, &case)?.prepare()?;
    let replacement = v2::prepare(
        builder(&session, &segments, &layout, &case)?,
        scan_file(&segments, &layout)?,
    )?;
    assert_eq!(replacement.dtype(), default.dtype());
    let dtype = default.dtype().clone();

    let expected = await_tasks(dtype.clone(), default.execute(execute_range.clone())?).await?;
    let actual = await_tasks(dtype, replacement.execute(execute_range)?).await?;
    assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn filter_with_limit_is_rejected() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_layout(&session).await?;
    let case = case(true, None, Some(10));
    let file = scan_file(&segments, &layout)?;
    assert!(v2::prepare(builder(&session, &segments, &layout, &case)?, file).is_err());
    Ok(())
}

/// Records every segment whose read is awaited.
///
/// A scan also registers segments it may read, so the source can coalesce them. Registration
/// alone reads nothing, so only awaited reads are recorded.
struct RecordingSegments {
    inner: Arc<dyn SegmentSource>,
    registered: Arc<Mutex<BTreeSet<u32>>>,
    reads: Arc<Mutex<BTreeSet<u32>>>,
}

impl SegmentSource for RecordingSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.registered.lock().insert(*id);
        let read = self.inner.request(id);
        let reads = Arc::clone(&self.reads);
        async move {
            reads.lock().insert(*id);
            read.await
        }
        .boxed()
    }
}

struct CountedSegments {
    inner: Arc<dyn SegmentSource>,
    reads: Arc<Mutex<Vec<SegmentId>>>,
}

impl SegmentSource for CountedSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let read = self.inner.request(id);
        let reads = Arc::clone(&self.reads);
        async move {
            reads.lock().push(id);
            read.await
        }
        .boxed()
    }
}

#[rstest]
#[case::whole(None, None, &[0, 1000, 2000, 3999], 5)]
#[case::range(Some(500..3500), None, &[1000, 2000], 3)]
#[case::limit(None, Some(2), &[0, 1000], 3)]
#[tokio::test]
async fn sparse_projection_reads_each_column_chunk_once(
    #[case] row_range: Option<Range<u64>>,
    #[case] limit: Option<u64>,
    #[case] expected_indices: &[usize],
    #[case] expected_reads: usize,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let written = Arc::new(TestSegments::default());
    let values = PrimitiveArray::from_option_iter((0..4000_i32).map(|i| (i % 7 != 0).then_some(i)))
        .into_array();
    let expected =
        StructArray::from_fields(&[("a", values.clone()), ("b", values.clone())])?.into_array();
    let mut fields = Vec::new();
    for chunk_rows in [1000, 4000] {
        let (mut sequence, eof) = SequenceId::root().split();
        let chunks = (0..4000)
            .step_by(chunk_rows)
            .map(|start| Ok((sequence.advance(), values.slice(start..start + chunk_rows)?)))
            .collect::<Vec<_>>();
        fields.push(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default())
                .write_stream(
                    ArrayContext::empty().into(),
                    Arc::<TestSegments>::clone(&written),
                    SequentialStreamAdapter::new(values.dtype().clone(), stream::iter(chunks))
                        .sendable(),
                    eof,
                    &session,
                )
                .await?,
        );
    }
    let layout = StructLayout::new(4000, expected.dtype().clone(), fields).into_layout();
    let reads = Arc::default();
    let segments: Arc<dyn SegmentSource> = Arc::new(CountedSegments {
        inner: written,
        reads: Arc::clone(&reads),
    });
    let case = Case {
        rows: Some(&[0, 1000, 2000, 3999]),
        row_range,
        limit,
        ..Default::default()
    };
    let scan = v2::prepare(
        builder(&session, &segments, &layout, &case)?,
        scan_file(&segments, &layout)?,
    )?;
    let expected = expected.filter(Mask::from_indices(4000, expected_indices.iter().copied()))?;
    for _ in 0..2 {
        reads.lock().clear();
        let actual = await_tasks(expected.dtype().clone(), scan.execute(None)?).await?;
        assert_eq!(reads.lock().len(), expected_reads);
        assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
    }
    Ok(())
}

#[rstest]
#[case::projection(false)]
#[case::filter_and_projection(true)]
#[tokio::test]
async fn ordinary_segment_reads_match_v1_on_each_execution(
    #[case] filter: bool,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_layout(&session).await?;
    let reads = Arc::default();
    let segments: Arc<dyn SegmentSource> = Arc::new(CountedSegments {
        inner: segments,
        reads: Arc::clone(&reads),
    });
    let case = case(filter, Some(1000..2000), None);
    let default = builder(&session, &segments, &layout, &case)?.prepare()?;
    let replacement = v2::prepare(
        builder(&session, &segments, &layout, &case)?,
        scan_file(&segments, &layout)?,
    )?;
    for _ in 0..2 {
        reads.lock().clear();
        let expected = await_tasks(DTYPE, default.execute(None)?).await?;
        let mut expected_reads = std::mem::take(&mut *reads.lock());
        expected_reads.sort_unstable();
        assert_eq!(expected_reads.len(), if filter { 2 } else { 1 });

        let actual = await_tasks(DTYPE, replacement.execute(None)?).await?;
        let mut actual_reads = std::mem::take(&mut *reads.lock());
        actual_reads.sort_unstable();
        assert_eq!(actual_reads, expected_reads);
        assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
    }
    Ok(())
}

/// Building tasks announces all their ranges without polling a single read.
#[tokio::test]
async fn tasks_announce_before_they_run() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_layout(&session).await?;
    let registered = Arc::new(Mutex::new(BTreeSet::new()));
    let reads = Arc::new(Mutex::new(BTreeSet::new()));
    let segments: Arc<dyn SegmentSource> = Arc::new(RecordingSegments {
        inner: segments,
        registered: Arc::clone(&registered),
        reads: Arc::clone(&reads),
    });
    let scan = v2::prepare(
        builder(&session, &segments, &layout, &Case::default())?,
        scan_file(&segments, &layout)?,
    )?;
    let tasks = scan.execute(None)?;
    assert_eq!(*registered.lock(), segment_ids(&layout)?);
    assert!(reads.lock().is_empty());
    let actual = await_tasks(DTYPE, tasks).await?;
    assert_arrays_eq!(
        actual,
        Buffer::from_iter(0..4 * CHUNK_ROWS).into_array(),
        &mut session.create_execution_ctx()
    );
    Ok(())
}

/// The ids of the segments below `layout`.
fn segment_ids(layout: &LayoutRef) -> VortexResult<BTreeSet<u32>> {
    let mut ids = BTreeSet::new();
    for layout in layout.depth_first_traversal() {
        ids.extend(layout?.segment_ids().into_iter().map(|id| *id));
    }
    Ok(ids)
}

/// Four 1000-row chunks of `0..4000`, one zone each.
async fn write_zoned_layout(
    session: &VortexSession,
) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef)> {
    let written = Arc::new(TestSegments::default());
    let segments: Arc<dyn SegmentSource> = Arc::clone(&written) as _;
    let (mut sequence_id, eof) = SequenceId::root().split();
    let chunks = (0..4)
        .map(|chunk| {
            let values = Buffer::from_iter(chunk * CHUNK_ROWS..(chunk + 1) * CHUNK_ROWS);
            Ok((sequence_id.advance(), values.into_array()))
        })
        .collect::<Vec<_>>();
    let layout = ZonedStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        FlatLayoutStrategy::default(),
        ZonedLayoutOptions {
            block_size: NonZeroUsize::new(CHUNK_ROWS as usize).unwrap_or(NonZeroUsize::MIN),
            ..Default::default()
        },
    )
    .write_stream(
        ArrayContext::empty().into(),
        written,
        SequentialStreamAdapter::new(DTYPE, stream::iter(chunks)).sendable(),
        eof,
        session,
    )
    .await?;
    Ok((segments, layout))
}

/// Pruning drops the zones whose statistics prove the filter false: of four 1000-row zones over
/// `0..4000`, the first cannot hold a value above 1500, so V2 reads the zone table and the other
/// three chunks, never the first, and still returns what the default path returns.
#[tokio::test(flavor = "multi_thread")]
async fn zone_pruning_reads_only_zones_that_can_match() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_zoned_layout(&session).await?;

    let data = layout
        .slot(0)?
        .ok_or_else(|| vortex_error::vortex_err!("no data"))?;
    let zones = layout
        .slot(1)?
        .ok_or_else(|| vortex_error::vortex_err!("no zones"))?;
    let data_chunks = data.children()?;
    let mut expected_reads = segment_ids(&zones)?;
    for chunk in &data_chunks[1..] {
        expected_reads.extend(segment_ids(chunk)?);
    }

    let case = case(true, None, None);
    let dtype = builder(&session, &segments, &layout, &case)?.dtype()?;
    let expected = builder(&session, &segments, &layout, &case)?
        .into_stream()?
        .try_collect::<Vec<_>>()
        .await?;

    let recording = Arc::new(RecordingSegments {
        registered: Default::default(),
        inner: Arc::clone(&segments),
        reads: Arc::default(),
    });
    let mut file = scan_file(&segments, &layout)?;
    file.segments = Arc::clone(&recording) as _;
    let actual = v2::into_stream(builder(&session, &segments, &layout, &case)?, file)?
        .try_collect::<Vec<_>>()
        .await?;

    assert_arrays_eq!(
        ChunkedArray::try_new(actual, dtype.clone())?,
        ChunkedArray::try_new(expected, dtype)?,
        &mut session.create_execution_ctx()
    );
    assert_eq!(*recording.reads.lock(), expected_reads);
    Ok(())
}

const WORDS: [&str; 4] = ["", "apple", "banana", "cherry"];

/// Four chunks of dictionary-encoded strings cycling through [`WORDS`].
async fn write_dict_layout(
    session: &VortexSession,
) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef)> {
    let segments = Arc::new(TestSegments::default());
    let (mut sequence_id, eof) = SequenceId::root().split();
    let chunks = (0..4)
        .map(|chunk| {
            let words = (0..CHUNK_ROWS).map(|row| WORDS[((row * 7 + chunk) % 4) as usize]);
            let values = VarBinArray::from_iter_nonnull(words, DType::Utf8(NonNullable));
            Ok((sequence_id.advance(), values.into_array()))
        })
        .collect::<Vec<_>>();
    let layout = DictStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        FlatLayoutStrategy::default(),
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        DictLayoutOptions::default(),
        // The fixture must stay dictionary encoded regardless of compressor heuristics.
        Arc::new(|chunk: &ArrayRef, ctx: &mut ExecutionCtx| {
            Ok(dict_encode(chunk, ctx)?.into_array())
        }),
    )
    .write_stream(
        ArrayContext::empty().into(),
        Arc::<TestSegments>::clone(&segments),
        SequentialStreamAdapter::new(DType::Utf8(NonNullable), stream::iter(chunks)).sendable(),
        eof,
        session,
    )
    .await?;
    Ok((segments, layout))
}

/// Expressions over dictionary values run on shared canonical values when the scan also reads
/// the values whole, and on the encoded values otherwise, including the negative-cost part of a
/// projection split off from the rest; each returns what the default path returns.
#[rstest]
#[case::values_read_whole(root(), Some(eq(root(), lit("apple"))))]
#[case::byte_length_split(
    cast(byte_length(root()), DType::Primitive(PType::I64, NonNullable)),
    Some(not_eq(root(), lit("")))
)]
#[case::byte_length_whole(byte_length(root()), Some(like(root(), lit("%an%"))))]
#[case::no_filter(byte_length(root()), None)]
#[tokio::test(flavor = "multi_thread")]
async fn dictionary_expressions_match_default(
    #[case] projection: Expression,
    #[case] filter: Option<Expression>,
    #[values(false, true)] with_selection: bool,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_dict_layout(&session).await?;
    assert!(
        layout
            .depth_first_traversal()
            .any(|layout| layout.is_ok_and(|layout| layout.is::<Dict>())),
        "the strings must be dictionary encoded"
    );
    let dtype = DType::Utf8(NonNullable);
    let builder = || -> VortexResult<ScanBuilder<ArrayRef>> {
        let reader = layout.new_reader(
            "".into(),
            Arc::clone(&segments),
            &session,
            &Default::default(),
        )?;
        let builder =
            ScanBuilder::new(session.clone(), reader).with_projection(projection.bind(&dtype)?);
        let builder = if with_selection {
            builder.with_row_indices(StrictSortedBuffer::try_new(Buffer::from(vec![
                0_u64, 103, 999, 2001, 2500, 3002,
            ]))?)
        } else {
            builder
        };
        Ok(match &filter {
            Some(filter) => builder.with_filter(filter.bind(&dtype)?),
            None => builder,
        })
    };
    let result_dtype = builder()?.dtype()?;

    let expected = builder()?.into_stream()?.try_collect::<Vec<_>>().await?;
    let actual = v2::into_stream(builder()?, scan_file(&segments, &layout)?)?
        .try_collect::<Vec<_>>()
        .await?;
    assert_arrays_eq!(
        ChunkedArray::try_new(actual, result_dtype.clone())?,
        ChunkedArray::try_new(expected, result_dtype)?,
        &mut session.create_execution_ctx()
    );
    Ok(())
}

/// Scans over one layout reader share its lowered file, and with it what executions learn, such
/// as dictionary values; scans over another reader of the same layout do not.
#[tokio::test(flavor = "multi_thread")]
async fn scans_over_one_reader_share_the_file() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_layout(&session).await?;
    let reader = || {
        layout.new_reader(
            "".into(),
            Arc::clone(&segments),
            &session,
            &Default::default(),
        )
    };
    let first = reader()?;
    let shared = shared_file(&first, scan_file(&segments, &layout)?)?;
    assert!(Arc::ptr_eq(
        &shared,
        &shared_file(&first, scan_file(&segments, &layout)?)?
    ));
    assert!(!Arc::ptr_eq(
        &shared,
        &shared_file(&reader()?, scan_file(&segments, &layout)?)?
    ));
    Ok(())
}

#[tokio::test]
async fn registry_does_not_retain_inactive_scan_caches() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_layout(&session).await?;
    let reader = layout.new_reader(
        "".into(),
        Arc::clone(&segments),
        &session,
        &Default::default(),
    )?;
    let shared = shared_file(&reader, scan_file(&segments, &layout)?)?;
    let weak = Arc::downgrade(&shared);
    drop(shared);
    assert!(weak.upgrade().is_none());
    assert_eq!(Arc::strong_count(&reader), 1);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn reader_cache_retains_dictionary_values_but_not_codes(
    #[values(false, true)] cached: bool,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_dict_layout(&session).await?;
    let mut values = BTreeSet::new();
    for node in layout.depth_first_traversal() {
        let node = node?;
        if node.is::<Dict>() {
            values.extend(segment_ids(
                &node.slot(0)?.ok_or_else(|| vortex_err!("missing values"))?,
            )?);
        }
    }
    assert!(!values.is_empty());
    let recording = Arc::new(RecordingSegments {
        inner: segments,
        reads: Default::default(),
        registered: Default::default(),
    });
    let source: Arc<dyn SegmentSource> = Arc::<RecordingSegments>::clone(&recording);
    let reader = layout.new_reader(
        "".into(),
        Arc::clone(&source),
        &session,
        &Default::default(),
    )?;
    let mut file = scan_file(&source, &layout)?;
    file.plans = cached.then(|| Arc::new(FilePlans::default()));
    let mut first_reads = BTreeSet::new();
    for iteration in 0..2 {
        recording.reads.lock().clear();
        let arrays = v2::into_stream(
            ScanBuilder::new(session.clone(), Arc::clone(&reader)),
            file.clone(),
        )?
        .try_collect::<Vec<_>>()
        .await?;
        let expected = VarBinArray::from_iter_nonnull(
            (0..4).flat_map(|chunk| {
                (0..CHUNK_ROWS).map(move |row| WORDS[((row * 7 + chunk) % 4) as usize])
            }),
            DType::Utf8(NonNullable),
        );
        assert_arrays_eq!(
            ChunkedArray::try_new(arrays, DType::Utf8(NonNullable))?,
            expected,
            &mut session.create_execution_ctx()
        );
        let reads = recording.reads.lock().clone();
        if iteration == 0 {
            assert!(values.is_subset(&reads));
            first_reads = reads;
        } else {
            let expected = if cached {
                first_reads.difference(&values).copied().collect()
            } else {
                first_reads.clone()
            };
            assert!(!expected.is_empty());
            assert_eq!(reads, expected);
        }
    }
    Ok(())
}

#[tokio::test]
async fn reader_plan_cache_has_no_ownership_cycle() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_layout(&session).await?;
    let reader = layout.new_reader(
        "".into(),
        Arc::clone(&segments),
        &session,
        &Default::default(),
    )?;
    let mut file = scan_file(&segments, &layout)?;
    let plans = Arc::new(FilePlans::default());
    let weak_plans = Arc::downgrade(&plans);
    file.plans = Some(plans);
    let shared = shared_file(&reader, file.clone())?;
    let root = shared.root.clone();
    let weak_shared = Arc::downgrade(&shared);
    drop(shared);
    assert!(weak_shared.upgrade().is_none());
    let shared = shared_file(&reader, file.clone())?;
    assert!(PlanRef::ptr_eq(&root, &shared.root));
    drop(shared);
    drop(file);
    assert!(weak_plans.upgrade().is_none());
    assert_eq!(Arc::strong_count(&reader), 1);
    Ok(())
}

#[tokio::test]
async fn cached_dictionary_accepts_different_queries() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_dict_layout(&session).await?;
    let reader = layout.new_reader(
        "".into(),
        Arc::clone(&segments),
        &session,
        &Default::default(),
    )?;
    let mut file = scan_file(&segments, &layout)?;
    file.plans = Some(Arc::default());
    let dtype = DType::Utf8(NonNullable);
    for (projection, filter) in [
        (root(), eq(root(), lit("apple"))),
        (byte_length(root()), gt(byte_length(root()), lit(3u64))),
        (root(), eq(root(), lit("banana"))),
    ] {
        let builder = || -> VortexResult<_> {
            Ok(ScanBuilder::new(session.clone(), Arc::clone(&reader))
                .with_projection(projection.bind(&dtype)?)
                .with_filter(filter.bind(&dtype)?))
        };
        let result_dtype = builder()?.dtype()?;
        let expected = builder()?.into_stream()?.try_collect::<Vec<_>>().await?;
        let actual = v2::into_stream(builder()?, file.clone())?
            .try_collect::<Vec<_>>()
            .await?;
        assert_arrays_eq!(
            ChunkedArray::try_new(actual, result_dtype.clone())?,
            ChunkedArray::try_new(expected, result_dtype)?,
            &mut session.create_execution_ctx()
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_scans_over_one_reader_share_initialization() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_layout(&session).await?;
    let reader = layout.new_reader(
        "".into(),
        Arc::clone(&segments),
        &session,
        &Default::default(),
    )?;
    let file = scan_file(&segments, &layout)?;
    let barrier = Barrier::new(8);
    let shared = thread::scope(|scope| {
        let threads = (0..8)
            .map(|_| {
                let file = file.clone();
                let barrier = &barrier;
                let reader = &reader;
                scope.spawn(move || {
                    barrier.wait();
                    shared_file(reader, file)
                })
            })
            .collect::<Vec<_>>();
        threads
            .into_iter()
            .map(|thread| {
                thread
                    .join()
                    .map_err(|_| vortex_err!("file preparation thread panicked"))?
            })
            .collect::<VortexResult<Vec<_>>>()
    })?;
    for other in &shared[1..] {
        assert!(Arc::ptr_eq(&shared[0], other));
    }
    Ok(())
}

/// Zone pruning follows a dynamic comparison the engine tightens during the scan: with no bound
/// every zone is read, and once the bound is above 2500 a later execution skips the two zones
/// below it, without reading the zone table again.
#[tokio::test(flavor = "multi_thread")]
async fn zone_pruning_follows_dynamic_comparisons() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_zoned_layout(&session).await?;
    let data = layout
        .slot(0)?
        .ok_or_else(|| vortex_error::vortex_err!("no data"))?;
    let data_chunks = data.children()?;

    let bound = Arc::new(Mutex::new(None::<i32>));
    let filter = {
        let bound = Arc::clone(&bound);
        dynamic(
            CompareOperator::Gt,
            move || bound.lock().map(Into::into),
            DTYPE,
            true,
            root(),
        )
    };
    let recording = Arc::new(RecordingSegments {
        registered: Default::default(),
        inner: Arc::clone(&segments),
        reads: Arc::default(),
    });
    let mut file = scan_file(&segments, &layout)?;
    file.segments = Arc::clone(&recording) as _;
    let reader = layout.new_reader(
        "".into(),
        Arc::clone(&segments),
        &session,
        &Default::default(),
    )?;
    let scan = v2::prepare(
        ScanBuilder::new(session.clone(), reader).with_filter(filter.bind(&DTYPE)?),
        file,
    )?;

    let everything = await_tasks(DTYPE, scan.execute(None)?).await?;
    assert_eq!(everything.len(), 4000);
    for chunk in &data_chunks {
        assert!(segment_ids(chunk)?.is_subset(&recording.reads.lock()));
    }

    *bound.lock() = Some(2500);
    recording.reads.lock().clear();
    let above = await_tasks(DTYPE, scan.execute(None)?).await?;
    assert_arrays_eq!(
        above,
        Buffer::from_iter(2501..4000).into_array(),
        &mut session.create_execution_ctx()
    );
    let mut expected_reads = segment_ids(&data_chunks[2])?;
    expected_reads.extend(segment_ids(&data_chunks[3])?);
    assert_eq!(*recording.reads.lock(), expected_reads);
    Ok(())
}

/// Four 1000-row chunks of a struct `{a, b}` with `a` over `0..4000` and `b = 2a`.
async fn write_struct_layout(
    session: &VortexSession,
    field_count: usize,
) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef)> {
    let segments = Arc::new(TestSegments::default());
    let (mut sequence_id, eof) = SequenceId::root().split();
    let chunks = (0..4)
        .map(|chunk| {
            let a = Buffer::from_iter(chunk * CHUNK_ROWS..(chunk + 1) * CHUNK_ROWS);
            let b =
                Buffer::from_iter((chunk * CHUNK_ROWS..(chunk + 1) * CHUNK_ROWS).map(|a| 2 * a));
            let mut fields = vec![
                ("a".to_owned(), a.into_array()),
                ("b".to_owned(), b.into_array()),
            ];
            for field in 2..field_count {
                let increment = i32::try_from(field)?;
                fields.push((
                    format!("f{field}"),
                    Buffer::from_iter(
                        (chunk * CHUNK_ROWS..(chunk + 1) * CHUNK_ROWS).map(|a| a + increment),
                    )
                    .into_array(),
                ));
            }
            let array = StructArray::from_fields(&fields)?;
            Ok((sequence_id.advance(), array.into_array()))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let dtype = chunks[0].1.dtype().clone();
    let layout = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default())
        .write_stream(
            ArrayContext::empty().into(),
            Arc::<TestSegments>::clone(&segments),
            SequentialStreamAdapter::new(dtype, stream::iter(chunks.into_iter().map(Ok)))
                .sendable(),
            eof,
            session,
        )
        .await?;
    Ok((segments, layout))
}

/// Narrow and wide projections retain one batch per selected cut. Both the joined result and
/// the stream preserve the reference executor's values while unused IO interests are withdrawn.
#[rstest::rstest]
#[case(2)]
#[case(8)]
#[tokio::test(flavor = "multi_thread")]
async fn struct_projection_splits_match_default(#[case] field_count: usize) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_struct_layout(&session, field_count).await?;
    let builder = || -> VortexResult<ScanBuilder<ArrayRef>> {
        let reader = layout.new_reader(
            "".into(),
            Arc::clone(&segments),
            &session,
            &Default::default(),
        )?;
        let filter = gt(get_item("a", root()), lit(1500_i32)).bind(layout.dtype())?;
        Ok(ScanBuilder::new(session.clone(), reader).with_filter(filter))
    };
    let dtype = builder()?.dtype()?;
    let expected = await_tasks(dtype.clone(), builder()?.prepare()?.execute(None)?).await?;

    let scan = v2::prepare(builder()?, scan_file(&segments, &layout)?)?;
    let tasks = scan.execute(None)?;
    assert_eq!(tasks.len(), 1, "4000 rows make one filter split");
    let joined = await_tasks(dtype.clone(), tasks).await?;
    assert_arrays_eq!(joined, expected, &mut session.create_execution_ctx());

    let streamed = v2::into_stream(builder()?, scan_file(&segments, &layout)?)?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(
        streamed.len(),
        3,
        "the three chunks with a > 1500 are projection splits"
    );
    assert_arrays_eq!(
        ChunkedArray::try_new(streamed, dtype)?,
        expected,
        &mut session.create_execution_ctx()
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn wide_projection_with_mismatched_chunks_and_mask_holes_matches_default() -> VortexResult<()>
{
    let session = new_session().with_tokio();
    let segments = Arc::new(TestSegments::default());
    let mut children = Vec::new();
    for field in 0..8_i32 {
        let (mut sequence_id, eof) = SequenceId::root().split();
        let chunk_rows = 700 + field * 91;
        let chunks = (0..4000)
            .step_by(usize::try_from(chunk_rows)?)
            .map(|start| {
                let rows = start..(start + chunk_rows).min(4000);
                let values = Buffer::from_iter(rows.map(|row| row + field * 10_000));
                Ok((sequence_id.advance(), values.into_array()))
            })
            .collect::<Vec<_>>();
        children.push(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default())
                .write_stream(
                    ArrayContext::empty().into(),
                    Arc::<TestSegments>::clone(&segments),
                    SequentialStreamAdapter::new(DTYPE, stream::iter(chunks)).sendable(),
                    eof,
                    &session,
                )
                .await?,
        );
    }
    let fields = StructFields::from_iter((0..8).map(|field| {
        (
            if field == 0 {
                "a".to_owned()
            } else {
                format!("f{field}")
            },
            DTYPE,
        )
    }));
    let dtype = DType::Struct(fields, NonNullable);
    let layout = StructLayout::new(4000, dtype.clone(), children).into_layout();
    let segments = segments as Arc<dyn SegmentSource>;
    let a = || get_item("a", root());
    let filter = or(
        lt(a(), lit(10_i32)),
        or(
            and(gt(a(), lit(1499_i32)), lt(a(), lit(1510_i32))),
            gt(a(), lit(3500_i32)),
        ),
    )
    .bind(&dtype)?;
    let builder = || -> VortexResult<ScanBuilder<ArrayRef>> {
        let reader = layout.new_reader(
            "".into(),
            Arc::clone(&segments),
            &session,
            &Default::default(),
        )?;
        Ok(ScanBuilder::new(session.clone(), reader).with_filter(filter.clone()))
    };
    let expected = await_tasks(dtype.clone(), builder()?.prepare()?.execute(None)?).await?;
    let actual = v2::into_stream(builder()?, scan_file(&segments, &layout)?)?
        .try_collect::<Vec<_>>()
        .await?;
    assert_arrays_eq!(
        ChunkedArray::try_new(actual, dtype)?,
        expected,
        &mut session.create_execution_ctx()
    );
    Ok(())
}

#[rstest]
#[case(1500, Some(1000..4000))]
#[case(4000, None)]
#[case(-1, Some(0..4000))]
#[tokio::test(flavor = "multi_thread")]
async fn file_pruning_precedes_data_split_creation(
    #[case] threshold: i32,
    #[case] expected_scope: Option<Range<u64>>,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_zoned_layout(&session).await?;
    let recording = Arc::new(RecordingSegments {
        registered: Default::default(),
        inner: Arc::clone(&segments),
        reads: Arc::default(),
    });
    let mut file = scan_file(&segments, &layout)?;
    file.segments = Arc::clone(&recording) as _;
    let reader = layout.new_reader(
        "".into(),
        Arc::clone(&segments),
        &session,
        &Default::default(),
    )?;
    let scan = v2::prepare(
        ScanBuilder::new(session.clone(), reader)
            .with_filter(gt(root(), lit(threshold)).bind(&DTYPE)?),
        file,
    )?;
    let splits = scan.pruned_split_plans(None).await?;
    assert_eq!(
        splits
            .iter()
            .map(|split| split.scope.rows.clone())
            .collect::<Vec<_>>(),
        expected_scope.into_iter().collect::<Vec<_>>()
    );
    let zones = layout
        .slot(1)?
        .ok_or_else(|| vortex_error::vortex_err!("no zones"))?;
    let zone_segments = segment_ids(&zones)?;
    assert_eq!(*recording.reads.lock(), zone_segments);
    assert!(recording.registered.lock().is_subset(&zone_segments));
    let mut arrays = Vec::new();
    for split in splits {
        arrays.extend(split.run(Arc::clone(scan.io())).await?);
    }
    assert_arrays_eq!(
        ChunkedArray::try_new(arrays, DTYPE)?,
        Buffer::from_iter((0..4000).filter(|value| *value > threshold)).into_array(),
        &mut session.create_execution_ctx()
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn file_pruning_rechecks_dynamic_bounds_after_splitting() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_zoned_layout(&session).await?;
    let bound = Arc::new(Mutex::new(None::<i32>));
    let filter = {
        let bound = Arc::clone(&bound);
        dynamic(
            CompareOperator::Gt,
            move || bound.lock().map(Into::into),
            DTYPE,
            true,
            root(),
        )
    };
    let reader = layout.new_reader(
        "".into(),
        Arc::clone(&segments),
        &session,
        &Default::default(),
    )?;
    let scan = v2::prepare(
        ScanBuilder::new(session.clone(), reader).with_filter(filter.bind(&DTYPE)?),
        scan_file(&segments, &layout)?,
    )?;
    let splits = scan.pruned_split_plans(None).await?;
    assert_eq!(splits[0].scope.rows, 0..4000);
    *bound.lock() = Some(2500);
    let mut arrays = Vec::new();
    for split in splits {
        arrays.extend(split.run(Arc::clone(scan.io())).await?);
    }
    assert_arrays_eq!(
        ChunkedArray::try_new(arrays, DTYPE)?,
        Buffer::from_iter(2501..4000).into_array(),
        &mut session.create_execution_ctx()
    );
    let splits = scan.pruned_split_plans(None).await?;
    assert_eq!(splits[0].scope.rows, 2000..4000);
    *bound.lock() = None;
    let splits = scan.pruned_split_plans(None).await?;
    assert_eq!(splits[0].scope.rows, 0..4000);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn file_pruning_keeps_disjoint_zones_in_one_filter_task() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_zoned_layout(&session).await?;
    let reader = layout.new_reader(
        "".into(),
        Arc::clone(&segments),
        &session,
        &Default::default(),
    )?;
    let filter = or(lt(root(), lit(500i32)), gt(root(), lit(3500i32))).bind(&DTYPE)?;
    let scan = v2::prepare(
        ScanBuilder::new(session.clone(), reader).with_filter(filter),
        scan_file(&segments, &layout)?,
    )?;
    let splits = scan.pruned_split_plans(None).await?;
    assert_eq!(splits.len(), 1);
    assert_eq!(splits[0].scope.rows, 0..4000);
    let mut arrays = Vec::new();
    for split in splits {
        arrays.extend(split.run(Arc::clone(scan.io())).await?);
    }
    assert_arrays_eq!(
        ChunkedArray::try_new(arrays, DTYPE)?,
        Buffer::from_iter((0..500).chain(3501..4000)).into_array(),
        &mut session.create_execution_ctx()
    );
    Ok(())
}
