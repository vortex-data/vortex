// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryStreamExt;
use futures::stream;
use parking_lot::Mutex;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::expr::and;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::root;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::row_idx::row_idx;
use crate::layouts::zoned::writer::ZonedLayoutOptions;
use crate::layouts::zoned::writer::ZonedStrategy;
use crate::scan::planning::SegmentLocation;
use crate::scan::scan_builder::ScanBuilder;
use crate::scan::v2;
use crate::scan::v2::ScanFile;
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
    reads: Arc<Mutex<BTreeSet<u32>>>,
}

impl SegmentSource for RecordingSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let read = self.inner.request(id);
        let reads = Arc::clone(&self.reads);
        async move {
            reads.lock().insert(*id);
            read.await
        }
        .boxed()
    }
}

/// The ids of the segments below `layout`.
fn segment_ids(layout: &LayoutRef) -> VortexResult<BTreeSet<u32>> {
    let mut ids = BTreeSet::new();
    for layout in layout.depth_first_traversal() {
        ids.extend(layout?.segment_ids().into_iter().map(|id| *id));
    }
    Ok(ids)
}

/// Pruning drops the zones whose statistics prove the filter false: of four 1000-row zones over
/// `0..4000`, the first cannot hold a value above 1500, so V2 reads the zone table and the other
/// three chunks, never the first, and still returns what the default path returns.
#[tokio::test(flavor = "multi_thread")]
async fn zone_pruning_reads_only_zones_that_can_match() -> VortexResult<()> {
    let session = new_session().with_tokio();
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
        &session,
    )
    .await?;

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
