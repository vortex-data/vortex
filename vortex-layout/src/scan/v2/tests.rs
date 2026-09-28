// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::TryStreamExt;
use futures::stream;
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
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSessionExt;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::row_idx::row_idx;
use crate::scan::planning::SegmentLocation;
use crate::scan::scan_builder::ScanBuilder;
use crate::scan::v2;
use crate::scan::v2::ScanFile;
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
/// so every location is a placeholder with the alignment the test segments already have.
fn scan_file(segments: &Arc<dyn SegmentSource>, layout: &LayoutRef) -> VortexResult<ScanFile> {
    let mut count = 0;
    for layout in layout.depth_first_traversal() {
        for id in layout?.segment_ids() {
            count = count.max(*id as usize + 1);
        }
    }
    let placeholder = SegmentLocation {
        offset: 0,
        length: 0,
        alignment: Alignment::none(),
    };
    Ok(ScanFile {
        layout: Arc::clone(layout),
        locations: vec![placeholder; count].into(),
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
        builder = builder.with_filter(gt(root(), lit(1500_i32)).bind(&DTYPE)?);
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
fn stream_matches_default(#[case] case: Case) -> VortexResult<()> {
    block_on(|handle| async move {
        let session = new_session().with_handle(handle);
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
    })
}

#[rstest]
#[case::whole_scan(case(true, None, None), None)]
#[case::execute_range(case(true, None, None), Some(700..3300))]
#[case::both_ranges(case(false, Some(500..2500), None), Some(2000..4000))]
#[case::empty_range(case(true, None, None), Some(1200..1200))]
fn execute_matches_default(
    #[case] case: Case,
    #[case] execute_range: Option<Range<u64>>,
) -> VortexResult<()> {
    block_on(|handle| async move {
        let session = new_session().with_handle(handle);
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
    })
}

#[test]
fn filter_with_limit_is_rejected() -> VortexResult<()> {
    block_on(|handle| async move {
        let session = new_session().with_handle(handle);
        let (segments, layout) = write_layout(&session).await?;
        let case = case(true, None, Some(10));
        let file = scan_file(&segments, &layout)?;
        assert!(v2::prepare(builder(&session, &segments, &layout, &case)?, file).is_err());
        Ok(())
    })
}
