// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;

use futures::TryStreamExt;
use roaring::RoaringTreemap;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::expr::checked_add;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::expr::stats::Precision;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::request::SegmentReader;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use super::MIN_PARTITION_ROWS;
use super::PlannedDataSource;
use super::ScanInput;
use super::SegmentFuture;
use super::SegmentId;
use super::SegmentSource;
use crate::DataSource;
use crate::PartitionRef;
use crate::ScanRequest;
use crate::execution::pipeline::synthetic::RowSource;
use crate::plan::LazyPlanRef;
use crate::selection::Selection;
use crate::strict_sorted_buffer::StrictSortedBuffer;

struct NoSegments;

impl SegmentSource for NoSegments {
    fn request(&self, _id: SegmentId) -> SegmentFuture {
        Box::pin(async { vortex_bail!("an in-memory source must not request bytes") })
    }
}

/// Rows of the source that unordered, unlimited scans split into four partitions.
const SPLIT_ROWS: u64 = 4 * MIN_PARTITION_ROWS;

fn source() -> PlannedDataSource {
    source_of(20)
}

fn source_of(row_count: u64) -> PlannedDataSource {
    let plan = RowSource::plan(row_count, vec![4, 7, 9], None, false);
    let input = ScanInput {
        dtype: plan.dtype().clone(),
        root: LazyPlanRef::ready(plan),
        locations: Arc::new([]),
        reader: SegmentReader::Opaque(Arc::new(NoSegments)),
        cache: None,
    };
    PlannedDataSource::new(VortexSession::empty().with_tokio(), input).with_row_count(row_count)
}

fn split_source(max_partitions: usize) -> VortexResult<PlannedDataSource> {
    let max_partitions = NonZeroUsize::new(max_partitions)
        .ok_or_else(|| vortex_err!("max_partitions must be positive"))?;
    Ok(source_of(SPLIT_ROWS).with_max_partitions(max_partitions))
}

async fn partitions(request: ScanRequest) -> VortexResult<Vec<PartitionRef>> {
    partitions_of(&source(), request).await
}

async fn partitions_of(
    source: &PlannedDataSource,
    request: ScanRequest,
) -> VortexResult<Vec<PartitionRef>> {
    source.scan(request).await?.partitions().try_collect().await
}

async fn collect(partitions: Vec<PartitionRef>) -> VortexResult<ArrayRef> {
    let dtype = source().dtype().clone();
    let mut arrays = Vec::new();
    for partition in partitions {
        let batches: Vec<ArrayRef> = partition.execute()?.try_collect().await?;
        arrays.extend(batches);
    }
    Ok(ChunkedArray::try_new(arrays, dtype)?.into_array())
}

#[rstest]
#[case::unordered(false)]
#[case::ordered(true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn execute_applies_the_request(#[case] ordered: bool) -> VortexResult<()> {
    let partitions = partitions(ScanRequest {
        projection: checked_add(root(), lit(100_u64)),
        filter: Some(gt(root(), lit(5_u64))),
        row_range: Some(2..18),
        ordered,
        ..Default::default()
    })
    .await?;
    assert_eq!(partitions.len(), 1);
    let mut ctx = VortexSession::empty().create_execution_ctx();
    let actual = collect(partitions)
        .await?
        .execute::<PrimitiveArray>(&mut ctx)?;
    let mut values = actual.as_slice::<u64>().to_vec();
    if !ordered {
        values.sort_unstable();
    }
    assert_eq!(values, (106_u64..118).collect::<Vec<_>>());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn execute_applies_the_limit() -> VortexResult<()> {
    let partitions = partitions(ScanRequest {
        filter: Some(gt(root(), lit(5_u64))),
        ordered: true,
        limit: Some(5),
        ..Default::default()
    })
    .await?;
    assert_arrays_eq!(
        collect(partitions).await?,
        Buffer::from_iter(6_u64..11).into_array(),
        &mut VortexSession::empty().create_execution_ctx()
    );
    Ok(())
}

#[tokio::test]
async fn partitions_report_their_rows() -> VortexResult<()> {
    let partitions = partitions(ScanRequest {
        row_range: Some(4..9),
        ..Default::default()
    })
    .await?;
    assert_eq!(partitions[0].row_count(), Precision::exact(5_u64));
    Ok(())
}

#[rstest]
#[case::zero_limit(ScanRequest { limit: Some(0), ..Default::default() })]
#[case::empty_rows(ScanRequest { row_range: Some(30..40), ..Default::default() })]
#[case::other_partitions(ScanRequest { partition_range: Some(1..4), ..Default::default() })]
#[tokio::test]
async fn requests_without_rows_have_no_partitions(
    #[case] request: ScanRequest,
) -> VortexResult<()> {
    assert!(partitions(request).await?.is_empty());
    Ok(())
}

/// The sorted values of every partition's output.
async fn sorted_values(partitions: Vec<PartitionRef>) -> VortexResult<Vec<u64>> {
    let mut ctx = VortexSession::empty().create_execution_ctx();
    let mut values = collect(partitions)
        .await?
        .execute::<PrimitiveArray>(&mut ctx)?
        .as_slice::<u64>()
        .to_vec();
    values.sort_unstable();
    Ok(values)
}

fn include(indices: impl IntoIterator<Item = u64>) -> VortexResult<Selection> {
    Ok(Selection::IncludeByIndex(StrictSortedBuffer::try_new(
        Buffer::from_iter(indices),
    )?))
}

#[rstest]
#[case::max_partitions(4, 4)]
#[case::row_floor(8, 4)]
#[case::single(1, 1)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unordered_scans_split_into_row_ranges(
    #[case] max_partitions: usize,
    #[case] expected: usize,
) -> VortexResult<()> {
    let source = split_source(max_partitions)?;
    let scan = source.scan(ScanRequest::default()).await?;
    assert_eq!(scan.partition_count(), Precision::exact(expected));
    let partitions: Vec<PartitionRef> = scan.partitions().try_collect().await?;
    assert_eq!(
        partitions.iter().map(|p| p.index()).collect::<Vec<_>>(),
        (0..expected).collect::<Vec<_>>()
    );
    for partition in &partitions {
        assert_eq!(
            partition.row_count(),
            Precision::exact(SPLIT_ROWS / expected as u64)
        );
    }
    let single = partitions_of(
        &source,
        ScanRequest {
            ordered: true,
            ..Default::default()
        },
    )
    .await?;
    assert_eq!(
        sorted_values(partitions).await?,
        sorted_values(single).await?
    );
    Ok(())
}

#[rstest]
#[case::range(ScanRequest { partition_range: Some(1..3), ..Default::default() }, vec![1, 2])]
#[case::selection(
    ScanRequest {
        partition_selection: Selection::IncludeRoaring(RoaringTreemap::from_iter([0_u64, 3])),
        ..Default::default()
    },
    vec![0, 3],
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn requests_select_split_partitions(
    #[case] request: ScanRequest,
    #[case] expected: Vec<usize>,
) -> VortexResult<()> {
    let partitions = partitions_of(&split_source(4)?, request).await?;
    assert_eq!(
        partitions.iter().map(|p| p.index()).collect::<Vec<_>>(),
        expected
    );
    let expected_values: Vec<u64> = expected
        .iter()
        .flat_map(|&i| i as u64 * MIN_PARTITION_ROWS..(i as u64 + 1) * MIN_PARTITION_ROWS)
        .collect();
    assert_eq!(sorted_values(partitions).await?, expected_values);
    Ok(())
}

#[rstest]
#[case::ordered(ScanRequest { ordered: true, ..Default::default() })]
#[case::limited(ScanRequest { limit: Some(10), ..Default::default() })]
#[tokio::test]
async fn ordered_and_limited_scans_keep_one_partition(
    #[case] request: ScanRequest,
) -> VortexResult<()> {
    let partitions = partitions_of(&split_source(4)?, request).await?;
    assert_eq!(partitions.len(), 1);
    assert_eq!(partitions[0].index(), 0);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn split_partitions_apply_the_request() -> VortexResult<()> {
    let row_range = 1_000..SPLIT_ROWS - 1_000;
    let selected = (0..SPLIT_ROWS).step_by(997);
    let partitions = partitions_of(
        &split_source(4)?,
        ScanRequest {
            filter: Some(gt(root(), lit(100_000_u64))),
            row_range: Some(row_range.clone()),
            selection: include(selected.clone())?,
            ..Default::default()
        },
    )
    .await?;
    assert_eq!(partitions.len(), 4);
    let expected: Vec<u64> = selected
        .filter(|row| row_range.contains(row) && *row > 100_000)
        .collect();
    assert_eq!(sorted_values(partitions).await?, expected);
    Ok(())
}
