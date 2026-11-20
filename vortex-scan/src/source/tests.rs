// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
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
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use super::PlanDataSource;
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

struct NoSegments;

impl SegmentSource for NoSegments {
    fn request(&self, _id: SegmentId) -> SegmentFuture {
        Box::pin(async { vortex_bail!("an in-memory source must not request bytes") })
    }
}

fn source() -> PlanDataSource {
    source_of(20)
}

fn source_of(row_count: u64) -> PlanDataSource {
    let plan = RowSource::plan(row_count, vec![4, 7, 9], None, false);
    let input = ScanInput {
        dtype: plan.dtype().clone(),
        root: LazyPlanRef::ready(plan),
        locations: Arc::new([]),
        segments: Arc::new(NoSegments),
        cache: None,
    };
    PlanDataSource::new(VortexSession::empty().with_tokio(), input).with_row_count(row_count)
}

async fn partitions(request: ScanRequest) -> VortexResult<Vec<PartitionRef>> {
    partitions_of(&source(), request).await
}

async fn partitions_of(
    source: &PlanDataSource,
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

/// A request that selects no rows of the source scans one partition, which reads nothing.
#[rstest]
#[case::zero_limit(ScanRequest { limit: Some(0), ..Default::default() })]
#[case::empty_rows(ScanRequest { row_range: Some(30..40), ..Default::default() })]
#[case::other_units(ScanRequest { partition_range: Some(1..4), ..Default::default() })]
#[case::unit_excluded(ScanRequest {
    partition_selection: Selection::ExcludeRoaring(RoaringTreemap::from_iter([0_u64])),
    ..Default::default()
})]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn requests_without_rows_read_nothing(#[case] request: ScanRequest) -> VortexResult<()> {
    let scan = source().scan(request).await?;
    assert_eq!(scan.row_count(), Precision::exact(0_u64));
    let partitions: Vec<PartitionRef> = scan.partitions().try_collect().await?;
    assert_eq!(collect(partitions).await?.len(), 0);
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

/// Selecting the source's unit, by range or by selection, returns its one partition.
#[rstest]
#[case::range(Some(0..1), Selection::All)]
#[case::wide_range(Some(0..5), Selection::All)]
#[case::selection(None, Selection::IncludeRoaring(RoaringTreemap::from_iter([0_u64, 3])))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn selecting_the_unit_returns_its_partition(
    #[case] partition_range: Option<Range<u64>>,
    #[case] partition_selection: Selection,
) -> VortexResult<()> {
    let partitions = partitions(ScanRequest {
        partition_selection,
        partition_range,
        ..Default::default()
    })
    .await?;
    assert_eq!(partitions.len(), 1);
    assert_eq!(partitions[0].index(), 0);
    assert_eq!(
        sorted_values(partitions).await?,
        (0..20).collect::<Vec<_>>()
    );
    Ok(())
}
