// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::TryStreamExt;
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
use vortex_io::request::SegmentReader;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

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

struct NoSegments;

impl SegmentSource for NoSegments {
    fn request(&self, _id: SegmentId) -> SegmentFuture {
        Box::pin(async { vortex_bail!("an in-memory source must not request bytes") })
    }
}

fn source() -> PlannedDataSource {
    let plan = RowSource::plan(20, vec![4, 7, 9], None, false);
    let input = ScanInput {
        dtype: plan.dtype().clone(),
        root: LazyPlanRef::ready(plan),
        locations: Arc::new([]),
        reader: SegmentReader::Opaque(Arc::new(NoSegments)),
        cache: None,
    };
    PlannedDataSource::new(VortexSession::empty().with_tokio(), input).with_row_count(20)
}

async fn partitions(request: ScanRequest) -> VortexResult<Vec<PartitionRef>> {
    source()
        .scan(request)
        .await?
        .partitions()
        .try_collect()
        .await
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
