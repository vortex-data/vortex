// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use futures::TryStreamExt;
use itertools::Itertools;
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
use vortex_utils::parallelism::get_available_parallelism;

use super::MIN_PARTITION_ROWS;
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
use crate::strict_sorted_buffer::StrictSortedBuffer;

struct NoSegments;

impl SegmentSource for NoSegments {
    fn request(&self, _id: SegmentId) -> SegmentFuture {
        Box::pin(async { vortex_bail!("an in-memory source must not request bytes") })
    }
}

/// Rows of the source that unordered, unlimited scans split into four partitions.
const SPLIT_ROWS: u64 = 4 * MIN_PARTITION_ROWS;

fn source() -> PlanDataSource {
    source_of(20)
}

fn source_of(row_count: u64) -> PlanDataSource {
    let plan = RowSource::plan(row_count, vec![4, 7, 9], None, false);
    let input = ScanInput {
        dtype: plan.dtype().clone(),
        root: LazyPlanRef::ready(plan),
        locations: Arc::new([]),
        reader: SegmentReader::Opaque(Arc::new(NoSegments)),
        cache: None,
    };
    PlanDataSource::new(VortexSession::empty().with_tokio(), input).with_row_count(row_count)
}

fn split_source() -> PlanDataSource {
    source_of(SPLIT_ROWS)
}

/// A request for an unordered, unlimited scan aiming for `target` partitions.
fn split_request(target: usize) -> VortexResult<ScanRequest> {
    Ok(ScanRequest {
        target_partitions: Some(
            NonZeroUsize::new(target).ok_or_else(|| vortex_err!("target must be positive"))?,
        ),
        ..Default::default()
    })
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

#[rstest]
#[case::zero_limit(ScanRequest { limit: Some(0), ..Default::default() })]
#[case::empty_rows(ScanRequest { row_range: Some(30..40), ..Default::default() })]
#[case::other_units(ScanRequest { partition_range: Some(1..4), ..Default::default() })]
#[case::unit_excluded(ScanRequest {
    partition_selection: Selection::ExcludeRoaring(RoaringTreemap::from_iter([0_u64])),
    ..Default::default()
})]
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

/// The request's hint sets the split count, below or above the default, capped so that no
/// partition covers fewer than `MIN_PARTITION_ROWS` rows. Without a hint the source splits by the
/// available parallelism.
#[rstest]
#[case::fewer(Some(2), 2)]
#[case::target(Some(4), 4)]
#[case::row_floor(Some(8), 4)]
#[case::single(Some(1), 1)]
#[case::default(None, get_available_parallelism().unwrap_or(1).min(4))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unordered_scans_split_into_row_ranges(
    #[case] target: Option<usize>,
    #[case] expected: usize,
) -> VortexResult<()> {
    let source = split_source();
    let request = match target {
        Some(target) => split_request(target)?,
        None => ScanRequest::default(),
    };
    let scan = source.scan(request).await?;
    assert_eq!(scan.partition_count(), Precision::exact(expected));
    let partitions: Vec<PartitionRef> = scan.partitions().try_collect().await?;
    assert_eq!(partitions.len(), expected);
    // The source is one unit, so every split shares its index.
    assert!(partitions.iter().all(|p| p.index() == 0));
    assert_near_equal_splits(&partitions, SPLIT_ROWS)?;
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

/// Asserts that `partitions` cover `rows` rows in splits whose lengths differ by at most one.
fn assert_near_equal_splits(partitions: &[PartitionRef], rows: u64) -> VortexResult<()> {
    let lengths = partitions
        .iter()
        .map(|p| {
            p.row_count()
                .as_exact()
                .ok_or_else(|| vortex_err!("split partitions know their rows"))
        })
        .collect::<VortexResult<Vec<u64>>>()?;
    assert_eq!(lengths.iter().sum::<u64>(), rows);
    let (min, max) = lengths
        .iter()
        .copied()
        .minmax()
        .into_option()
        .ok_or_else(|| vortex_err!("a split scan has partitions"))?;
    assert!(
        max - min <= 1,
        "split lengths {lengths:?} are not near-equal"
    );
    Ok(())
}

/// A hint above the available parallelism splits a large enough source further than the default.
#[tokio::test]
async fn hints_above_the_default_split_further() -> VortexResult<()> {
    let target = get_available_parallelism().unwrap_or(1) + 2;
    let rows = target as u64 * MIN_PARTITION_ROWS;
    let partitions = partitions_of(&source_of(rows), split_request(target)?).await?;
    assert_eq!(partitions.len(), target);
    assert!(partitions.iter().all(|p| p.index() == 0));
    assert_near_equal_splits(&partitions, rows)
}

/// Selecting the source's unit, by range or by selection, returns every split of it.
#[rstest]
#[case::range(Some(0..1), Selection::All)]
#[case::wide_range(Some(0..5), Selection::All)]
#[case::selection(None, Selection::IncludeRoaring(RoaringTreemap::from_iter([0_u64, 3])))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn selecting_the_unit_returns_every_split(
    #[case] partition_range: Option<Range<u64>>,
    #[case] partition_selection: Selection,
) -> VortexResult<()> {
    let request = ScanRequest {
        partition_selection,
        partition_range,
        ..split_request(4)?
    };
    let partitions = partitions_of(&split_source(), request).await?;
    assert_eq!(partitions.len(), 4);
    assert!(partitions.iter().all(|p| p.index() == 0));
    assert_eq!(
        sorted_values(partitions).await?,
        (0..SPLIT_ROWS).collect::<Vec<_>>()
    );
    Ok(())
}

#[rstest]
#[case::ordered(ScanRequest { ordered: true, ..Default::default() })]
#[case::limited(ScanRequest { limit: Some(10), ..Default::default() })]
#[tokio::test]
async fn ordered_and_limited_scans_keep_one_partition(
    #[case] request: ScanRequest,
) -> VortexResult<()> {
    let request = ScanRequest {
        target_partitions: NonZeroUsize::new(4),
        ..request
    };
    let partitions = partitions_of(&split_source(), request).await?;
    assert_eq!(partitions.len(), 1);
    assert_eq!(partitions[0].index(), 0);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn split_partitions_apply_the_request() -> VortexResult<()> {
    let row_range = 1_000..SPLIT_ROWS - 1_000;
    let selected = (0..SPLIT_ROWS).step_by(997);
    let partitions = partitions_of(
        &split_source(),
        ScanRequest {
            filter: Some(gt(root(), lit(100_000_u64))),
            row_range: Some(row_range.clone()),
            selection: include(selected.clone())?,
            ..split_request(4)?
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
