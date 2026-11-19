// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
use parking_lot::Mutex;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::expr::eq;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::expr::stats::Precision;
use vortex_array::stats::StatsSet;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::request::SegmentReader;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use super::DeferredDataSource;
use super::UnionDataSource;
use crate::DataSource;
use crate::DataSourceRef;
use crate::DataSourceScanRef;
use crate::PartitionRef;
use crate::ScanRequest;
use crate::execution::pipeline::synthetic::RowSource;
use crate::plan::LazyPlanRef;
use crate::selection::Selection;
use crate::source::PlanDataSource;
use crate::source::ScanInput;
use crate::source::SegmentFuture;
use crate::source::SegmentId;
use crate::source::SegmentSource;
use crate::strict_sorted_buffer::StrictSortedBuffer;

struct NoSegments;

impl SegmentSource for NoSegments {
    fn request(&self, _id: SegmentId) -> SegmentFuture {
        Box::pin(async { vortex_bail!("an in-memory source must not request bytes") })
    }
}

fn session() -> VortexSession {
    VortexSession::empty().with_tokio()
}

/// A child of `rows` rows, each the row's index as a `u64`, emitted in arrays of two rows.
fn child(rows: usize) -> DataSourceRef {
    let plan = RowSource::plan(rows as u64, vec![2; rows.div_ceil(2)], None, false);
    let input = ScanInput {
        dtype: plan.dtype().clone(),
        root: LazyPlanRef::ready(plan),
        locations: Arc::new([]),
        reader: SegmentReader::Opaque(Arc::new(NoSegments)),
        cache: None,
    };
    Arc::new(PlanDataSource::new(session(), input).with_row_count(rows as u64))
}

/// Opens its source, counting the opens; a deferred child without a source skips itself.
struct CountingDeferred {
    source: Option<DataSourceRef>,
    opens: AtomicUsize,
}

impl CountingDeferred {
    fn new(source: Option<DataSourceRef>) -> Arc<Self> {
        Arc::new(Self {
            source,
            opens: AtomicUsize::new(0),
        })
    }

    fn opens(&self) -> usize {
        self.opens.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl DeferredDataSource for CountingDeferred {
    async fn open(&self) -> VortexResult<Option<DataSourceRef>> {
        self.opens.fetch_add(1, Ordering::Relaxed);
        Ok(self.source.clone())
    }
}

/// A source over children of the given row counts, the first opened and the rest deferred.
fn union_of(rows: &[usize]) -> (UnionDataSource, Vec<Arc<CountingDeferred>>) {
    let deferred: Vec<_> = rows[1..]
        .iter()
        .map(|&rows| CountingDeferred::new(Some(child(rows))))
        .collect();
    let source = UnionDataSource::new_with_first(
        child(rows[0]),
        deferred.iter().map(|f| Arc::clone(f) as _).collect(),
        vec![],
        &session(),
    );
    (source, deferred)
}

async fn partitions(
    source: &UnionDataSource,
    request: ScanRequest,
) -> VortexResult<Vec<PartitionRef>> {
    source.scan(request).await?.partitions().try_collect().await
}

/// Executes each partition in turn, returning its index and values.
async fn execute(partitions: Vec<PartitionRef>) -> VortexResult<Vec<(usize, Vec<u64>)>> {
    let mut ctx = session().create_execution_ctx();
    let mut results = Vec::new();
    for partition in partitions {
        let index = partition.index();
        let arrays: Vec<ArrayRef> = partition.execute()?.try_collect().await?;
        let mut values = Vec::new();
        for array in arrays {
            values.extend_from_slice(array.execute::<PrimitiveArray>(&mut ctx)?.as_slice::<u64>());
        }
        results.push((index, values));
    }
    results.sort_by_key(|(index, _)| *index);
    Ok(results)
}

async fn scan_all(
    source: &UnionDataSource,
    request: ScanRequest,
) -> VortexResult<Vec<(usize, Vec<u64>)>> {
    execute(partitions(source, request).await?).await
}

#[rstest]
#[case::unordered(false)]
#[case::ordered(true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scans_every_child(#[case] ordered: bool) -> VortexResult<()> {
    let (source, _) = union_of(&[3, 5, 2]);
    let request = ScanRequest {
        ordered,
        ..Default::default()
    };
    let expected = vec![
        (0, (0..3).collect::<Vec<u64>>()),
        (1, (0..5).collect()),
        (2, (0..2).collect()),
    ];

    // A second scan of the source reports the same indices.
    assert_eq!(scan_all(&source, request.clone()).await?, expected);
    assert_eq!(scan_all(&source, request).await?, expected);
    Ok(())
}

#[tokio::test]
async fn ordered_scans_yield_children_in_order() -> VortexResult<()> {
    let (source, _) = union_of(&[3, 5, 2, 4]);
    let partitions = partitions(
        &source,
        ScanRequest {
            ordered: true,
            ..Default::default()
        },
    )
    .await?;
    let indices: Vec<usize> = partitions.iter().map(|p| p.index()).collect();
    assert_eq!(indices, vec![0, 1, 2, 3]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn children_apply_the_request() -> VortexResult<()> {
    let (source, _) = union_of(&[6, 8]);
    let values = scan_all(
        &source,
        ScanRequest {
            filter: Some(gt(root(), lit(2_u64))),
            row_range: Some(1..7),
            ordered: true,
            ..Default::default()
        },
    )
    .await?;
    assert_eq!(values, vec![(0, vec![3, 4, 5]), (1, vec![3, 4, 5, 6])]);
    Ok(())
}

fn indices(indices: &[u64]) -> StrictSortedBuffer<u64> {
    StrictSortedBuffer::try_new(Buffer::copy_from(indices)).vortex_expect("indices are sorted")
}

/// Partitions are selected by child before a deferred child is opened, so unselected children
/// are never opened.
#[rstest]
#[case::range(ScanRequest { partition_range: Some(2..3), ..Default::default() })]
#[case::include(ScanRequest {
    partition_selection: Selection::IncludeByIndex(indices(&[2])),
    ..Default::default()
})]
#[case::exclude(ScanRequest {
    partition_selection: Selection::ExcludeByIndex(indices(&[0, 1, 3])),
    ..Default::default()
})]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unselected_children_are_not_opened(#[case] request: ScanRequest) -> VortexResult<()> {
    let (source, deferred) = union_of(&[3, 5, 2, 4]);
    assert_eq!(scan_all(&source, request).await?, vec![(2, vec![0, 1])]);
    let opens: Vec<usize> = deferred.iter().map(|f| f.opens()).collect();
    assert_eq!(opens, vec![0, 1, 0]);
    Ok(())
}

/// An unordered limit is shared by every child, so together they return exactly the limit.
#[rstest]
#[case::no_filter(None)]
#[case::filter(Some(gt(root(), lit(0_u64))))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unordered_limit_is_shared_across_children(
    #[case] filter: Option<vortex_array::expr::Expression>,
) -> VortexResult<()> {
    let (source, _) = union_of(&[5, 5, 5]);
    let values = scan_all(
        &source,
        ScanRequest {
            filter,
            limit: Some(7),
            ordered: false,
            ..Default::default()
        },
    )
    .await?;
    let rows: usize = values.iter().map(|(_, values)| values.len()).sum();
    assert_eq!(rows, 7);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ordered_limit_applies_to_each_child() -> VortexResult<()> {
    let (source, _) = union_of(&[5, 5, 5]);
    let values = scan_all(
        &source,
        ScanRequest {
            limit: Some(3),
            ordered: true,
            ..Default::default()
        },
    )
    .await?;
    assert_eq!(
        values,
        vec![(0, vec![0, 1, 2]), (1, vec![0, 1, 2]), (2, vec![0, 1, 2])]
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn skipped_children_have_no_partitions() -> VortexResult<()> {
    let skipped = CountingDeferred::new(None);
    let source = UnionDataSource::new_with_first(
        child(3),
        vec![
            Arc::clone(&skipped) as _,
            CountingDeferred::new(Some(child(2))) as _,
        ],
        vec![],
        &session(),
    );
    let values = scan_all(&source, ScanRequest::default()).await?;
    assert_eq!(values, vec![(0, vec![0, 1, 2]), (2, vec![0, 1])]);
    assert_eq!(skipped.opens(), 1);
    Ok(())
}

/// A source with another dtype, which must never be scanned.
struct OtherDType(DType);

#[async_trait]
impl DataSource for OtherDType {
    fn dtype(&self) -> &DType {
        &self.0
    }

    async fn scan(&self, _request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        vortex_bail!("a child with another dtype must not be scanned")
    }

    async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_child_with_another_dtype_is_an_error() -> VortexResult<()> {
    let other: DataSourceRef = Arc::new(OtherDType(DType::Bool(Nullability::NonNullable)));
    let source = UnionDataSource::new_with_first(
        child(3),
        vec![CountingDeferred::new(Some(other)) as _],
        vec![],
        &session(),
    );
    let results: Vec<_> = source
        .scan(ScanRequest {
            ordered: true,
            ..Default::default()
        })
        .await?
        .partitions()
        .collect()
        .await;
    assert_eq!(results.len(), 2);
    assert!(results[0].is_ok());
    let error = results[1].as_ref().err().map(ToString::to_string);
    assert!(
        error.is_some_and(|error| error.contains("another dtype")),
        "expected a dtype error"
    );
    Ok(())
}

/// A filter that does not bind fails the children's scans, not the source's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn filter_binding_errors_are_deferred() -> VortexResult<()> {
    let (source, _) = union_of(&[3, 2]);
    let scan = source
        .scan(ScanRequest {
            filter: Some(eq(root(), lit(67_i32))),
            ..Default::default()
        })
        .await?;
    assert_eq!(scan.dtype(), source.dtype());
    let partitions: VortexResult<Vec<PartitionRef>> = scan.partitions().try_collect().await;
    assert!(partitions.is_err());
    Ok(())
}

#[tokio::test]
async fn projection_binding_errors_fail_the_scan() {
    let (source, _) = union_of(&[3]);
    let scan = source
        .scan(ScanRequest {
            projection: vortex_array::expr::get_item("missing", root()),
            ..Default::default()
        })
        .await;
    assert!(scan.is_err());
}

fn deferred(byte_sizes: Vec<Option<u64>>) -> UnionDataSource {
    let deferred = byte_sizes
        .iter()
        .map(|_| CountingDeferred::new(None) as _)
        .collect();
    UnionDataSource::new_deferred(
        DType::Bool(Nullability::NonNullable),
        deferred,
        byte_sizes,
        &session(),
    )
}

#[rstest]
#[case::all_known(vec![Some(10), Some(20), Some(30)], Precision::exact(60u64))]
#[case::some_known_extrapolates(vec![Some(10), None, Some(30)], Precision::inexact(60u64))]
#[case::none_known(vec![None, None], Precision::Absent)]
#[case::no_children(vec![], Precision::exact(0u64))]
fn byte_size_precision(#[case] sizes: Vec<Option<u64>>, #[case] expected: Precision<u64>) {
    assert_eq!(deferred(sizes).byte_size(), expected);
}

#[rstest]
#[case::all_opened(&[4], Precision::exact(4u64))]
#[case::deferred_extrapolates(&[4, 1, 1], Precision::inexact(12u64))]
fn row_count_precision(#[case] rows: &[usize], #[case] expected: Precision<u64>) {
    assert_eq!(union_of(rows).0.row_count(), expected);
}

#[test]
fn deferred_row_counts_are_absent() {
    assert_eq!(deferred(vec![None]).row_count(), Precision::Absent);
}

/// Records the partition hint of each request it scans, scanning a small child.
struct RecordsTarget {
    inner: DataSourceRef,
    targets: Mutex<Vec<Option<NonZeroUsize>>>,
}

impl RecordsTarget {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: child(2),
            targets: Mutex::new(Vec::new()),
        })
    }

    fn targets(&self) -> Vec<Option<usize>> {
        self.targets
            .lock()
            .iter()
            .map(|target| target.map(NonZeroUsize::get))
            .collect()
    }
}

#[async_trait]
impl DataSource for RecordsTarget {
    fn dtype(&self) -> &DType {
        self.inner.dtype()
    }

    async fn scan(&self, request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        self.targets.lock().push(request.target_partitions);
        self.inner.scan(request).await
    }

    async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

/// The hint is divided across the children the request selects, rounding up, and every child
/// aims for at least one partition. Deferred children count once their ordinal is selected.
#[rstest]
#[case::even(3, Some(6), None, Some(2))]
#[case::rounds_up(3, Some(4), None, Some(2))]
#[case::at_least_one(3, Some(1), None, Some(1))]
#[case::selected_only(4, Some(4), Some(1..3), Some(2))]
#[case::none(3, None, None, None)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn target_partitions_are_divided_across_selected_children(
    #[case] children: usize,
    #[case] target: Option<usize>,
    #[case] partition_range: Option<Range<u64>>,
    #[case] expected: Option<usize>,
) -> VortexResult<()> {
    let recorders: Vec<_> = (0..children).map(|_| RecordsTarget::new()).collect();
    let source = UnionDataSource::new_with_first(
        Arc::clone(&recorders[0]) as _,
        recorders[1..]
            .iter()
            .map(|recorder| CountingDeferred::new(Some(Arc::clone(recorder) as _)) as _)
            .collect(),
        vec![],
        &session(),
    );
    let request = ScanRequest {
        target_partitions: target.and_then(NonZeroUsize::new),
        partition_range: partition_range.clone(),
        ..Default::default()
    };
    scan_all(&source, request).await?;
    for (ordinal, recorder) in recorders.iter().enumerate() {
        let selected = partition_range
            .as_ref()
            .is_none_or(|range| range.contains(&(ordinal as u64)));
        let expected = if selected { vec![expected] } else { vec![] };
        assert_eq!(recorder.targets(), expected, "child {ordinal}");
    }
    Ok(())
}
