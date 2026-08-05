use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::SchemaRef;
use datafusion::arrow::array::DictionaryArray;
use datafusion::arrow::array::Int32Array;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::array::StringArray;
use datafusion::arrow::array::StructArray;
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::datatypes::Schema;
use datafusion::arrow::datatypes::UInt32Type;
use datafusion::arrow::util::display::FormatOptions;
use datafusion::arrow::util::pretty::pretty_format_batches_with_options;
use datafusion::common::record_batch;
use datafusion::logical_expr::col;
use datafusion::logical_expr::lit;
use datafusion::physical_expr::planner::logical2physical;
use datafusion::physical_expr_adapter::DefaultPhysicalExprAdapterFactory;
use datafusion::scalar::ScalarValue;
use datafusion_common::stats::Precision;
use datafusion_execution::cache::DefaultFilesMetadataCache;
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions as df_expr;
use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion_physical_expr::projection::ProjectionExpr;
use futures::FutureExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::BoxStream;
use insta::assert_snapshot;
use itertools::Itertools;
use object_store::ObjectStore;
use object_store::memory::InMemory;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::buffer::Buffer;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::VortexWrite;
use vortex::io::object_store::ObjectStoreWrite;
use vortex::metrics::DefaultMetricsRegistry;
use vortex::scan::selection::Selection;
use vortex::session::VortexSession;
use vortex_arrow::FromArrowArray;

use super::*;
use crate::VortexAccessPlan;
use crate::convert::exprs::DefaultExpressionConvertor;
use crate::persistent::reader::DefaultVortexReaderFactory;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

type TestOpenFuture = BoxFuture<'static, DFResult<BoxStream<'static, DFResult<RecordBatch>>>>;

trait TestMorselizerExt {
    fn open(&self, file: PartitionedFile) -> DFResult<TestOpenFuture>;
}

impl TestMorselizerExt for VortexMorselizer {
    fn open(&self, file: PartitionedFile) -> DFResult<TestOpenFuture> {
        let planner = self.plan_file(file)?;
        Ok(read_morsels(planner).boxed())
    }
}

async fn read_morsels(
    planner: Box<dyn MorselPlanner>,
) -> DFResult<BoxStream<'static, DFResult<RecordBatch>>> {
    let mut planners = VecDeque::from([planner]);
    let mut batches = Vec::new();

    while let Some(planner) = planners.pop_front() {
        let Some(mut plan) = planner.plan()? else {
            continue;
        };

        for morsel in plan.take_morsels() {
            batches.extend(morsel.into_stream().try_collect::<Vec<_>>().await?);
        }
        planners.extend(plan.take_ready_planners());
        if let Some(pending_planner) = plan.take_pending_planner() {
            planners.push_back(pending_planner.await?);
        }
    }

    Ok(stream::iter(batches.into_iter().map(Ok)).boxed())
}

/// Test-only expr used to test error reporting.
#[derive(Debug, Eq, Hash, PartialEq)]
struct SnapshotErrorExpr;

impl fmt::Display for SnapshotErrorExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "snapshot_error")
    }
}

impl PhysicalExpr for SnapshotErrorExpr {
    fn data_type(&self, _input_schema: &Schema) -> DFResult<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &Schema) -> DFResult<bool> {
        Ok(false)
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }

    fn evaluate(&self, _batch: &RecordBatch) -> DFResult<datafusion_expr::ColumnarValue> {
        Err(DataFusionError::Internal(
            "intentional snapshot error".to_owned(),
        ))
    }

    fn children(&self) -> Vec<&PhysicalExprRef> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<PhysicalExprRef>,
    ) -> DFResult<PhysicalExprRef> {
        assert!(children.is_empty());
        Ok(self)
    }

    fn snapshot(&self) -> DFResult<Option<PhysicalExprRef>> {
        Err(DataFusionError::Internal(
            "intentional snapshot error".to_owned(),
        ))
    }
}

/// Build [`NaturalSplits`] from contiguous split ranges, the shape the layout walk produces.
fn natural_splits(total_size: u64, split_ranges: &[Range<u64>]) -> NaturalSplits {
    let mut row_boundaries = Vec::with_capacity(split_ranges.len() + 1);
    if let Some(first) = split_ranges.first() {
        row_boundaries.push(first.start);
        row_boundaries.extend(split_ranges.iter().map(|range| range.end));
    }
    NaturalSplits::new(row_boundaries.into(), total_size)
}

#[rstest]
#[case(0..3, 10, vec![0..2, 2..5, 5..10], Some(0..2))]
#[case(3..7, 10, vec![0..2, 2..5, 5..10], Some(2..5))]
#[case(1..8, 10, vec![0..1, 1..9, 9..10], Some(1..9))]
#[case(1..4, 16, vec![0..1, 1..2, 2..3, 3..4], None)]
#[case(0..1, 10, vec![0..2, 2..10], Some(0..2))]
fn test_split_aligned_row_range(
    #[case] byte_range: Range<u64>,
    #[case] total_size: u64,
    #[case] split_ranges: Vec<Range<u64>>,
    #[case] expected: Option<Range<u64>>,
) {
    assert_eq!(
        split_aligned_row_range(byte_range, &natural_splits(total_size, &split_ranges)),
        expected
    );
}

#[test]
fn test_split_aligned_ranges_cover_splits_exactly_once() {
    let split_ranges = vec![0..1, 1..4, 4..10, 10..13];
    let byte_ranges = [0..4, 4..8, 8..12, 12..16];
    let natural_splits = natural_splits(16, &split_ranges);

    let assigned = byte_ranges
        .into_iter()
        .filter_map(|byte_range| split_aligned_row_range(byte_range, &natural_splits))
        .collect::<Vec<_>>();

    assert_eq!(assigned, vec![0..4, 4..10, 10..13]);
    assert_eq!(
        assigned
            .iter()
            .map(|range| range.end - range.start)
            .sum::<u64>(),
        13
    );

    let split_starts = split_ranges
        .iter()
        .map(|range| range.start)
        .collect::<Vec<_>>();
    let split_ends = split_ranges
        .iter()
        .map(|range| range.end)
        .collect::<Vec<_>>();

    for range in &assigned {
        assert!(split_starts.contains(&range.start));
        assert!(split_ends.contains(&range.end));
    }

    for (left, right) in assigned.iter().tuple_windows() {
        assert_eq!(left.end, right.start);
    }
}

/// Splits whose assignment bytes collide must stay with a single byte range.
#[test]
fn test_split_aligned_row_range_keeps_colliding_assignments_together() {
    let natural_splits = natural_splits(2, &[0..1, 1..2, 2..3, 3..4]);

    assert_eq!(natural_splits.assignment_bytes.as_ref(), [0, 0, 1, 1]);
    assert_eq!(split_aligned_row_range(0..1, &natural_splits), Some(0..2));
    assert_eq!(split_aligned_row_range(1..2, &natural_splits), Some(2..4));
}

#[tokio::test]
async fn test_natural_split_cell_is_shared_by_file_path() {
    let cache = NaturalSplitCache::default();
    let path = Path::from("shared.vortex");
    let first = natural_split_cell_for_file(&cache, &path);
    let second = natural_split_cell_for_file(&cache, &path);
    assert!(Arc::ptr_eq(&first, &second));

    let initializations = Arc::new(AtomicUsize::new(0));
    let first_initializations = Arc::clone(&initializations);
    let second_initializations = Arc::clone(&initializations);
    let first_cell = Arc::clone(&first);
    let second_cell = Arc::clone(&second);

    let (first_ranges, second_ranges) = tokio::join!(
        first_cell.get_or_init(|| async move {
            first_initializations.fetch_add(1, Ordering::Relaxed);
            tokio::task::yield_now().await;
            Arc::new(natural_splits(10, &[0..5, 5..10]))
        }),
        second_cell.get_or_init(|| async move {
            second_initializations.fetch_add(1, Ordering::Relaxed);
            Arc::new(natural_splits(20, &[10..15, 15..20]))
        }),
    );

    assert_eq!(initializations.load(Ordering::Relaxed), 1);
    assert!(Arc::ptr_eq(first_ranges, second_ranges));
    assert_eq!(first_ranges.row_boundaries.as_ref(), [0, 5, 10]);
}

async fn write_arrow_to_vortex(
    object_store: Arc<dyn ObjectStore>,
    path: &str,
    rb: RecordBatch,
) -> anyhow::Result<u64> {
    let array = ArrayRef::from_arrow(rb, false)?;
    let path = Path::parse(path)?;

    let mut write = ObjectStoreWrite::new(object_store, &path).await?;
    let summary = SESSION
        .write_options()
        .write(&mut write, array.to_array_stream())
        .await?;
    write.shutdown().await?;

    Ok(summary.size())
}

fn make_morselizer(
    object_store: Arc<dyn ObjectStore>,
    table_schema: TableSchema,
    filter: Option<PhysicalExprRef>,
) -> VortexMorselizer {
    VortexMorselizer {
        partition: 1,
        session: SESSION.clone(),
        vortex_reader_factory: Arc::new(DefaultVortexReaderFactory::new(object_store)),
        projection: ProjectionExprs::from_indices(&[0], table_schema.file_schema()),
        filter,
        file_pruning_predicate: None,
        expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
        table_schema,
        limit: None,
        metrics_registry: Arc::new(DefaultMetricsRegistry::default()),
        df_metrics: ExecutionPlanMetricsSet::new(),
        layout_readers: Default::default(),
        natural_splits: Default::default(),
        has_output_ordering: false,
        expression_convertor: Arc::new(DefaultExpressionConvertor::default()),
        file_metadata_cache: None,
        projection_pushdown: false,
        scan_concurrency: None,
    }
}

#[tokio::test]
async fn test_open() -> anyhow::Result<()> {
    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let file_path = "part=1/file.vortex";
    let batch = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
    let data_size =
        write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;

    let file_schema = batch.schema();
    let mut file = PartitionedFile::new(file_path.to_string(), data_size);
    file.partition_values = vec![ScalarValue::Int32(Some(1))];

    let table_schema = TableSchema::new(
        Arc::clone(&file_schema),
        vec![Arc::new(Field::new("part", DataType::Int32, false))],
    );

    // filter matches partition value
    let filter = col("part").eq(lit(1));
    let filter = logical2physical(&filter, table_schema.table_schema());

    let morselizer = make_morselizer(
        Arc::clone(&object_store),
        table_schema.clone(),
        Some(filter),
    );
    let stream = morselizer.open(file.clone()).unwrap().await.unwrap();

    let data = stream.try_collect::<Vec<_>>().await?;
    let num_batches = data.len();
    let num_rows = data.iter().map(|rb| rb.num_rows()).sum::<usize>();

    assert_eq!((num_batches, num_rows), (1, 3));

    // filter doesn't matches partition value
    let filter = col("part").eq(lit(2));
    let filter = logical2physical(&filter, table_schema.table_schema());

    let morselizer = make_morselizer(
        Arc::clone(&object_store),
        table_schema.clone(),
        Some(filter),
    );
    let stream = morselizer.open(file.clone()).unwrap().await.unwrap();

    let data = stream.try_collect::<Vec<_>>().await?;
    let num_batches = data.len();
    let num_rows = data.iter().map(|rb| rb.num_rows()).sum::<usize>();
    assert_eq!((num_batches, num_rows), (0, 0));

    Ok(())
}
#[tokio::test]
async fn test_open_preserves_declared_schema_metadata() -> anyhow::Result<()> {
    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let file_path = "part=1/file.vortex";
    let batch = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)]))?;
    let data_size =
        write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;

    let file_schema = Arc::new(
        batch.schema().as_ref().clone().with_metadata(
            [("table".to_string(), "metadata".to_string())]
                .into_iter()
                .collect(),
        ),
    );
    let table_schema = TableSchema::new(
        file_schema,
        vec![Arc::new(
            Field::new("part", DataType::Int32, false).with_metadata(
                [("partition".to_string(), "metadata".to_string())]
                    .into_iter()
                    .collect(),
            ),
        )],
    );
    let projection = ProjectionExprs::from_indices(&[0, 1], table_schema.table_schema());
    let expected_schema = Arc::new(projection.project_schema(table_schema.table_schema())?);

    assert_eq!(
        expected_schema.metadata().get("table"),
        Some(&"metadata".to_string())
    );
    assert_eq!(
        expected_schema.field(1).metadata().get("partition"),
        Some(&"metadata".to_string())
    );

    for projection_pushdown in [false, true] {
        let mut morselizer = make_morselizer(Arc::clone(&object_store), table_schema.clone(), None);
        morselizer.projection = projection.clone();
        morselizer.projection_pushdown = projection_pushdown;

        let mut file = PartitionedFile::new(file_path.to_string(), data_size);
        file.partition_values = vec![ScalarValue::Int32(Some(1))];
        let batches = morselizer
            .open(file)?
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        assert!(!batches.is_empty());
        for batch in batches {
            assert_eq!(batch.schema().as_ref(), expected_schema.as_ref());
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_open_all_valid_nullable_columns_with_nonnullable_table_schema() -> anyhow::Result<()>
{
    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let file_path = "nullable/file.vortex";
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)])),
        vec![Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3)]))],
    )?;
    let data_size = write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch).await?;

    let expected_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let table_schema = TableSchema::from_file_schema(Arc::clone(&expected_schema));

    for projection_pushdown in [false, true] {
        let mut morselizer = make_morselizer(Arc::clone(&object_store), table_schema.clone(), None);
        morselizer.projection_pushdown = projection_pushdown;

        let file = PartitionedFile::new(file_path.to_string(), data_size);
        let batches = morselizer
            .open(file)?
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].schema().as_ref(), expected_schema.as_ref());
    }

    Ok(())
}

#[tokio::test]
async fn test_file_pruning_replaces_partition_columns_without_file_statistics() -> anyhow::Result<()>
{
    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let file_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    let table_schema = TableSchema::new(
        Arc::clone(&file_schema),
        vec![Arc::new(Field::new("part", DataType::Int32, false))],
    );

    let partition_column = Arc::new(df_expr::Column::new("part", 1)) as PhysicalExprRef;
    let predicate = Arc::new(df_expr::BinaryExpr::new(
        Arc::clone(&partition_column),
        Operator::Gt,
        df_expr::lit(ScalarValue::Int32(Some(1))),
    )) as PhysicalExprRef;
    let dynamic_predicate = Arc::new(DynamicFilterPhysicalExpr::new(
        vec![partition_column],
        predicate,
    )) as PhysicalExprRef;

    let mut morselizer = make_morselizer(object_store, table_schema, None);
    morselizer.file_pruning_predicate = Some(dynamic_predicate);
    let df_metrics = morselizer.df_metrics.clone();

    // The file does not exist and has no statistics. Replacing `part` with 1
    // makes the predicate false, so pruning must happen before any file I/O.
    let mut file = PartitionedFile::new("missing.vortex", 1);
    file.partition_values = vec![ScalarValue::Int32(Some(1))];
    let batches = morselizer
        .open(file)?
        .await?
        .try_collect::<Vec<_>>()
        .await?;

    assert!(batches.is_empty());
    assert_eq!(
        df_metrics
            .clone_inner()
            .sum_by_name("num_predicate_creation_errors")
            .map(|metric| metric.as_usize()),
        Some(0)
    );

    Ok(())
}

#[tokio::test]
async fn test_file_pruning_creation_errors_are_reported() -> anyhow::Result<()> {
    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let file_path = "metrics/file.vortex";
    let batch = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
    let data_size =
        write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;
    let mut statistics = Statistics::new_unknown(batch.schema().as_ref());
    statistics.column_statistics[0].null_count = Precision::Exact(0);
    let file = PartitionedFile::new(file_path, data_size).with_statistics(Arc::new(statistics));

    let mut morselizer = make_morselizer(
        object_store,
        TableSchema::from_file_schema(batch.schema()),
        None,
    );
    morselizer.file_pruning_predicate = Some(Arc::new(SnapshotErrorExpr));
    let df_metrics = morselizer.df_metrics.clone();

    let batches = morselizer
        .open(file)?
        .await?
        .try_collect::<Vec<_>>()
        .await?;

    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    assert_eq!(
        df_metrics
            .clone_inner()
            .sum_by_name("num_predicate_creation_errors")
            .map(|metric| metric.as_usize()),
        Some(1)
    );

    Ok(())
}

#[tokio::test]
async fn test_open_applies_limit_after_filtering() -> anyhow::Result<()> {
    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let file_path = "filtered-limit/file.vortex";
    let batch = record_batch!((
        "a",
        Int32,
        vec![Some(1), Some(2), Some(3), Some(4), Some(5), Some(6)]
    ))
    .unwrap();
    let data_size =
        write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;
    let file = PartitionedFile::new(file_path.to_string(), data_size);
    let table_schema = TableSchema::from_file_schema(batch.schema());
    // `a > 3` excludes the first three rows, so a limit applied *before* filtering would take
    // rows [1, 2, 3] and filter them all out (yielding nothing), whereas a limit applied
    // *after* filtering yields the first three matching rows [4, 5, 6]. Asserting the values
    // (not just the count) is what makes this test able to detect a pre-filter regression.
    let filter = logical2physical(&col("a").gt(lit(3_i32)), table_schema.table_schema());

    let mut morselizer = make_morselizer(object_store, table_schema, Some(filter));
    morselizer.limit = Some(3);

    let batches = morselizer
        .open(file)?
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    let values = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("projected column should be Int32")
                .values()
                .to_vec()
        })
        .collect::<Vec<i32>>();

    assert_eq!(values, [4, 5, 6]);

    Ok(())
}

#[tokio::test]
async fn test_open_empty_file() -> anyhow::Result<()> {
    use futures::TryStreamExt;

    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let data_batch = record_batch!(("a", Int32, Vec::<i32>::new())).unwrap();
    let file_path = "part=1/empty.vortex";
    let file_size =
        write_arrow_to_vortex(Arc::clone(&object_store), file_path, data_batch.clone()).await?;

    let file_schema = data_batch.schema();
    // Parallel scans may attach a byte range even for empty files; the
    // morselizer must return early before attempting split-aligned translation.
    let file =
        PartitionedFile::new_with_range(file_path.to_string(), file_size, 0, file_size as i64);

    let table_schema = TableSchema::from_file_schema(Arc::clone(&file_schema));

    let morselizer = make_morselizer(object_store, table_schema, None);
    let stream = morselizer.open(file)?.await?;
    let data = stream.try_collect::<Vec<_>>().await?;

    assert_eq!(data.len(), 0);

    Ok(())
}

#[tokio::test]
async fn test_open_populates_file_metadata_cache() -> anyhow::Result<()> {
    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let file_path = "cached/file.vortex";
    let batch = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
    let data_size =
        write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;

    let file = PartitionedFile::new(file_path.to_string(), data_size);
    let table_schema = TableSchema::from_file_schema(batch.schema());

    let cache: Arc<dyn FileMetadataCache> =
        Arc::new(DefaultFilesMetadataCache::new(64 * 1024 * 1024));
    let mut morselizer = make_morselizer(Arc::clone(&object_store), table_schema, None);
    morselizer.file_metadata_cache = Some(Arc::clone(&cache));

    // The first open misses the cache and must write the parsed footer back.
    let stream = morselizer.open(file.clone())?.await?;
    stream.try_collect::<Vec<_>>().await?;

    let entry = cache
        .get(file.path())
        .ok_or_else(|| anyhow::anyhow!("footer was not cached after open"))?;
    assert!(entry.is_valid_for(&file.object_meta));
    assert!(
        entry
            .file_metadata
            .as_any()
            .downcast_ref::<CachedVortexMetadata>()
            .is_some()
    );

    // The second open hits the cache and still returns the same data.
    let stream = morselizer.open(file.clone())?.await?;
    let data = stream.try_collect::<Vec<_>>().await?;
    assert_eq!(data.iter().map(|rb| rb.num_rows()).sum::<usize>(), 3);

    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_open_files_different_table_schema() -> anyhow::Result<()> {
    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;

    let file1 = {
        let file1_path = "/path/file1.vortex";
        let batch1 = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
        let data_size1 =
            write_arrow_to_vortex(Arc::clone(&object_store), file1_path, batch1).await?;
        PartitionedFile::new(file1_path.to_string(), data_size1)
    };

    let file2 = {
        let file2_path = "/path/file2.vortex";
        let batch2 = record_batch!(("a", Int16, vec![Some(-1), Some(-2), Some(-3)])).unwrap();
        let data_size2 =
            write_arrow_to_vortex(Arc::clone(&object_store), file2_path, batch2).await?;
        PartitionedFile::new(file2_path.to_string(), data_size2)
    };

    // Table schema has can accommodate both files
    let table_schema = TableSchema::from_file_schema(Arc::new(Schema::new(vec![Field::new(
        "a",
        DataType::Int32,
        true,
    )])));

    let make_morselizer = |filter| VortexMorselizer {
        partition: 1,
        session: SESSION.clone(),
        vortex_reader_factory: Arc::new(DefaultVortexReaderFactory::new(Arc::clone(&object_store))),
        projection: ProjectionExprs::from_indices(&[0], table_schema.file_schema()),
        filter: Some(filter),
        file_pruning_predicate: None,
        expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
        table_schema: table_schema.clone(),
        limit: None,
        metrics_registry: Arc::new(DefaultMetricsRegistry::default()),
        df_metrics: ExecutionPlanMetricsSet::new(),
        layout_readers: Default::default(),
        natural_splits: Default::default(),
        has_output_ordering: false,
        expression_convertor: Arc::new(DefaultExpressionConvertor::default()),
        file_metadata_cache: None,
        projection_pushdown: false,
        scan_concurrency: None,
    };

    let filter = col("a").lt(lit(100_i32));
    let filter = logical2physical(&filter, table_schema.table_schema());

    let morselizer1 = make_morselizer(Arc::clone(&filter));
    let stream = morselizer1.open(file1)?.await?;

    let format_opts = FormatOptions::new().with_types_info(true);

    let data = stream.try_collect::<Vec<_>>().await?;
    assert_snapshot!(pretty_format_batches_with_options(&data, &format_opts)?.to_string(), @r"
    +-------+
    | a     |
    | Int32 |
    +-------+
    | 1     |
    | 2     |
    | 3     |
    +-------+
    ");

    let morselizer2 = make_morselizer(Arc::clone(&filter));
    let stream = morselizer2.open(file2)?.await?;

    let data = stream.try_collect::<Vec<_>>().await?;
    assert_snapshot!(pretty_format_batches_with_options(&data, &format_opts)?.to_string(), @r"
    +-------+
    | a     |
    | Int32 |
    +-------+
    | -1    |
    | -2    |
    | -3    |
    +-------+
    ");

    Ok(())
}

#[tokio::test]
// This test verifies that files with different column order than the
// table schema can be opened without errors. The fix ensures that the
// schema mapper is only used for type casting, not for reordering,
// since the vortex projection already handles reordering.
async fn test_schema_different_column_order() -> anyhow::Result<()> {
    use datafusion::arrow::util::pretty::pretty_format_batches_with_options;

    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let file_path = "/path/file.vortex";

    // File has columns in order: c, b, a
    let batch = record_batch!(
        ("c", Int32, vec![Some(300), Some(301), Some(302)]),
        ("b", Int32, vec![Some(200), Some(201), Some(202)]),
        ("a", Int32, vec![Some(100), Some(101), Some(102)])
    )
    .unwrap();
    let data_size = write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch).await?;
    let file = PartitionedFile::new(file_path.to_string(), data_size);

    // Table schema has columns in different order: a, b, c
    let table_schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
        Field::new("c", DataType::Int32, true),
    ]));

    let morselizer = VortexMorselizer {
        partition: 1,
        session: SESSION.clone(),
        vortex_reader_factory: Arc::new(DefaultVortexReaderFactory::new(object_store)),
        projection: ProjectionExprs::from_indices(&[0, 1, 2], &table_schema),
        filter: None,
        file_pruning_predicate: None,
        expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
        table_schema: TableSchema::from_file_schema(Arc::clone(&table_schema)),
        limit: None,
        metrics_registry: Arc::new(DefaultMetricsRegistry::default()),
        df_metrics: ExecutionPlanMetricsSet::new(),
        layout_readers: Default::default(),
        natural_splits: Default::default(),
        has_output_ordering: false,
        expression_convertor: Arc::new(DefaultExpressionConvertor::default()),
        file_metadata_cache: None,
        projection_pushdown: false,
        scan_concurrency: None,
    };

    let stream = morselizer.open(file)?.await?;

    let format_opts = FormatOptions::new().with_types_info(true);
    let data = stream.try_collect::<Vec<_>>().await?;

    // Verify the output has columns in table schema order (a, b, c)
    // not file order (c, b, a)
    assert_snapshot!(pretty_format_batches_with_options(&data, &format_opts)?.to_string(), @r"
    +-------+-------+-------+
    | a     | b     | c     |
    | Int32 | Int32 | Int32 |
    +-------+-------+-------+
    | 100   | 200   | 300   |
    | 101   | 201   | 301   |
    | 102   | 202   | 302   |
    +-------+-------+-------+
    ");

    Ok(())
}

#[tokio::test]
// This test verifies that expression rewriting doesn't fail when there is
// a nested schema mismatch between the physical file schema and logical
// table schema.
async fn test_adapter_logical_physical_struct_mismatch() -> anyhow::Result<()> {
    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let file_path = "/path/file.vortex";
    let file_struct_fields = Fields::from(vec![
        Field::new("field1", DataType::Utf8, true),
        Field::new("field2", DataType::Utf8, true),
    ]);
    let struct_array = StructArray::new(
        file_struct_fields.clone(),
        vec![
            Arc::new(StringArray::from(vec!["value1", "value2", "value3"])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
        None,
    );
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "my_struct",
            DataType::Struct(file_struct_fields),
            true,
        )])),
        vec![Arc::new(struct_array)],
    )?;
    let data_size = write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch).await?;

    // Table schema has an extra utf8 field.
    let table_schema = TableSchema::from_file_schema(Arc::new(Schema::new(vec![Field::new(
        "my_struct",
        DataType::Struct(Fields::from(vec![
            Field::new(
                "field1",
                DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                true,
            ),
            Field::new(
                "field2",
                DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                true,
            ),
            Field::new("field3", DataType::Utf8, true),
        ])),
        true,
    )])));

    let morselizer = make_morselizer(
        Arc::clone(&object_store),
        table_schema.clone(),
        // expression references my_struct column which has different fields in each
        // field.
        Some(logical2physical(
            &col("my_struct").is_not_null(),
            table_schema.table_schema(),
        )),
    );

    // The morselizer should be able to open the file with a filter on the
    // struct column.
    let data = morselizer
        .open(PartitionedFile::new(file_path.to_string(), data_size))?
        .await?
        .try_collect::<Vec<_>>()
        .await?;

    assert_eq!(data.len(), 1);
    assert_eq!(data[0].num_rows(), 3);

    Ok(())
}

#[tokio::test]
// Minimal reproducing test for the schema projection bug.
// Before the fix, this would fail with a cast error when the file schema
// and table schema have different field orders and we project a subset of columns.
async fn test_projection_bug_minimal_repro() -> anyhow::Result<()> {
    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let file_path = "/path/file.vortex";

    // File has columns in order: a, b, c with simple types
    let batch = record_batch!(
        ("a", Int32, vec![Some(1)]),
        ("b", Utf8, vec![Some("test")]),
        ("c", Int32, vec![Some(2)])
    )
    .unwrap();
    let data_size = write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch).await?;

    // Table schema has columns in DIFFERENT order: c, a, b
    // and different types that require casting (Utf8 -> Dictionary)
    let table_schema = TableSchema::new(
        Arc::new(Schema::new(vec![
            Field::new("c", DataType::Int32, true),
            Field::new("a", DataType::Int32, true),
            Field::new(
                "b",
                DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                true,
            ),
        ])),
        vec![],
    );

    // Project columns [0, 2] from table schema, which should give us: c, b
    // Before the fix, the schema adapter would get confused about which fields
    // to select from the file, causing incorrect type mappings.
    let projection = vec![0, 2];

    let morselizer = VortexMorselizer {
        partition: 1,
        session: SESSION.clone(),
        vortex_reader_factory: Arc::new(DefaultVortexReaderFactory::new(Arc::clone(&object_store))),
        projection: ProjectionExprs::from_indices(projection.as_ref(), table_schema.file_schema()),
        filter: None,
        file_pruning_predicate: None,
        expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
        table_schema: table_schema.clone(),
        limit: None,
        metrics_registry: Arc::new(DefaultMetricsRegistry::default()),
        df_metrics: ExecutionPlanMetricsSet::new(),
        layout_readers: Default::default(),
        natural_splits: Default::default(),
        has_output_ordering: false,
        expression_convertor: Arc::new(DefaultExpressionConvertor::default()),
        file_metadata_cache: None,
        projection_pushdown: false,
        scan_concurrency: None,
    };

    // This should succeed and return the correctly projected and cast data
    let data = morselizer
        .open(PartitionedFile::new(file_path.to_string(), data_size))?
        .await?
        .try_collect::<Vec<_>>()
        .await?;

    // Verify the columns are in the right order and have the right values
    use datafusion::arrow::util::pretty::pretty_format_batches_with_options;
    let format_opts = FormatOptions::new().with_types_info(true);
    assert_snapshot!(pretty_format_batches_with_options(&data, &format_opts)?.to_string(), @r"
    +-------+--------------------------+
    | c     | b                        |
    | Int32 | Dictionary(UInt32, Utf8) |
    +-------+--------------------------+
    | 2     | test                     |
    +-------+--------------------------+
    ");

    Ok(())
}

fn make_test_batch_with_10_rows() -> RecordBatch {
    record_batch!(
        ("a", Int32, (0..=9).map(Some).collect::<Vec<_>>()),
        (
            "b",
            Utf8,
            (0..=9).map(|i| Some(format!("r{}", i))).collect::<Vec<_>>()
        )
    )
    .unwrap()
}

fn make_test_morselizer(
    object_store: Arc<dyn ObjectStore>,
    schema: SchemaRef,
    projection: ProjectionExprs,
) -> VortexMorselizer {
    VortexMorselizer {
        partition: 1,
        session: SESSION.clone(),
        vortex_reader_factory: Arc::new(DefaultVortexReaderFactory::new(object_store)),
        projection,
        filter: None,
        file_pruning_predicate: None,
        expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
        table_schema: TableSchema::from_file_schema(schema),
        limit: None,
        metrics_registry: Arc::new(DefaultMetricsRegistry::default()),
        df_metrics: ExecutionPlanMetricsSet::new(),
        layout_readers: Default::default(),
        natural_splits: Default::default(),
        has_output_ordering: false,
        expression_convertor: Arc::new(DefaultExpressionConvertor::default()),
        file_metadata_cache: None,
        projection_pushdown: false,
        scan_concurrency: None,
    }
}

#[tokio::test]
// Test that Selection::IncludeByIndex filters to specific row indices.
async fn test_selection_include_by_index() -> anyhow::Result<()> {
    use datafusion::arrow::util::pretty::pretty_format_batches_with_options;
    use vortex::buffer::Buffer;
    use vortex::scan::selection::Selection;

    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let file_path = "/path/file.vortex";

    let batch = make_test_batch_with_10_rows();
    let data_size =
        write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;

    let schema = batch.schema();
    let mut file = PartitionedFile::new(file_path.to_string(), data_size);
    file.extensions.insert(
        VortexAccessPlan::default().with_selection(Selection::IncludeByIndex(Buffer::from_iter(
            vec![1, 3, 5, 7],
        ))),
    );

    let morselizer = make_test_morselizer(
        Arc::clone(&object_store),
        Arc::clone(&schema),
        ProjectionExprs::from_indices(&[0, 1], &schema),
    );

    let stream = morselizer.open(file)?.await?;
    let data = stream.try_collect::<Vec<_>>().await?;
    let format_opts = FormatOptions::new().with_types_info(true);

    assert_snapshot!(pretty_format_batches_with_options(&data, &format_opts)?.to_string(), @r"
    +-------+------+
    | a     | b    |
    | Int32 | Utf8 |
    +-------+------+
    | 1     | r1   |
    | 3     | r3   |
    | 5     | r5   |
    | 7     | r7   |
    +-------+------+
    ");

    Ok(())
}

#[tokio::test]
// Test that Selection::ExcludeByIndex excludes specific row indices.
async fn test_selection_exclude_by_index() -> anyhow::Result<()> {
    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let file_path = "/path/file.vortex";

    let batch = make_test_batch_with_10_rows();
    let data_size =
        write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;

    let schema = batch.schema();
    let mut file = PartitionedFile::new(file_path.to_string(), data_size);
    file.extensions.insert(
        VortexAccessPlan::default().with_selection(Selection::ExcludeByIndex(Buffer::from_iter(
            vec![0, 2, 4, 6, 8],
        ))),
    );

    let morselizer = make_test_morselizer(
        Arc::clone(&object_store),
        Arc::clone(&schema),
        ProjectionExprs::from_indices(&[0, 1], &schema),
    );

    let stream = morselizer.open(file)?.await?;
    let data = stream.try_collect::<Vec<_>>().await?;
    let format_opts = FormatOptions::new().with_types_info(true);

    assert_snapshot!(pretty_format_batches_with_options(&data, &format_opts)?.to_string(), @r"
    +-------+------+
    | a     | b    |
    | Int32 | Utf8 |
    +-------+------+
    | 1     | r1   |
    | 3     | r3   |
    | 5     | r5   |
    | 7     | r7   |
    | 9     | r9   |
    +-------+------+
    ");

    Ok(())
}

#[tokio::test]
// Test that Selection::All returns all rows.
async fn test_selection_all() -> anyhow::Result<()> {
    use vortex::scan::selection::Selection;

    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let file_path = "/path/file.vortex";

    let batch = make_test_batch_with_10_rows();
    let data_size =
        write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;

    let schema = batch.schema();
    let mut file = PartitionedFile::new(file_path.to_string(), data_size);
    file.extensions
        .insert(VortexAccessPlan::default().with_selection(Selection::All));

    let morselizer = make_test_morselizer(
        Arc::clone(&object_store),
        Arc::clone(&schema),
        ProjectionExprs::from_indices(&[0], &schema),
    );

    let stream = morselizer.open(file)?.await?;
    let data = stream.try_collect::<Vec<_>>().await?;

    let total_rows: usize = data.iter().map(|rb| rb.num_rows()).sum();
    assert_eq!(total_rows, 10);

    Ok(())
}

#[tokio::test]
// Test that when no extensions are provided, all rows are returned (backward compatibility).
async fn test_selection_no_extensions() -> anyhow::Result<()> {
    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let file_path = "/path/file.vortex";

    let batch = make_test_batch_with_10_rows();
    let data_size =
        write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;

    let schema = batch.schema();
    let file = PartitionedFile::new(file_path.to_string(), data_size);
    // file.extensions is None by default

    let morselizer = make_test_morselizer(
        Arc::clone(&object_store),
        Arc::clone(&schema),
        ProjectionExprs::from_indices(&[0], &schema),
    );

    let stream = morselizer.open(file)?.await?;
    let data = stream.try_collect::<Vec<_>>().await?;

    let total_rows: usize = data.iter().map(|rb| rb.num_rows()).sum();
    assert_eq!(total_rows, 10);

    Ok(())
}

#[tokio::test]
async fn test_projection_expr_pushdown() -> anyhow::Result<()> {
    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
    let file_path = "/path/file.vortex";

    let batch = record_batch!(
        ("a", Int32, vec![Some(1), Some(2), Some(3)]),
        ("b", Int32, vec![Some(10), Some(20), Some(30)])
    )
    .unwrap();
    let data_size =
        write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;

    let file_schema = batch.schema();
    let table_schema = TableSchema::from_file_schema(Arc::clone(&file_schema));

    // Create a projection that includes an arithmetic expression: a + b * 2
    let col_a = df_expr::col("a", &file_schema)?;
    let col_b = df_expr::col("b", &file_schema)?;
    let two = df_expr::lit(ScalarValue::Int32(Some(2)));

    // b * 2
    let b_times_2 = df_expr::binary(col_b, Operator::Multiply, two, &file_schema)?;
    // a + (b * 2)
    let a_plus_b_times_2 = df_expr::binary(col_a, Operator::Plus, b_times_2, &file_schema)?;

    let projection = ProjectionExprs::new(vec![ProjectionExpr::new(
        a_plus_b_times_2,
        "result".to_string(),
    )]);

    let morselizer = VortexMorselizer {
        partition: 1,
        session: SESSION.clone(),
        vortex_reader_factory: Arc::new(DefaultVortexReaderFactory::new(Arc::clone(&object_store))),
        projection,
        filter: None,
        file_pruning_predicate: None,
        expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
        table_schema,
        limit: None,
        metrics_registry: Arc::new(DefaultMetricsRegistry::default()),
        df_metrics: ExecutionPlanMetricsSet::new(),
        layout_readers: Default::default(),
        natural_splits: Default::default(),
        has_output_ordering: false,
        expression_convertor: Arc::new(DefaultExpressionConvertor::default()),
        file_metadata_cache: None,
        projection_pushdown: false,
        scan_concurrency: None,
    };

    let file = PartitionedFile::new(file_path.to_string(), data_size);
    let stream = morselizer.open(file)?.await?;
    let data = stream.try_collect::<Vec<_>>().await?;

    // Expected: a + b * 2
    // row 0: 1 + 10 * 2 = 21
    // row 1: 2 + 20 * 2 = 42
    // row 2: 3 + 30 * 2 = 63
    assert_snapshot!(pretty_format_batches_with_options(&data, &FormatOptions::new().with_types_info(true))?.to_string(), @r"
    +--------+
    | result |
    | Int32  |
    +--------+
    | 21     |
    | 42     |
    | 63     |
    +--------+
    ");

    Ok(())
}

/// When a Struct contains Dictionary fields, writing to vortex and reading back
/// should preserve the Dictionary type.
#[tokio::test]
async fn test_struct_with_dictionary_roundtrip() -> anyhow::Result<()> {
    let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;

    let struct_fields = Fields::from(vec![
        Field::new_dictionary("a", DataType::UInt32, DataType::Utf8, true),
        Field::new_dictionary("b", DataType::UInt32, DataType::Utf8, true),
    ]);
    let struct_array = StructArray::new(
        struct_fields.clone(),
        vec![
            Arc::new(DictionaryArray::<UInt32Type>::from_iter(["x", "y", "x"])),
            Arc::new(DictionaryArray::<UInt32Type>::from_iter(["p", "p", "q"])),
        ],
        None,
    );

    let schema = Arc::new(Schema::new(vec![Field::new(
        "labels",
        DataType::Struct(struct_fields.clone()),
        false,
    )]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(struct_array)])?;

    let file_path = "/test.vortex";
    let data_size = write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch).await?;

    let morselizer = make_test_morselizer(
        Arc::clone(&object_store),
        Arc::clone(&schema),
        ProjectionExprs::from_indices(&[0], &schema),
    );
    let data: Vec<_> = morselizer
        .open(PartitionedFile::new(file_path.to_string(), data_size))?
        .await?
        .try_collect()
        .await?;

    assert_eq!(
        data[0].schema().field(0).data_type(),
        &DataType::Struct(struct_fields),
        "Struct(Dictionary) type should be preserved"
    );
    Ok(())
}
