// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::{Arc, Weak};

use arrow_schema::{ArrowError, Field, SchemaRef};
use datafusion_common::{DataFusionError, Result as DFResult};
use datafusion_datasource::file_meta::FileMeta;
use datafusion_datasource::file_stream::{FileOpenFuture, FileOpener};
use datafusion_datasource::schema_adapter::SchemaAdapterFactory;
use datafusion_datasource::{FileRange, PartitionedFile};
use datafusion_physical_expr::schema_rewriter::PhysicalExprAdapterFactory;
use datafusion_physical_expr::simplifier::PhysicalExprSimplifier;
use datafusion_physical_expr::{PhysicalExprRef, split_conjunction};
use futures::{FutureExt, StreamExt, TryStreamExt, stream};
use object_store::ObjectStore;
use object_store::path::Path;
use vortex::dtype::FieldName;
use vortex::error::VortexError;
use vortex::expr::{root, select};
use vortex::layout::LayoutReader;
use vortex::metrics::VortexMetrics;
use vortex::scan::ScanBuilder;
use vortex::{ArrayRef, ToCanonical};
use vortex_utils::aliases::dash_map::{DashMap, Entry};

use super::cache::VortexFileCache;
use crate::convert::exprs::{can_be_pushed_down, make_vortex_predicate};

#[derive(Clone)]
pub(crate) struct VortexOpener {
    pub object_store: Arc<dyn ObjectStore>,
    /// Projection by index of the file's columns
    pub projection: Option<Arc<[usize]>>,
    pub filter: Option<PhysicalExprRef>,
    pub expr_adapter_factory: Option<Arc<dyn PhysicalExprAdapterFactory>>,
    pub schema_adapter_factory: Arc<dyn SchemaAdapterFactory>,
    /// Hive-style partitioning columns
    pub partition_fields: Vec<Arc<Field>>,
    pub file_cache: VortexFileCache,
    /// This is the table's schema without partition columns. It might be different than
    /// the physical schema, and the stream's type will be a projection of it.
    pub logical_schema: SchemaRef,
    pub batch_size: usize,
    pub limit: Option<usize>,
    pub metrics: VortexMetrics,
    pub layout_readers: Arc<DashMap<Path, Weak<dyn LayoutReader>>>,
}

impl FileOpener for VortexOpener {
    fn open(&self, file_meta: FileMeta, file: PartitionedFile) -> DFResult<FileOpenFuture> {
        let object_store = self.object_store.clone();
        let projection = self.projection.clone();
        let mut filter = self.filter.clone();
        let expr_adapter_factory = self.expr_adapter_factory.clone();
        let partition_fields = self.partition_fields.clone();
        let file_cache = self.file_cache.clone();
        let logical_schema = self.logical_schema.clone();
        let batch_size = self.batch_size;
        let limit = self.limit;
        let metrics = self.metrics.clone();
        let layout_reader = self.layout_readers.clone();

        let projected_schema = match projection.as_ref() {
            None => logical_schema.clone(),
            Some(indices) => Arc::new(logical_schema.project(indices)?),
        };

        let mut predicate_file_schema = logical_schema.clone();

        let schema_adapter = self
            .schema_adapter_factory
            .create(projected_schema, logical_schema.clone());

        Ok(async move {
            let vxf = file_cache
                .try_get(&file_meta.object_meta, object_store)
                .await
                .map_err(|e| {
                    DataFusionError::Execution(format!("Failed to open Vortex file {e}"))
                })?;

            let physical_file_schema = Arc::new(vxf.dtype().to_arrow_schema().map_err(|e| {
                DataFusionError::Execution(format!("Failed to convert file schema to arrow: {e}"))
            })?);

            if let Some(expr_adapter_factory) = expr_adapter_factory {
                let partition_values = partition_fields
                    .iter()
                    .cloned()
                    .zip(file.partition_values)
                    .collect::<Vec<_>>();

                // The adapter rewrites the expression to the local file schema, allowing
                // for schema evolution and divergence between the table's schema and individual files.
                filter = filter
                    .map(|filter| {
                        let expr = expr_adapter_factory
                            .create(logical_schema.clone(), physical_file_schema.clone())
                            .with_partition_values(partition_values)
                            .rewrite(filter)?;

                        // Expression might now reference columns that don't exist in the file, so we can give it
                        // another simplification pass.
                        PhysicalExprSimplifier::new(&physical_file_schema).simplify(expr)
                    })
                    .transpose()?;

                predicate_file_schema = physical_file_schema.clone();
            }

            let (schema_mapping, adapted_projections) =
                schema_adapter.map_schema(&physical_file_schema)?;

            let fields = adapted_projections
                .iter()
                .map(|idx| {
                    let field = logical_schema.field(*idx);
                    FieldName::from(field.name().as_str())
                })
                .collect::<Vec<_>>();
            let projection_expr = select(fields, root());

            // We share our layout readers with others partitions in the scan, so we can only need to read each layout in each file once.
            let layout_reader = match layout_reader.entry(file_meta.object_meta.location.clone()) {
                Entry::Occupied(mut occupied_entry) => {
                    if let Some(reader) = occupied_entry.get().upgrade() {
                        log::trace!("reusing layout reader for {}", occupied_entry.key());
                        reader
                    } else {
                        log::trace!("creating layout reader for {}", occupied_entry.key());
                        let reader = vxf.layout_reader().map_err(|e| {
                            DataFusionError::Execution(format!(
                                "Failed to create layout reader: {e}"
                            ))
                        })?;
                        occupied_entry.insert(Arc::downgrade(&reader));
                        reader
                    }
                }
                Entry::Vacant(vacant_entry) => {
                    log::trace!("creating layout reader for {}", vacant_entry.key());
                    let reader = vxf.layout_reader().map_err(|e| {
                        DataFusionError::Execution(format!("Failed to create layout reader: {e}"))
                    })?;
                    vacant_entry.insert(Arc::downgrade(&reader));

                    reader
                }
            };

            let mut scan_builder = ScanBuilder::new(layout_reader);
            if let Some(file_range) = file_meta.range {
                scan_builder = apply_byte_range(
                    file_range,
                    file_meta.object_meta.size,
                    vxf.row_count(),
                    scan_builder,
                );
            }

            let filter = filter
                .and_then(|f| {
                    let exprs = split_conjunction(&f)
                        .into_iter()
                        .filter(|expr| can_be_pushed_down(expr, &predicate_file_schema))
                        .collect::<Vec<_>>();

                    make_vortex_predicate(&exprs).transpose()
                })
                .transpose()
                .map_err(|e| DataFusionError::External(e.into()))?;

            if let Some(limit) = limit
                && filter.is_none()
            {
                scan_builder = scan_builder.with_limit(limit);
            }

            let stream = scan_builder
                .with_metrics(metrics)
                .with_projection(projection_expr)
                .with_some_filter(filter)
                .map(|chunk| chunk.to_struct().into_record_batch())
                .into_tokio_stream()
                .map_err(|e| {
                    DataFusionError::Execution(format!("Failed to create Vortex stream: {e}"))
                })?
                .map_ok(move |rb| {
                    // We try and slice the stream into respecting datafusion's configured batch size.
                    stream::iter(
                        (0..rb.num_rows().div_ceil(batch_size * 2))
                            .flat_map(move |block_idx| {
                                let offset = block_idx * batch_size * 2;

                                // If we have less than two batches worth of rows left, we keep them together as a single batch.
                                if rb.num_rows() - offset < 2 * batch_size {
                                    let length = rb.num_rows() - offset;
                                    [Some(rb.slice(offset, length)), None].into_iter()
                                } else {
                                    let first = rb.slice(offset, batch_size);
                                    let second = rb.slice(offset + batch_size, batch_size);
                                    [Some(first), Some(second)].into_iter()
                                }
                            })
                            .flatten()
                            .map(Ok),
                    )
                })
                .map_err(move |e: VortexError| {
                    ArrowError::ExternalError(Box::new(e.with_context(format!(
                        "Failed to read Vortex file: {}",
                        file_meta.object_meta.location
                    ))))
                })
                .try_flatten()
                .map(move |batch| {
                    batch.and_then(|b| schema_mapping.map_batch(b).map_err(Into::into))
                })
                .boxed();

            Ok(stream)
        }
        .boxed())
    }
}

/// If the file has a [`FileRange`](datafusion::datasource::listing::FileRange), we translate it into a row range in the file for the scan.
fn apply_byte_range(
    file_range: FileRange,
    total_size: u64,
    row_count: u64,
    scan_builder: ScanBuilder<ArrayRef>,
) -> ScanBuilder<ArrayRef> {
    let row_range = byte_range_to_row_range(
        file_range.start as u64..file_range.end as u64,
        row_count,
        total_size,
    );

    scan_builder.with_row_range(row_range)
}

fn byte_range_to_row_range(byte_range: Range<u64>, row_count: u64, total_size: u64) -> Range<u64> {
    let average_row = total_size / row_count;
    assert!(average_row > 0, "A row must always have at least one byte");

    let start_row = byte_range.start / average_row;
    let end_row = byte_range.end / average_row;

    // We take the min here as `end_row` might overshoot
    start_row..u64::min(row_count, end_row)
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use datafusion::arrow;
    use datafusion::arrow::array::RecordBatch;
    use datafusion::arrow::datatypes::{DataType, Schema};
    use datafusion::arrow::util::pretty::print_batches;
    use datafusion::common::record_batch;
    use datafusion::datasource::schema_adapter::DefaultSchemaAdapterFactory;
    use datafusion::logical_expr::{col, lit};
    use datafusion::physical_expr::planner::logical2physical;
    use datafusion::physical_expr::schema_rewriter::DefaultPhysicalExprAdapterFactory;
    use datafusion::scalar::ScalarValue;
    use futures::stream::BoxStream;
    use itertools::Itertools;
    use object_store::ObjectMeta;
    use object_store::memory::InMemory;
    use rstest::rstest;
    use vortex::arrow::FromArrowArray;
    use vortex::file::VortexWriteOptions;
    use vortex::session::VortexSession;

    use super::*;

    #[rstest]
    #[case(0..100, 100, 100, 0..100)]
    #[case(0..105, 100, 105, 0..100)]
    #[case(0..50, 100, 105, 0..50)]
    #[case(50..105, 100, 105, 50..100)]
    #[case(0..1, 4, 8, 0..0)]
    #[case(1..8, 4, 8, 0..4)]
    fn test_range_translation(
        #[case] byte_range: Range<u64>,
        #[case] row_count: u64,
        #[case] total_size: u64,
        #[case] expected: Range<u64>,
    ) {
        assert_eq!(
            byte_range_to_row_range(byte_range, row_count, total_size),
            expected
        );
    }

    #[test]
    fn test_consecutive_ranges() {
        let row_count = 100;
        let total_size = 429;
        let bytes_a = 0..143;
        let bytes_b = 143..286;
        let bytes_c = 286..429;

        let rows_a = byte_range_to_row_range(bytes_a, row_count, total_size);
        let rows_b = byte_range_to_row_range(bytes_b, row_count, total_size);
        let rows_c = byte_range_to_row_range(bytes_c, row_count, total_size);

        assert_eq!(rows_a.end - rows_a.start, 35);
        assert_eq!(rows_b.end - rows_b.start, 36);
        assert_eq!(rows_c.end - rows_c.start, 29);

        assert_eq!(rows_a.start, 0);
        assert_eq!(rows_c.end, 100);
        for (left, right) in [rows_a, rows_b, rows_c].iter().tuple_windows() {
            assert_eq!(left.end, right.start);
        }
    }

    async fn write_arrow_to_vortex(
        object_store: Arc<dyn ObjectStore>,
        path: &str,
        rb: RecordBatch,
    ) -> anyhow::Result<u64> {
        let array = ArrayRef::from_arrow(rb, false);
        let path = Path::parse(path)?;

        VortexWriteOptions::default()
            .write_object_store(&object_store, &path, array.to_array_stream())
            .await?;

        Ok(object_store.head(&path).await?.size)
    }

    async fn count_data(
        mut stream: BoxStream<'static, Result<RecordBatch, ArrowError>>,
    ) -> anyhow::Result<(usize, usize)> {
        let mut batches = vec![];

        while let Some(rb) = stream.next().await {
            let rb = rb?;

            batches.push(rb);
        }

        print_batches(&batches)?;
        let num_rows = batches.iter().map(|v| v.num_rows()).sum::<usize>();
        Ok((batches.len(), num_rows))
    }

    fn make_meta(path: &str, data_size: u64) -> FileMeta {
        FileMeta {
            object_meta: ObjectMeta {
                location: Path::from(path),
                last_modified: Utc::now(),
                size: data_size,
                e_tag: None,
                version: None,
            },
            range: None,
            extensions: None,
            metadata_size_hint: None,
        }
    }

    #[rstest]
    #[case(Some(Arc::new(DefaultPhysicalExprAdapterFactory) as _), (1, 3), (0, 0))]
    // If we don't have a physical expr adapter, we just drop filters on partition values
    #[case(None, (1, 3), (1, 3))]
    #[tokio::test]
    async fn test_adapter_optimization_partition_column(
        #[case] expr_adapter_factory: Option<Arc<dyn PhysicalExprAdapterFactory>>,
        #[case] expected_result1: (usize, usize),
        #[case] expected_result2: (usize, usize),
    ) -> anyhow::Result<()> {
        let vx_session = Arc::new(VortexSession::default());
        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file_path = "part=1/file.vortex";
        let batch = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
        let data_size =
            write_arrow_to_vortex(object_store.clone(), file_path, batch.clone()).await?;

        let file_schema = batch.schema();
        let mut file = PartitionedFile::new(file_path.to_string(), data_size);
        file.partition_values = vec![ScalarValue::Int32(Some(1))];

        let table_schema = Arc::new(Schema::new(vec![
            Field::new("part", DataType::Int32, false),
            Field::new("a", DataType::Int32, false),
        ]));

        let make_opener = |filter| VortexOpener {
            object_store: object_store.clone(),
            projection: Some([0].into()),
            filter: Some(filter),
            expr_adapter_factory: expr_adapter_factory.clone(),
            schema_adapter_factory: Arc::new(DefaultSchemaAdapterFactory),
            partition_fields: vec![Arc::new(Field::new("part", DataType::Int32, false))],
            file_cache: VortexFileCache::new(1, 1, vx_session.clone()),
            logical_schema: file_schema.clone(),
            batch_size: 100,
            limit: None,
            metrics: Default::default(),
            layout_readers: Default::default(),
        };

        // filter matches partition value
        let filter = col("part").eq(lit(1));
        let filter = logical2physical(&filter, table_schema.as_ref());

        let opener = make_opener(filter);
        let stream = opener
            .open(make_meta(file_path, data_size), file.clone())
            .unwrap()
            .await
            .unwrap();
        let (num_batches, num_rows) = count_data(stream).await?;
        assert_eq!((num_batches, num_rows), expected_result1);

        // filter doesn't matches partition value
        let filter = col("part").eq(lit(2));
        let filter = logical2physical(&filter, table_schema.as_ref());

        let opener = make_opener(filter);
        let stream = opener
            .open(make_meta(file_path, data_size), file.clone())
            .unwrap()
            .await
            .unwrap();
        let (num_batches, num_rows) = count_data(stream).await?;
        assert_eq!((num_batches, num_rows), expected_result2);

        Ok(())
    }

    #[rstest]
    #[case(Some(Arc::new(DefaultPhysicalExprAdapterFactory) as _))]
    // If we don't have a physical expr adapter, we just drop filters on partition values.
    // This is currently not supported, the work to support it requires to rewrite the predicate with appropriate casts.
    // Seems like datafusion is moving towards having DefaultPhysicalExprAdapterFactory be always provided, which would make it work OOTB.
    // See: https://github.com/apache/datafusion/issues/16800
    // #[case(None)]
    #[tokio::test]
    async fn test_open_files_different_table_schema(
        #[case] expr_adapter_factory: Option<Arc<dyn PhysicalExprAdapterFactory>>,
    ) -> anyhow::Result<()> {
        let vx_session = Arc::new(VortexSession::default());
        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file1_path = "/path/file1.vortex";
        let batch1 = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
        let data_size1 = write_arrow_to_vortex(object_store.clone(), file1_path, batch1).await?;
        let file1 = PartitionedFile::new(file1_path.to_string(), data_size1);

        let file2_path = "/path/file2.vortex";
        let batch2 = record_batch!(("a", Int16, vec![Some(-1), Some(-2), Some(-3)])).unwrap();
        let data_size2 = write_arrow_to_vortex(object_store.clone(), file2_path, batch2).await?;
        let file2 = PartitionedFile::new(file1_path.to_string(), data_size1);

        // Table schema has can accommodate both files
        let table_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));

        let make_opener = |filter| VortexOpener {
            object_store: object_store.clone(),
            projection: Some([0].into()),
            filter: Some(filter),
            expr_adapter_factory: expr_adapter_factory.clone(),
            schema_adapter_factory: Arc::new(DefaultSchemaAdapterFactory),
            partition_fields: vec![],
            file_cache: VortexFileCache::new(1, 1, vx_session.clone()),
            logical_schema: table_schema.clone(),
            batch_size: 100,
            limit: None,
            metrics: Default::default(),
            layout_readers: Default::default(),
        };

        let filter = col("a").lt(lit(100_i32));
        let filter = logical2physical(&filter, table_schema.as_ref());

        let opener1 = make_opener(filter.clone());
        let stream = opener1
            .open(make_meta(file1_path, data_size1), file1)
            .unwrap()
            .await
            .unwrap();
        let (num_batches, num_rows) = count_data(stream).await?;
        assert_eq!(num_batches, 1);
        assert_eq!(num_rows, 3);

        let opener2 = make_opener(filter.clone());
        let stream = opener2
            .open(make_meta(file2_path, data_size2), file2)
            .unwrap()
            .await
            .unwrap();
        let (num_batches, num_rows) = count_data(stream).await?;
        assert_eq!(num_batches, 1);
        assert_eq!(num_rows, 3);

        Ok(())
    }
}
